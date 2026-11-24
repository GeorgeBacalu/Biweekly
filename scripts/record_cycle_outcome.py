"""Prepare a Season 2 Cycle 1 handoff and record operator-observed outcomes."""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath, PureWindowsPath
from time import sleep
from uuid import uuid4

if os.name == "nt":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    LOCK_VIOLATION = 33
    lock_file = ctypes.WinDLL("kernel32", use_last_error=True).LockFile
    lock_file.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD)
    lock_file.restype = wintypes.BOOL
else:
    import fcntl

if __package__:
    from .validate_walk_forward import CITY_MAP_PAGE, CITY_MIN_DISTANCE, PHASES, ROOT, atomic_json, finite_number, utc_now
else:
    from validate_walk_forward import CITY_MAP_PAGE, CITY_MIN_DISTANCE, PHASES, ROOT, atomic_json, finite_number, utc_now


SEASON = 2
CYCLE = 1
SCHEMA_VERSION = 1
HANDOFF_ROOT = ROOT / "_bmad-output" / "evidence" / "submissions" / f"season-{SEASON}-cycle-{CYCLE}"
ATTEMPT_REPORT = "attempt.json"
NOVELTY_REPORT = "novelty.json"
DECISION_REPORT = "decision.json"
HANDOFF_REPORT = "handoff.json"
OBSERVATIONS = "observations"
SHA256 = re.compile(r"[0-9a-f]{64}")
OBSERVATION_FILENAME = re.compile(r"[0-9a-f]{32}\.json")


def read_local_file(directory, basename):
    path = directory / basename
    if path.resolve() != path:
        raise ValueError(f"{basename} must be a file directly in {directory}")
    if not path.is_file():
        raise ValueError(f"Missing or invalid evidence file: {path}")
    return path.read_bytes()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def json_object(data, filename):
    try:
        report = json.loads(data.decode("utf-8"), object_pairs_hook=unique_object)
    except ValueError as exc:
        raise ValueError(f"Invalid JSON in {filename}: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError(f"{filename} must contain a JSON object")
    return report


def object_field(report, key, filename):
    value = report.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid {key} in {filename}")
    return value


def require_integer(report, field, expected, filename):
    value = report.get(field)
    if type(value) is not int or value != expected:
        raise ValueError(f"{field} in {filename} must be the integer {expected}")


def require_utc_timestamp(report, field, filename):
    try:
        timestamp = datetime.fromisoformat(report.get(field))
    except (TypeError, ValueError):
        timestamp = None
    if timestamp is None or timestamp.utcoffset() != timedelta(0):
        raise ValueError(f"{field} in {filename} must be a valid UTC timestamp")
    return timestamp


def candidate_filename(name):
    if (not isinstance(name, str) or Path(name).suffix.lower() != ".py"
            or any(separator in name for separator in ("/", "\\", ":"))):
        raise ValueError("The candidate filename must be a .py basename")
    return name


def verify_ready_decision(decision, attempt_id, digest):
    if decision.get("status") != "ready" or decision.get("attempt_id") != attempt_id:
        raise ValueError(f"{DECISION_REPORT} must contain a ready decision for attempt {attempt_id}")
    require_integer(decision, "schema_version", SCHEMA_VERSION, DECISION_REPORT)
    decided_at = require_utc_timestamp(decision, "decided_at", DECISION_REPORT)
    if decision.get("reasons") != []:
        raise ValueError(f"The ready decision in {DECISION_REPORT} must have an empty reasons list")
    retained = object_field(decision, "candidate_at_decision", DECISION_REPORT)
    if decision.get("candidate_sha256") != digest or retained.get("sha256") != digest:
        raise ValueError(f"Candidate SHA-256 mismatch in {DECISION_REPORT}")
    if decision.get("cap2_report") != ATTEMPT_REPORT or decision.get("novelty_report") != NOVELTY_REPORT:
        raise ValueError(f"{DECISION_REPORT} must link to {ATTEMPT_REPORT} and {NOVELTY_REPORT}")
    return decided_at


def require_number(report, field, context):
    value = report.get(field)
    try:
        number = finite_number(value)
    except OverflowError:
        number = None
    if number is None:
        raise ValueError(f"{field} in {context} must be a finite number")
    return value


def require_digest(report, field, context):
    digest = report.get(field)
    if not isinstance(digest, str) or not SHA256.fullmatch(digest):
        raise ValueError(f"Missing or invalid {field} SHA-256 in {context}")
    return digest


def verify_input_digest(report, field, expected, context):
    entry = object_field(report, field, context)
    if require_digest(entry, "sha256", f"{context} {field}") != expected:
        raise ValueError(f"Input SHA-256 mismatch in {context} {field}")


def verify_checks(report, failure_field, context):
    if report.get(failure_field) != []:
        raise ValueError(f"{context} must have an empty {failure_field} list")
    checks = report.get("checks")
    if not isinstance(checks, list):
        raise ValueError(f"Missing or invalid checks in {context}")
    for check in checks:
        if not isinstance(check, dict) or check.get("passed") is not True:
            name = check.get("name") if isinstance(check, dict) else None
            raise ValueError(f"Failed or invalid check in {context}: {name}")
    return checks


def verify_phase(phase, inputs, context, candidate_digest=None):
    if (phase.get("passed") is not True or phase.get("interrupted", False) is not False
            or phase.get("exit_code", 0) != 0):
        raise ValueError(f"{context} must contain a passed phase")
    checks = verify_checks(phase, "failures", context)
    verify_input_digest(phase, "input_manifest", inputs, context)
    if candidate_digest is not None:
        artifact = object_field(phase, "artifact", context)
        if require_digest(artifact, "sha256", f"{context} artifact") != candidate_digest:
            raise ValueError(f"Candidate worker artifact SHA-256 mismatch in {context}")
        for check in checks:
            if check.get("name") == "worker_artifact":
                for field in ("expected_sha256", "actual_sha256"):
                    if field in check and check[field] != candidate_digest:
                        raise ValueError(f"Candidate worker artifact SHA-256 mismatch in {context} {field}")


def verify_ready_metrics(attempt, phases, novelty_phase, decision_metrics):
    comparison = object_field(attempt, "comparison", ATTEMPT_REPORT)
    scores = object_field(comparison, "net_sharpe", ATTEMPT_REPORT)
    candidate = require_number(scores, "candidate", ATTEMPT_REPORT)
    baseline = require_number(scores, "baseline", ATTEMPT_REPORT)
    control = require_number(scores, "control", ATTEMPT_REPORT)
    if not candidate > max(0, baseline):
        raise ValueError("Candidate Sharpe must be above zero and the baseline")
    for field in ("candidate_finite", "candidate_positive", "candidate_strictly_above_baseline",
                  "candidate_performance_passed"):
        if comparison.get(field) is not True:
            raise ValueError(f"{field} in {ATTEMPT_REPORT} must be true for a ready candidate")

    for role, score in (("candidate", candidate), ("baseline", baseline), ("control", control)):
        phase = phases[f"{role}-full"]
        result = object_field(phase, "runner_result", f"{ATTEMPT_REPORT} {role}-full")
        if (require_number(result, "net_sharpe", f"{ATTEMPT_REPORT} {role}-full") != score
                or require_number(decision_metrics, role, DECISION_REPORT) != score):
            raise ValueError(f"{role} Sharpe mismatch between the comparison, full scores, and decision")
    novelty_result = object_field(novelty_phase, "runner_result", NOVELTY_REPORT)
    if require_number(novelty_result, "net_sharpe", NOVELTY_REPORT) != candidate:
        raise ValueError(f"Candidate Sharpe mismatch in {NOVELTY_REPORT}")
    if novelty_result != phases["candidate-full"]["runner_result"]:
        raise ValueError("Candidate runner result mismatch between CAP-2 and novelty")
    delta_field = "candidate_minus_control"
    delta = require_number(comparison, delta_field, ATTEMPT_REPORT)
    if delta != float(candidate) - float(control) or require_number(decision_metrics, delta_field, DECISION_REPORT) != delta:
        raise ValueError("Candidate-minus-control Sharpe mismatch between the comparison and decision")


def verify_ready_novelty(attempt, novelty, decision, candidate_digest):
    metrics = object_field(decision, "metrics", DECISION_REPORT)
    phase = object_field(novelty, "phase", NOVELTY_REPORT)
    phases = object_field(attempt, "phases", ATTEMPT_REPORT)
    inputs = require_digest(decision, "inputs_sha256", DECISION_REPORT)
    verify_checks(attempt, "errors", ATTEMPT_REPORT)
    verify_checks(novelty, "failures", NOVELTY_REPORT)
    for name in ("metadata", *(name for name, _, _ in PHASES)):
        worker = object_field(phases, name, ATTEMPT_REPORT)
        verify_phase(worker, inputs, f"{ATTEMPT_REPORT} {name}",
                     candidate_digest if name == "candidate-full" else None)
    verify_phase(phase, inputs, f"{NOVELTY_REPORT} phase", candidate_digest)
    verify_ready_metrics(attempt, phases, phase, metrics)
    city = object_field(phase, "novelty", NOVELTY_REPORT)
    distance = require_number(city, "city_novelty_deg", NOVELTY_REPORT)
    if not CITY_MIN_DISTANCE <= distance <= 180:
        raise ValueError(f"Unrounded city distance must be between {CITY_MIN_DISTANCE:g}° and 180°")
    nearest = city.get("nearest_city")
    if not isinstance(nearest, str) or not nearest.strip() or nearest.strip().upper() == "N/A":
        raise ValueError(f"A named nearest_city is required in {NOVELTY_REPORT}")
    if require_number(metrics, "city_novelty_deg", DECISION_REPORT) != distance or metrics.get("nearest_city") != nearest:
        raise ValueError("City distance or nearest_city mismatch between the novelty report and decision")

    confirmation = object_field(novelty, "confirmation", NOVELTY_REPORT)
    if confirmation.get("status") != "confirmed":
        raise ValueError(f"The scored city map must be confirmed in {NOVELTY_REPORT}")
    staged_map = object_field(novelty, "map", NOVELTY_REPORT)
    staged_map_after = object_field(novelty, "staged_map_after", NOVELTY_REPORT)
    scored_map = require_digest(decision, "scored_map_sha256", DECISION_REPORT)
    for report, field, context in ((city, "map_sha256", "novelty phase"),
                                   (staged_map, "sha256", "staged map"),
                                   (staged_map_after, "sha256", "staged_map_after"),
                                   (confirmation, "sha256", "map confirmation")):
        if require_digest(report, field, f"{NOVELTY_REPORT} {context}") != scored_map:
            raise ValueError(f"Scored map SHA-256 mismatch in {NOVELTY_REPORT} {context}")

    for report, field, filename in ((attempt, "inputs_before", ATTEMPT_REPORT),
                                     (attempt, "inputs_after", ATTEMPT_REPORT),
                                     (novelty, "input_manifest", NOVELTY_REPORT),
                                     (novelty, "inputs_after", NOVELTY_REPORT),
                                     (novelty, "inputs_at_decision", NOVELTY_REPORT)):
        verify_input_digest(report, field, inputs, filename)


def read_ready_attempt(attempt_dir):
    decision_bytes = read_local_file(attempt_dir, DECISION_REPORT)
    decision = json_object(decision_bytes, DECISION_REPORT)
    if decision.get("status") == "hold":
        reasons = decision.get("reasons")
        messages = []
        if isinstance(reasons, list):
            messages = [str(reason["message"]) for reason in reasons
                        if isinstance(reason, dict) and reason.get("message")]
        detail = " ".join(messages) or f"See {attempt_dir / DECISION_REPORT}."
        raise ValueError(f"Decision is hold. {detail}")

    attempt = json_object(read_local_file(attempt_dir, ATTEMPT_REPORT), ATTEMPT_REPORT)
    decision_link = object_field(attempt, "decision", ATTEMPT_REPORT)
    novelty_link = object_field(attempt, "novelty", ATTEMPT_REPORT)
    if (attempt.get("status") != "passed" or decision_link.get("path") != DECISION_REPORT
            or decision_link.get("status") != "ready" or novelty_link.get("path") != NOVELTY_REPORT):
        raise ValueError(f"{ATTEMPT_REPORT} must link to the finalized ready decision and novelty report")
    require_integer(attempt, "schema_version", SCHEMA_VERSION, ATTEMPT_REPORT)
    cap2_finished = require_utc_timestamp(attempt, "finished_at", ATTEMPT_REPORT)

    novelty = json_object(read_local_file(attempt_dir, NOVELTY_REPORT), NOVELTY_REPORT)
    require_integer(novelty, "schema_version", SCHEMA_VERSION, NOVELTY_REPORT)
    novelty_finished = require_utc_timestamp(novelty, "finished_at", NOVELTY_REPORT)
    attempt_id = attempt_dir.name
    if attempt.get("attempt_id") != attempt_id or novelty.get("attempt_id") != attempt_id:
        raise ValueError(f"Attempt ID mismatch across reports for {attempt_id}")
    if novelty.get("cap2_report") != ATTEMPT_REPORT:
        raise ValueError(f"{NOVELTY_REPORT} must link to {ATTEMPT_REPORT}")

    artifacts = object_field(attempt, "artifacts", ATTEMPT_REPORT)
    candidate = object_field(artifacts, "candidate", ATTEMPT_REPORT)
    digest = candidate.get("sha256")
    if not isinstance(digest, str) or not SHA256.fullmatch(digest):
        raise ValueError(f"Missing or invalid candidate SHA-256 in {ATTEMPT_REPORT}")
    decided_at = verify_ready_decision(decision, attempt_id, digest)
    if not cap2_finished <= novelty_finished <= decided_at:
        raise ValueError("Completion timestamps must be in CAP-2, novelty, and decision order")
    novelty_candidate = object_field(novelty, "candidate", NOVELTY_REPORT)
    if novelty_candidate.get("sha256") != digest:
        raise ValueError(f"Candidate SHA-256 mismatch in {NOVELTY_REPORT}")

    staged = candidate.get("staged")
    if not isinstance(staged, str) or not staged:
        raise ValueError(f"Missing staged candidate path in {ATTEMPT_REPORT}")
    if novelty_candidate.get("staged") != staged:
        raise ValueError(f"Staged candidate path mismatch between {ATTEMPT_REPORT} and {NOVELTY_REPORT}")
    staged_path = attempt_dir / staged
    resolved = staged_path.resolve()
    if not resolved.is_relative_to(attempt_dir):
        raise ValueError(f"Staged candidate path escapes the attempt directory: {staged}")
    name = candidate_filename(staged_path.name)
    if not resolved.is_file():
        raise ValueError(f"Missing or invalid staged candidate file: {staged_path}")
    candidate_bytes = resolved.read_bytes()
    if hashlib.sha256(candidate_bytes).hexdigest() != digest:
        raise ValueError(f"Staged candidate SHA-256 mismatch: {staged_path}")
    verify_ready_novelty(attempt, novelty, decision, digest)
    return name, digest, candidate_bytes, decision_bytes


def verify_handoff(handoff_dir):
    handoff = json_object(read_local_file(handoff_dir, HANDOFF_REPORT), HANDOFF_REPORT)
    require_integer(handoff, "schema_version", SCHEMA_VERSION, HANDOFF_REPORT)
    require_integer(handoff, "season", SEASON, HANDOFF_REPORT)
    require_integer(handoff, "cycle", CYCLE, HANDOFF_REPORT)
    if handoff.get("attempt_id") != handoff_dir.name:
        raise ValueError(f"{HANDOFF_REPORT} does not identify this Season {SEASON} Cycle {CYCLE} handoff")
    require_utc_timestamp(handoff, "prepared_at", HANDOFF_REPORT)
    source = handoff.get("source_attempt")
    if (not isinstance(source, str) or "\x00" in source
            or not any(path.is_absolute() and path.name == handoff_dir.name and ".." not in path.parts
                       for path in (PurePosixPath(source), PureWindowsPath(source)))):
        raise ValueError(f"source_attempt in {HANDOFF_REPORT} must be an absolute path with the handoff's attempt ID")

    name = candidate_filename(handoff.get("candidate_filename"))
    digest = handoff.get("candidate_sha256")
    if hashlib.sha256(read_local_file(handoff_dir, name)).hexdigest() != digest:
        raise ValueError(f"Archived candidate SHA-256 mismatch: {name}")
    decision_bytes = read_local_file(handoff_dir, DECISION_REPORT)
    if hashlib.sha256(decision_bytes).hexdigest() != handoff.get("decision_sha256"):
        raise ValueError(f"Archived decision SHA-256 mismatch: {DECISION_REPORT}")
    verify_ready_decision(json_object(decision_bytes, DECISION_REPORT), handoff_dir.name, digest)
    observations = handoff_dir / OBSERVATIONS
    if observations.resolve() != observations or not observations.is_dir():
        raise ValueError(f"Missing or invalid observation directory: {observations}")
    return handoff


def publish_archive(pending, archive):
    for retry in range(6):
        try:
            pending.rename(archive)
            return True
        except OSError as exc:
            if archive.exists() or archive.is_symlink():
                return False  # The caller verifies the archive published by the other writer.
            if retry == 5 or getattr(exc, "winerror", None) not in (5, 32, 33):
                raise
            sleep(0.05 * 2 ** retry)


def publish_observation(pending, path):
    # Windows rename fails if the destination exists. POSIX rename may replace it.
    if os.name == "nt":
        os.rename(pending, path)
    else:
        os.link(pending, path)


def emit_message(message, stream):
    if stream is None:
        return False
    try:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        message = message.encode(encoding, "backslashreplace").decode(encoding)
        print(message, file=stream, flush=True)
    except (OSError, ValueError):
        if stream is sys.__stdout__ or stream is sys.__stderr__:
            # Prevent Python's exit-time flush from overriding the command's result.
            try:
                with open(os.devnull, "w") as sink:
                    os.dup2(sink.fileno(), stream.fileno())
            except (OSError, ValueError):
                pass
        return False
    return True


def report_success(message):
    if not emit_message(message, sys.stdout):
        emit_message(message, sys.stderr)


def cleanup_temporary(path, *, directory=False):
    try:
        if directory:
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    except FileNotFoundError:
        pass
    except OSError as exc:
        emit_message(f"Temporary cleanup failed for {path}: {exc}", sys.stderr)


@contextmanager
def lock_handoff(directory):
    stream = (directory / HANDOFF_REPORT).open("rb")
    try:
        if os.name == "nt":
            # Lock beyond EOF so Windows readers can still verify the handoff JSON.
            offset = stream.seek(1, os.SEEK_END)
            handle = msvcrt.get_osfhandle(stream.fileno())
            while not lock_file(handle, offset & 0xffffffff, offset >> 32, 1, 0):
                error = ctypes.get_last_error()
                if error != LOCK_VIOLATION:
                    raise ctypes.WinError(error)
                sleep(0.05)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            stream.close()
        except OSError as exc:
            emit_message(f"Handoff lock cleanup failed for {directory}: {exc}", sys.stderr)


def check_recording_support(handoff_dir):
    directory = handoff_dir / OBSERVATIONS
    try:
        with lock_handoff(handoff_dir):
            with tempfile.NamedTemporaryFile(dir=directory, prefix=".outcome-probe-", suffix=".tmp",
                                             delete=False) as temporary:
                pending = Path(temporary.name)
            probe = pending.with_suffix(".published")
            try:
                publish_observation(pending, probe)
            except OSError:
                cleanup_temporary(pending)
                raise
            try:
                pending.unlink(missing_ok=True)
            finally:
                probe.unlink()
    except OSError as exc:
        raise ValueError(f"Safe observation publication is unavailable at {directory}: {exc}") from exc


def prepare(attempt_dir, handoff_root):
    attempt_dir = attempt_dir.resolve(strict=True)
    name, digest, candidate_bytes, decision_bytes = read_ready_attempt(attempt_dir)
    handoff_root.mkdir(parents=True, exist_ok=True)
    handoff_root = handoff_root.resolve(strict=True)
    archive = handoff_root / attempt_dir.name
    identity = {
        "schema_version": SCHEMA_VERSION,
        "season": SEASON,
        "cycle": CYCLE,
        "attempt_id": attempt_dir.name,
        "source_attempt": str(attempt_dir),
        "candidate_filename": name,
        "candidate_sha256": digest,
        "decision_sha256": hashlib.sha256(decision_bytes).hexdigest(),
    }
    if not archive.exists() and not archive.is_symlink():
        pending = handoff_root / f".handoff-{uuid4().hex}.tmp"
        pending.mkdir()
        try:
            (pending / name).write_bytes(candidate_bytes)
            (pending / DECISION_REPORT).write_bytes(decision_bytes)
            (pending / OBSERVATIONS).mkdir()
            atomic_json(pending / HANDOFF_REPORT, {**identity, "prepared_at": utc_now()})
            check_recording_support(pending)
            if publish_archive(pending, archive):
                return archive / name, digest
        finally:
            cleanup_temporary(pending, directory=True)

    if archive.resolve() != archive:
        raise ValueError(f"The existing handoff must be a directory at {archive}")
    existing = verify_handoff(archive)
    if any(existing.get(key) != value for key, value in identity.items()):
        raise ValueError(f"The existing handoff does not match the ready attempt: {archive}")
    check_recording_support(archive)
    return archive / name, digest


def require_entry_id_for_multiple_uploads(directory, handoff):
    identity = {
        "schema_version": SCHEMA_VERSION,
        "season": SEASON,
        "cycle": CYCLE,
        "handoff_id": handoff["attempt_id"],
        "candidate_sha256": handoff["candidate_sha256"],
    }
    known = set()
    for path in directory.iterdir():
        if not OBSERVATION_FILENAME.fullmatch(path.name):
            continue
        observation = json_object(read_local_file(directory, path.name), path)
        if any(type(observation.get(field)) is not type(value) or observation.get(field) != value
               for field, value in identity.items()):
            continue
        entry = observation.get("official_entry_id")
        if entry is None:
            continue
        if not isinstance(entry, str):
            raise ValueError(f"Invalid official_entry_id in {path}")
        if not entry.strip():
            continue
        known.add(entry)
        if len(known) > 1:
            raise ValueError("Use --official-entry-id when the handoff has multiple official entry IDs")


def record(handoff_dir, args):
    handoff_dir = handoff_dir.resolve(strict=True)
    if handoff_dir.parent != HANDOFF_ROOT.resolve():
        raise ValueError(f"Use the prepared handoff directory under {HANDOFF_ROOT}")
    handoff = verify_handoff(handoff_dir)
    digest = args.submitted_sha256.lower()
    if digest != handoff["candidate_sha256"]:
        raise ValueError("Submitted SHA-256 does not match the archived candidate")
    if args.verdict == "rejected" and args.quality_set_membership == "yes":
        raise ValueError("A rejected verdict cannot have quality-set membership yes")
    if args.official_entry_id is not None and not args.official_entry_id.strip():
        raise ValueError("--official-entry-id must be nonblank when supplied")

    directory = handoff_dir / OBSERVATIONS
    with lock_handoff(handoff_dir):
        if args.official_entry_id is None:
            require_entry_id_for_multiple_uploads(directory, handoff)
        return save_observation(directory, handoff, digest, args)


def save_observation(directory, handoff, digest, args):
    observation = {
        "schema_version": SCHEMA_VERSION,
        "observed_at": utc_now(),
        "season": SEASON,
        "cycle": CYCLE,
        "handoff_id": handoff["attempt_id"],
        "candidate_sha256": digest,
        "source": CITY_MAP_PAGE,
        "verdict": args.verdict,
        "official_entry_id": args.official_entry_id,
        "verdict_detail": args.verdict_detail,
        "settled_sharpe": args.settled_sharpe,
        "quality_set_membership": (None if args.quality_set_membership is None
                                   else args.quality_set_membership == "yes"),
    }
    path = directory / f"{uuid4().hex}.json"
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".outcome-", suffix=".tmp", delete=False) as temporary:
        pending = Path(temporary.name)
    try:
        atomic_json(pending, observation)
        publish_observation(pending, path)
    except FileExistsError as exc:
        raise ValueError(f"An observation already exists at {path}") from exc
    finally:
        cleanup_temporary(pending)
    return path


def finite_sharpe(value):
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("settled Sharpe must be finite")
    mantissa = value.lower().partition("e")[0]
    if number == 0 and any(digit.isdecimal() and int(digit) != 0 for digit in mantissa):
        raise argparse.ArgumentTypeError("settled Sharpe underflows to zero")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Archive a finalized ready candidate for manual upload")
    prepare_parser.add_argument("attempt_dir", metavar="ATTEMPT_DIR", type=Path)
    record_parser = commands.add_parser("record", help="Append an official outcome for a manually submitted file")
    record_parser.add_argument("handoff_dir", metavar="HANDOFF_DIR", type=Path)
    record_parser.add_argument("--submitted-sha256", metavar="SHA256", required=True)
    record_parser.add_argument("--verdict", choices=("pending", "accepted", "rejected"), required=True)
    record_parser.add_argument("--official-entry-id")
    record_parser.add_argument("--verdict-detail")
    record_parser.add_argument("--settled-sharpe", type=finite_sharpe)
    record_parser.add_argument("--quality-set-membership", choices=("yes", "no"))
    args = parser.parse_args(argv)

    try:
        if args.command == "prepare":
            candidate, digest = prepare(args.attempt_dir, HANDOFF_ROOT)
            report_success(f"Upload this archived candidate: {candidate}\n"
                           f"Handoff directory: {candidate.parent}\nSHA-256: {digest}")
        else:
            report_success(f"Recorded outcome: {record(args.handoff_dir, args)}")
    except (OSError, RuntimeError, ValueError) as exc:
        emit_message(f"{args.command}: {exc}", sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
