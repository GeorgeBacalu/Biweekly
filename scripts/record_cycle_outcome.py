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
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep
from uuid import uuid4

if __package__:
    from .validate_walk_forward import CITY_MAP_PAGE, ROOT, atomic_json, utc_now
else:
    from validate_walk_forward import CITY_MAP_PAGE, ROOT, atomic_json, utc_now


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


def read_local_file(directory, basename):
    path = directory / basename
    if path.resolve() != path:
        raise ValueError(f"{basename} must be a file directly in {directory}")
    if not path.is_file():
        raise ValueError(f"Missing or invalid evidence file: {path}")
    return path.read_bytes()


def json_object(data, filename):
    try:
        report = json.loads(data.decode("utf-8"))
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


def candidate_filename(name):
    if (not isinstance(name, str) or Path(name).suffix.lower() != ".py"
            or any(separator in name for separator in ("/", "\\", ":"))):
        raise ValueError("The candidate filename must be a .py basename")
    return name


def verify_ready_decision(decision, attempt_id, digest):
    if decision.get("status") != "ready" or decision.get("attempt_id") != attempt_id:
        raise ValueError(f"{DECISION_REPORT} must contain a ready decision for attempt {attempt_id}")
    require_integer(decision, "schema_version", SCHEMA_VERSION, DECISION_REPORT)
    require_utc_timestamp(decision, "decided_at", DECISION_REPORT)
    if decision.get("reasons") != []:
        raise ValueError(f"The ready decision in {DECISION_REPORT} must have an empty reasons list")
    retained = object_field(decision, "candidate_at_decision", DECISION_REPORT)
    if decision.get("candidate_sha256") != digest or retained.get("sha256") != digest:
        raise ValueError(f"Candidate SHA-256 mismatch in {DECISION_REPORT}")
    if decision.get("cap2_report") != ATTEMPT_REPORT or decision.get("novelty_report") != NOVELTY_REPORT:
        raise ValueError(f"{DECISION_REPORT} must link to {ATTEMPT_REPORT} and {NOVELTY_REPORT}")


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
    require_utc_timestamp(attempt, "finished_at", ATTEMPT_REPORT)

    novelty = json_object(read_local_file(attempt_dir, NOVELTY_REPORT), NOVELTY_REPORT)
    require_integer(novelty, "schema_version", SCHEMA_VERSION, NOVELTY_REPORT)
    require_utc_timestamp(novelty, "finished_at", NOVELTY_REPORT)
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
    verify_ready_decision(decision, attempt_id, digest)
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
    if not isinstance(source, str) or not Path(source).is_absolute() or Path(source).name != handoff_dir.name:
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
            return
        except OSError as exc:
            if archive.exists() or archive.is_symlink():
                return  # The caller verifies the archive published by the other writer.
            if retry == 5 or getattr(exc, "winerror", None) not in (5, 32, 33):
                raise
            sleep(0.05 * 2 ** retry)


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
            publish_archive(pending, archive)
        finally:
            if pending.exists():
                shutil.rmtree(pending)

    if archive.resolve() != archive:
        raise ValueError(f"The existing handoff must be a directory at {archive}")
    existing = verify_handoff(archive)
    if any(existing.get(key) != value for key, value in identity.items()):
        raise ValueError(f"The existing handoff does not match the ready attempt: {archive}")
    return archive / name, digest


def record(handoff_dir, args):
    handoff_dir = handoff_dir.resolve(strict=True)
    if handoff_dir.parent != HANDOFF_ROOT.resolve():
        raise ValueError(f"Use the prepared handoff directory under {HANDOFF_ROOT}")
    handoff = verify_handoff(handoff_dir)
    digest = args.submitted_sha256.lower()
    if digest != handoff["candidate_sha256"]:
        raise ValueError("Submitted SHA-256 does not match the archived candidate")

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
    directory = handoff_dir / OBSERVATIONS
    path = directory / f"{uuid4().hex}.json"
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".outcome-", suffix=".tmp", delete=False) as temporary:
        pending = Path(temporary.name)
    try:
        atomic_json(pending, observation)
        # Publish the complete file under its final name without replacing an existing observation.
        os.link(pending, path)
    except FileExistsError as exc:
        raise ValueError(f"An observation already exists at {path}") from exc
    finally:
        pending.unlink(missing_ok=True)
    return path


def finite_sharpe(value):
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("settled Sharpe must be finite")
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
            print(f"Upload this archived candidate: {candidate}")
            print(f"Handoff directory: {candidate.parent}")
            print(f"SHA-256: {digest}")
        else:
            print(f"Recorded outcome: {record(args.handoff_dir, args)}")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"{args.command}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
