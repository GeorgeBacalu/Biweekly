"""Collect offline walk-forward evidence and a local ready-or-hold decision."""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from uuid import uuid4

if __package__:
    from .materialize_control import derive_control
else:
    from materialize_control import derive_control

ROOT = Path(__file__).resolve().parents[1]
PERIODS = tuple(f"{number:03}" for number in range(1, 32))
UNIVERSE = "small"
DATA_DIR = "data/train"
GAUGE_DIMENSIONS = (19, 20)
CITY_WORLD = "small-4335-s2"
CITY_MIN_DISTANCE = 60.0
CITY_MAP_PAGE = "https://www.alphanova.tech/competition/biweekly/season-2/cycle-1"
NOVELTY_SETTINGS = "novelty-settings.json"
TRAIN_LIMIT = 240.0
PREDICT_LIMIT = 60.0
STARTUP_ALLOWANCE = 180.0
PEAK_RSS_LIMIT = 8_000_000_000
ZERO_SUM_TOLERANCE = 1e-12
CPU_ENV = {
    "COMP6_DISABLE_PIP_INSTALL": "1",
    "CUDA_VISIBLE_DEVICES": "-1",
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "MPLBACKEND": "Agg",
    "PYTHONDONTWRITEBYTECODE": "1",
}
ARTIFACTS = {
    "candidate": "submissions/weighted_vote.py",
    "control": "submissions/weighted_vote_control.py",
    "baseline": "requirements/demo_engineered.py",
}
PHASES = (
    ("candidate-quick", "quick", "candidate"),
    ("candidate-behavior", "behavior", "candidate"),
    ("candidate-full", "full", "candidate"),
    ("control-full", "full", "control"),
    ("baseline-full", "full", "baseline"),
)
REPRESENTATIVE_PERIODS = (PERIODS[0], PERIODS[len(PERIODS) // 2], PERIODS[-1])
SCORE_INPUT = re.compile(r"\d{3}\.small\.(features|returns|target|dispersion)\.parquet")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, contents):
    temporary = tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
        mode="w", encoding="utf-8", newline="\n",
    )
    pending = Path(temporary.name)
    try:
        with temporary:
            json.dump(contents, temporary, indent=2, sort_keys=True, allow_nan=False)
            temporary.write("\n")
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def fingerprint(path):
    before = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    identity = (before.st_size, before.st_mtime_ns, before.st_ino)
    if identity != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f"File changed while hashing: {path}")
    return {"sha256": digest, "bytes": before.st_size, "mtime_ns": before.st_mtime_ns}


def file_evidence(path):
    try:
        return fingerprint(path)
    except (OSError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def gauge_manifest(kit, expected):
    paths = [f"{DATA_DIR}/gauge/{period}.{UNIVERSE}.gauge.npy" for period in expected]
    return {relative: file_evidence(kit / relative) for relative in paths}


def input_manifest(root):
    kit = root / "requirements"
    data = kit / DATA_DIR
    sources = {kit / name for name in ("runner.py", "walkforward.py", "evaluation.py",
                                       "data_loader.py", "predictor.py", "requirements.txt")}
    sources.update(kit.rglob("*.py"))
    data_files = {
        data / f"{period}.{UNIVERSE}.{kind}.parquet"
        for period in PERIODS for kind in ("features", "returns", "target")
    }
    if data.is_dir():
        data_files.update(path for path in data.iterdir() if SCORE_INPUT.fullmatch(path.name))
    files = {path.relative_to(root).as_posix(): file_evidence(path)
             for path in sorted(sources | data_files)}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": files}


def changed_inputs(before, after):
    return [key for key in sorted(before["files"].keys() | after["files"].keys())
            if before["files"].get(key) != after["files"].get(key)]


def stage_artifacts(root, attempt):
    staging = attempt / "artifacts"
    staging.mkdir()
    artifacts = {}
    for name, relative in ARTIFACTS.items():
        path = root / relative
        staged = staging / path.name
        record = artifacts[name] = {
            "source": relative, "staged": staged.relative_to(attempt).as_posix(),
        }
        try:
            source = path.read_bytes()
            staged.write_bytes(source)
            record.update(sha256=hashlib.sha256(source).hexdigest(), bytes=len(source))
        except OSError as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
    return artifacts


def check(name, passed, **details):
    return {"name": name, "passed": bool(passed), **details}


def score_value(value):
    number = finite_number(value)
    reason = None
    if number is None:
        reason = ("non-finite runner Sharpe" if isinstance(value, Real) and not isinstance(value, bool)
                  else "missing or nonnumeric runner Sharpe")
    return {"net_sharpe": number, "net_sharpe_reason": reason}


def finite_number(value):
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def exposure_diagnostic(periods):
    """Summarize the scored candidate rows observed across periods."""
    rows = sum(len(points) for points in periods.values())
    if not rows:
        return {"passed": False, "rows": 0, "reason": "no aligned scored candidate predictions"}
    median = statistics.median(gross for points in periods.values() for _, gross in points)
    threshold = max(1e-12, 0.01 * median)
    flagged_rows = longest_streak = 0
    per_period = {}
    for period, points in periods.items():
        flagged = streak = longest = 0
        previous = None
        for hour, gross in points:
            if gross <= threshold:
                flagged += 1
                streak = streak + 1 if previous is not None and hour == previous + 1 else 1
                longest = max(longest, streak)
            else:
                streak = 0
            previous = hour
        per_period[period] = {"rows": len(points), "flagged_rows": flagged, "longest_streak": longest}
        flagged_rows += flagged
        longest_streak = max(longest_streak, longest)
    all_flat = flagged_rows == rows
    return {
        "passed": not all_flat and flagged_rows * 20 <= rows and longest_streak < 24,
        "rows": rows, "median_gross_exposure": median, "threshold": threshold,
        "flagged_rows": flagged_rows, "flagged_fraction": flagged_rows / rows,
        "longest_streak": longest_streak, "all_flat": all_flat, "per_period": per_period,
    }


def coverage_checks(result, expected):
    per_period = result.get("per_period", {})
    if not isinstance(per_period, dict):
        per_period = {}
    actual = {key: entry.get("n_val") if isinstance(entry, dict) else None
              for key, entry in per_period.items()}
    return [
        check("period_keys", actual.keys() == expected.keys(),
              missing=sorted(expected.keys() - actual.keys()), extra=sorted(actual.keys() - expected.keys())),
        check("period_row_counts", all(type(actual.get(period)) is int and actual[period] == rows
                                       for period, rows in expected.items()), actual=actual),
        check("period_count", type(result.get("n_periods")) is int and result["n_periods"] == len(expected),
              actual=result.get("n_periods"), expected=len(expected)),
        check("observation_count", type(result.get("n_observations")) is int
              and result["n_observations"] == sum(expected.values()),
              actual=result.get("n_observations"), expected=sum(expected.values())),
    ]


def read_events(path):
    if not path.is_file():
        return []
    events = []
    with path.open("rb") as stream:
        for line in stream:
            try:
                events.append(json.loads(line))
            except (ValueError, UnicodeError) as exc:
                events.append({"event": "failure", "phase": "journal",
                               "error": f"Invalid worker event: {exc}"})
    return events


def read_json(path):
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def phase_deadline(mode):
    # Allow all scheduled calls to use their full budgets, plus non-model work.
    train_calls, predict_calls = {
        "metadata": (0, 0),
        "quick": (1, 1),
        "full": (len(PERIODS), len(PERIODS)),
        "novelty": (len(PERIODS), len(PERIODS)),
        "behavior": (5 * len(REPRESENTATIVE_PERIODS), 20 * len(REPRESENTATIVE_PERIODS)),
    }[mode]
    return STARTUP_ALLOWANCE + train_calls * TRAIN_LIMIT + predict_calls * PREDICT_LIMIT


def run_worker(kit, attempt, name, mode, role, artifact, manifest_sha, settings_sha, *, novelty=None):
    phase = attempt / name
    phase.mkdir()
    if novelty is not None:
        atomic_json(phase / "novelty-inputs.json", novelty)
    worker = Path(__file__).with_name("_walk_forward_worker.py")
    artifact_path = str(attempt / artifact["staged"]) if artifact else ""
    arguments = [sys.executable, "-u", str(worker), mode, role or "", artifact_path, str(phase)]
    error = None
    exit_code = None
    interrupted = False
    deadline = phase_deadline(mode)
    started = time.perf_counter()
    with (phase / "stdout.log").open("wb") as stdout, (phase / "stderr.log").open("wb") as stderr:
        try:
            process = subprocess.run(arguments, cwd=kit, stdout=stdout, stderr=stderr,
                                     env={**os.environ, **CPU_ENV}, timeout=deadline)
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            error = f"Worker exceeded the phase deadline of {deadline:g} seconds"
        except KeyboardInterrupt:
            interrupted = True
            error = "Worker interrupted by operator"
        except OSError as exc:
            error = f"{type(exc).__name__}: {exc}"
    worker_seconds = time.perf_counter() - started
    events = read_events(phase / "events.ndjson")
    runtime = read_json(phase / "runtime.json")
    final = read_json(phase / "result.json")
    checks = [event["check"] for event in events if event["event"] == "check"]
    failures = [event for event in events if event["event"] == "failure"]
    last_phase = next((event["phase"] for event in reversed(events) if "phase" in event), "startup")
    if exit_code != 0 or not final:
        failures.append({"phase": last_phase, "error": error or f"Worker exit code {exit_code}; "
                         f"result report {'present' if final else 'missing'}"})
    starts = {event["id"]: event for event in events if event["event"] == "call_start"}
    for event in events:
        if event["event"] == "call_finish":
            event = dict(event)
            starts[event["id"]].update(event)
    calls = list(starts.values())
    peak = max((event["peak_rss_bytes"] for event in events
                if event.get("peak_rss_bytes") is not None), default=None)
    checks.extend([
        check("worker_exit", exit_code == 0 and bool(final), exit_code=exit_code),
        check("runtime_recorded", bool(runtime)),
        check("peak_resident_bytes", peak is not None and peak < PEAK_RSS_LIMIT,
              peak_rss_bytes=peak, limit_exclusive_bytes=PEAK_RSS_LIMIT,
              measured_at_worker_exit=final.get("peak_rss_bytes") is not None),
    ])
    if mode != "metadata":
        checks.append(check("model_calls", bool(calls) and all(
            call.get("seconds") is not None and call["seconds"] <= call["limit_seconds"]
            and not call.get("error") for call in calls)))
        artifact_sha = final.get("artifact_sha256") or next(
            (event["sha256"] for event in events if event["event"] == "artifact_hashed"), None)
        checks.append(check("worker_artifact", artifact["sha256"] == artifact_sha,
                            expected_sha256=artifact["sha256"], actual_sha256=artifact_sha))
    result = final.get("runner_result")
    if result is None:
        result = next((event["result"] for event in reversed(events)
                       if event["event"] == "runner_result"), {})
    record = {
        "mode": mode, "artifact": artifact, "exit_code": exit_code, "worker_seconds": worker_seconds,
        "deadline_seconds": deadline, "interrupted": interrupted,
        "input_manifest": {"path": "inputs-before.json", "sha256": manifest_sha},
        "settings": {"path": NOVELTY_SETTINGS if mode == "novelty" else "settings.json", "sha256": settings_sha},
        "files": {key: f"{name}/{key}" for key in (
            "stdout.log", "stderr.log", "events.ndjson", "runtime.json", "result.json", "runner.csv",
            "novelty-inputs.json",
        ) if (phase / key).is_file()},
        "runtime": runtime, "checks": checks, "failures": failures, "calls": calls,
        "peak_rss_bytes": peak, "runner_result": result,
        "worker_process_peak_rss_bytes": final.get("peak_rss_bytes"),
    }
    if final.get("artifact_snapshot"):
        record["files"]["artifact_snapshot"] = f"{name}/{final['artifact_snapshot']}"
    if mode == "behavior":
        record["reload_workers"] = [event for event in events if event["event"] == "reload_worker"]
    for key in ("expected_rows", "discovered_periods"):
        if key in final:
            record[key] = final[key]
    if name == "candidate-full":
        periods = {event["period"]: event["rows"] for event in events if event["event"] == "exposure"}
        record["exposure"] = exposure_diagnostic(periods)
        record["observed_scored_rows"] = {period: len(rows) for period, rows in periods.items()}
    if mode == "novelty":
        record["novelty"] = next((event["observation"] for event in reversed(events)
                                  if event["event"] == "novelty_result"), {})
    record["observed_periods"] = {
        event["period"]: event["observation"] for event in events if event["event"] == "scored_period"
    }
    return record


def evaluate_phase(name, record, expected, runtime):
    checks = record["checks"]
    result = record["runner_result"]
    runtime_matches = bool(runtime) and record["runtime"] == runtime
    checks.append(check("same_runtime", runtime_matches))
    finite = score_value(result.get("net_sharpe"))["net_sharpe"] is not None
    if record["mode"] != "behavior":
        checks.extend([
            check("runner_result", bool(result) and not result.get("error"),
                  error=result.get("error"), error_phase=result.get("error_phase")),
            check("finite_net_sharpe", finite,
                  reason=result.get("net_sharpe_reason", "no runner result")),
        ])
    if record["mode"] in ("full", "novelty"):
        coverage = coverage_checks(result, expected)
        checks.extend(coverage)
        record["comparable"] = (
            bool(result) and not result.get("error") and finite
            and runtime_matches and all(item["passed"] for item in coverage)
            and any(item["name"] == "worker_artifact" and item["passed"] for item in checks)
        )
        for period in expected:
            period_calls = [call for call in record["calls"] if call.get("period") == period]
            checks.append(check("period_model_calls", [call.get("phase") for call in period_calls]
                                == ["train", "predict"], period=period))
    if name == "candidate-quick":
        checks.extend([
            check("quick_demean", result.get("demean_check") is True),
            check("quick_row_count", result.get("val_rows") == expected[PERIODS[-1]],
                  actual=result.get("val_rows"), expected=expected[PERIODS[-1]]),
        ])
    if name == "candidate-full":
        checks.extend([
            check("candidate_exposure_coverage", record["observed_scored_rows"] == expected),
            check("candidate_exposure", record["exposure"]["passed"]),
        ])
    if name == "candidate-behavior":
        reloads = [item["period"] for item in checks if item["name"] == "reload" and item["passed"]]
        checks.append(check("representative_reloads", reloads == list(REPRESENTATIVE_PERIODS),
                            actual=reloads, expected=list(REPRESENTATIVE_PERIODS)))
    record["passed"] = not record["failures"] and all(item["passed"] for item in checks)


def comparison(phases, inputs_stable):
    full = {role: phases.get(f"{role}-full", {}) for role in ARTIFACTS}
    scores = {role: finite_number(phase.get("runner_result", {}).get("net_sharpe"))
              for role, phase in full.items()}
    candidate = scores["candidate"]
    finite = candidate is not None
    positive = finite and candidate > 0
    above = None
    delta = None
    if inputs_stable and finite and full["candidate"].get("comparable"):
        if scores["baseline"] is not None and full["baseline"].get("comparable"):
            above = candidate > scores["baseline"]
        if scores["control"] is not None and full["control"].get("comparable"):
            delta = candidate - scores["control"]
    return {
        "net_sharpe": scores, "candidate_minus_control": delta,
        "candidate_finite": finite, "candidate_positive": positive,
        "candidate_strictly_above_baseline": above,
        "candidate_performance_passed": positive and above is True,
    }


def stage_city_map(attempt, source):
    record = {"source": None, "staged": None, "verified_at": None}
    if source is None:
        return {**record, "error": "No city map supplied via --city-map"}
    source = Path(source).absolute()
    staged = attempt / "city-map.parquet"
    record["source"] = str(source)
    try:
        contents = source.read_bytes()
        staged.write_bytes(contents)
    except OSError as exc:
        return {**record, "error": f"{type(exc).__name__}: {exc}"}
    record.update(staged=staged.name, staged_at=utc_now(), bytes=len(contents),
                  sha256=hashlib.sha256(contents).hexdigest())
    return record


def confirm_city_map(scored_sha, staged_map):
    confirmation = {"status": "declined", "source": CITY_MAP_PAGE, "attested_at": None,
                    "verified_at": None}
    try:
        print(f"Re-download the latest official Season 2 Cycle 1 city map from {CITY_MAP_PAGE} and extract it.",
              flush=True)
        answer = input("Have you just downloaded the latest map from that page? [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            return confirmation
        confirmation["attested_at"] = utc_now()
        path = input("Path to the newly extracted Parquet (blank to decline): ").strip()
    except (EOFError, KeyboardInterrupt):
        return confirmation
    except (OSError, UnicodeError) as exc:
        return {**confirmation, "status": "prompt_error", "error": f"{type(exc).__name__}: {exc}"}
    if not path:
        return confirmation
    try:
        file = Path(path).absolute()
        confirmation["local_path"] = str(file)
        if file.samefile(staged_map):
            return {**confirmation, "status": "staged_map"}
        verified = file_evidence(file)
    except KeyboardInterrupt:
        return confirmation
    except (OSError, ValueError) as exc:
        return {**confirmation, "status": "unreadable", "error": f"{type(exc).__name__}: {exc}"}
    if "error" in verified:
        return {**confirmation, "status": "unreadable", "error": verified["error"]}
    sha = verified["sha256"]
    return {**confirmation, "sha256": sha, "verified_at": utc_now(),
            "status": "confirmed" if sha == scored_sha else "map_changed"}


def collect_novelty(root, attempt, report, before, city_map):
    candidate = report.get("artifacts", {}).get("candidate", {})
    metadata = report["phases"].get("metadata", {})
    expected = metadata.get("expected_rows", {})
    novelty = {
        "schema_version": 1, "attempt_id": attempt.name, "cap2_report": "attempt.json",
        "candidate": candidate, "input_manifest": report.get("inputs_before"),
        "cap2_settings": report.get("settings"), "gauge_fix": True, "started_at": utc_now(),
        "expected_coverage": {"per_period": expected, "total": sum(expected.values())},
        "checks": [], "failures": [], "phase": {},
    }
    checks = novelty["checks"]
    try:
        staged = novelty["map"] = stage_city_map(attempt, city_map)
        checks.append(check("map_staged", "sha256" in staged, error=staged.get("error")))
        inputs_stable = any(item["name"] == "input_drift" and item["passed"] for item in report["checks"])
        interrupted = (any(record.get("interrupted") for record in report["phases"].values())
                       or any(error.get("interrupted") for error in report["errors"]))
        eligible = inputs_stable and not interrupted and metadata.get("passed") and "sha256" in candidate
        checks.append(check("novelty_inputs", eligible, inputs_stable=inputs_stable,
                            cap2_interrupted=interrupted, metadata_passed=bool(metadata.get("passed"))))
        if "sha256" not in staged or not eligible:
            return novelty

        kit = root / "requirements"
        settings_path = attempt / NOVELTY_SETTINGS
        atomic_json(settings_path, {**read_json(attempt / report["settings"]["path"]), "gauge_fix": True})
        settings_sha = fingerprint(settings_path)["sha256"]
        novelty["settings"] = {"path": settings_path.name, "sha256": settings_sha}
        novelty["gauges_before"] = gauge_manifest(kit, expected)
        print("Running candidate-novelty", flush=True)
        try:
            record = novelty["phase"] = run_worker(
                kit, attempt, "candidate-novelty", "novelty", "candidate", candidate,
                report["inputs_before"]["sha256"], settings_sha,
                novelty={"expected_rows": expected, "map_path": str(attempt / staged["staged"]),
                         "map_sha256": staged["sha256"],
                         "gauge_sha256": {path: entry.get("sha256")
                                          for path, entry in novelty["gauges_before"].items()}},
            )
            evaluate_phase("candidate-novelty", record, expected, metadata["runtime"])
            checks.append(check("novelty_worker", record["passed"]))
            full = report["phases"].get("candidate-full", {}).get("runner_result")
            checks.append(check("candidate_score_agreement", bool(full) and record["runner_result"] == full))
        finally:
            novelty["gauges_after"] = gauge_manifest(kit, expected)
            changed = [path for path, entry in novelty["gauges_before"].items()
                       if novelty["gauges_after"].get(path) != entry]
            missing = [path for path, entry in novelty["gauges_before"].items() if "error" in entry]
            checks.append(check("gauge_digests", not changed and not missing, changed=changed, unavailable=missing))
            map_after = novelty["staged_map_after"] = file_evidence(attempt / staged["staged"])
            checks.append(check("staged_map_drift", map_after.get("sha256") == staged["sha256"]))
            after = input_manifest(root)
            atomic_json(attempt / "inputs-after-novelty.json", after)
            novelty["inputs_after"] = {"path": "inputs-after-novelty.json", "sha256": after["sha256"]}
            changes = changed_inputs(before, after)
            checks.append(check("novelty_input_drift", not changes, changed=changes))
    except (Exception, KeyboardInterrupt) as exc:
        novelty["failures"].append(f"{type(exc).__name__}: {exc}")
    return novelty


NOVELTY_CHECK_REASONS = {
    "map_staged": "A supplied city map could not be staged.",
    "novelty_inputs": "CAP-2 did not provide stable scoring inputs and metadata for novelty.",
    "novelty_worker": "The novelty worker failed validation. See its checks and failures.",
    "candidate_score_agreement": "The novelty Sharpe or its coverage differs from the candidate's CAP-2 full result.",
    "gauge_digests": "Expected raw gauge files were unavailable or changed during novelty scoring.",
    "staged_map_drift": "The staged map changed during novelty scoring.",
    "novelty_input_drift": "CAP-2 scoring inputs changed during novelty scoring.",
}


def decision_prerequisites(attempt, report, novelty):
    comparison = report["comparison"]
    metrics = dict(comparison["net_sharpe"])
    candidate = novelty["candidate"]
    observation = novelty["phase"].get("novelty", {})
    distance = finite_number(observation.get("city_novelty_deg"))
    nearest = observation.get("nearest_city")
    scored_map = observation.get("map_sha256")
    retained = file_evidence(attempt / candidate["staged"]) if candidate.get("staged") else {}
    metrics.update(candidate_minus_control=comparison["candidate_minus_control"],
                   city_novelty_deg=distance, nearest_city=nearest)
    gates = [
        ("cap2_failed", report["status"] == "passed", "CAP-2 checks failed. See attempt.json."),
        ("candidate_digest", bool(candidate.get("sha256")) and retained.get("sha256") == candidate["sha256"],
         "The staged candidate is unavailable or its digest differs from the scored candidate."),
        ("candidate_sharpe_finite", comparison["candidate_finite"], "The candidate Sharpe is not finite."),
        ("candidate_sharpe_positive", not comparison["candidate_finite"] or comparison["candidate_positive"],
         "The candidate Sharpe must be greater than zero."),
        ("candidate_beats_baseline", comparison["candidate_strictly_above_baseline"] is True,
         "The candidate did not show a strictly higher Sharpe than the shipped baseline."),
        ("novelty_failure", not novelty["failures"], "Novelty collection raised an error. See novelty.json."),
        ("scored_map_digest", bool(scored_map) and scored_map == novelty.get("map", {}).get("sha256"),
         "No city comparison is linked to the staged map digest."),
        ("city_distance_available", distance is not None and 0 <= distance <= 180,
         "A finite nearest-city distance in [0°, 180°] is unavailable."),
        ("nearest_city_available", bool(nearest) and nearest != "N/A", "The nearest city is unavailable."),
        ("city_distance_threshold", distance is None or distance >= CITY_MIN_DISTANCE,
         "The raw nearest-city distance is below 60°."),
    ]
    gates.extend((item["name"], item["passed"], item.get("error") or NOVELTY_CHECK_REASONS[item["name"]])
                 for item in novelty["checks"])
    reasons = [{"code": code, "message": message} for code, passed, message in gates if not passed]
    for reason in reasons:
        if reason["code"] == "novelty_worker":
            phase = novelty["phase"]
            reason["failed_checks"] = [
                {key: item[key] for key in ("name", "period", "error") if key in item and item[key] is not None}
                for item in phase["checks"] if not item["passed"]
            ]
            reason["failures"] = phase["failures"]
    return metrics, retained, reasons


def record_decision(root, attempt, report, novelty, before, request_confirmation):
    metrics, retained, reasons = decision_prerequisites(attempt, report, novelty)
    observation = novelty["phase"].get("novelty", {})
    scored_sha = observation.get("map_sha256")
    confirmation = {"status": "not_requested", "verified_at": None}
    if request_confirmation:
        if reasons:
            confirmation["status"] = "prerequisites_failed"
        else:
            staged_map = attempt / novelty["map"]["staged"]
            confirmation = confirm_city_map(scored_sha, staged_map)
            drift_reasons = []
            try:
                after = input_manifest(root)
                atomic_json(attempt / "inputs-at-decision.json", after)
                novelty["inputs_at_decision"] = {"path": "inputs-at-decision.json", "sha256": after["sha256"]}
                changed = changed_inputs(before, after)
                if changed:
                    drift_reasons.append({"code": "decision_input_drift", "changed": changed,
                                          "message": "Scoring inputs changed before the decision."})
                staged_digest = file_evidence(staged_map).get("sha256")
                if staged_digest != scored_sha:
                    drift_reasons.append({"code": "decision_map_drift",
                                          "message": "The staged map changed before the decision."})
                gauges = gauge_manifest(root / "requirements", novelty["expected_coverage"]["per_period"])
                if gauges != novelty["gauges_after"]:
                    drift_reasons.append({"code": "decision_gauge_drift",
                                          "message": "Gauge files changed before the decision."})
            except (Exception, KeyboardInterrupt) as exc:
                drift_reasons.append({"code": "decision_provenance_unavailable", "error": str(exc),
                                      "message": "Scoring input digests could not be checked before the decision."})
            _, retained, reasons = decision_prerequisites(attempt, report, novelty)
            reasons.extend(drift_reasons)
    if confirmation["status"] != "confirmed" or confirmation.get("sha256") != scored_sha:
        messages = {
            "map_changed": "The latest official map differs from the scored map. Run novelty again for this candidate.",
            "unreadable": "The confirmed download could not be read.",
            "staged_map": "Supply the newly extracted official download in a separate file from the attempt snapshot.",
            "prompt_error": "The current-map confirmation prompt could not be completed.",
        }
        reason = {"code": "current_map_confirmation",
                  "message": messages.get(confirmation["status"], "The scored city map lacks current official confirmation.")}
        if "error" in confirmation:
            reason["error"] = confirmation["error"]
        reasons.append(reason)
    novelty["confirmation"] = confirmation
    if confirmation["status"] == "confirmed":
        novelty["map"]["verified_at"] = confirmation["verified_at"]
        novelty["map"]["verified_source"] = confirmation["source"]
    novelty["finished_at"] = utc_now()
    decision = {
        "schema_version": 1, "attempt_id": attempt.name, "decided_at": utc_now(),
        "status": "hold" if reasons else "ready", "reasons": reasons, "metrics": metrics,
        "candidate_sha256": novelty["candidate"].get("sha256"), "candidate_at_decision": retained,
        "scored_map_sha256": scored_sha, "inputs_sha256": report.get("inputs_before", {}).get("sha256"),
        "cap2_report": "attempt.json", "novelty_report": "novelty.json",
    }
    atomic_json(attempt / "novelty.json", novelty)
    atomic_json(attempt / "decision.json", decision)
    return decision


def collect(root=ROOT, evidence_dir=None, *, city_map=None, confirm_current_map=False):
    evidence_dir = evidence_dir or root / "_bmad-output" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    attempt = (evidence_dir / f"{stamp}-{uuid4().hex}").resolve()
    attempt.mkdir()
    settings = {
        "environment": CPU_ENV, "interpreter": sys.executable, "platform": platform.platform(),
        "universe": UNIVERSE, "data_dir": DATA_DIR, "gauge_fix": False,
        "train_limit_seconds": TRAIN_LIMIT, "predict_limit_seconds": PREDICT_LIMIT,
        "peak_rss_limit_exclusive_bytes": PEAK_RSS_LIMIT, "zero_sum_tolerance": ZERO_SUM_TOLERANCE,
        "phase_startup_allowance_seconds": STARTUP_ALLOWANCE,
    }
    report = {"schema_version": 1, "attempt_id": attempt.name, "started_at": utc_now(),
              "status": "in-progress", "checks": [], "phases": {}, "errors": []}
    atomic_json(attempt / "attempt.json", report)
    print(f"Evidence attempt: {attempt}", flush=True)
    checks = report["checks"]
    phases = report["phases"]
    artifacts = {}
    before = None
    try:
        atomic_json(attempt / "settings.json", settings)
        settings_sha = fingerprint(attempt / "settings.json")["sha256"]
        report["settings"] = {"path": "settings.json", "sha256": settings_sha}
        artifacts = report["artifacts"] = stage_artifacts(root, attempt)
        for name, artifact in artifacts.items():
            checks.append(check("artifact_staged", "sha256" in artifact, artifact=name,
                                error=artifact.get("error")))
        if all("sha256" in artifacts[name] for name in ("candidate", "control")):
            candidate = attempt / artifacts["candidate"]["staged"]
            control = attempt / artifacts["control"]["staged"]
            try:
                matched = derive_control(candidate.read_bytes(), candidate.name) == control.read_bytes()
                checks.append(check("matched_control", matched))
            except (OSError, SyntaxError, ValueError) as exc:
                checks.append(check("matched_control", False, error=str(exc)))
        else:
            checks.append(check("matched_control", False, error="candidate or control was not staged"))
        before = input_manifest(root)
        atomic_json(attempt / "inputs-before.json", before)
        report["inputs_before"] = {"path": "inputs-before.json", "sha256": before["sha256"]}
        missing = {path: entry["error"] for path, entry in before["files"].items() if "error" in entry}
        checks.append(check("input_files", not missing, errors=missing))
        atomic_json(attempt / "attempt.json", report)

        if all(item["passed"] for item in checks):
            kit = root / "requirements"
            phases["metadata"] = metadata = run_worker(
                kit, attempt, "metadata", "metadata", None, None, before["sha256"], settings_sha,
            )
            expected = metadata.get("expected_rows", {})
            discovered = metadata.get("discovered_periods")
            metadata["checks"].append(check("release_periods", discovered == list(PERIODS),
                                            expected=list(PERIODS), actual=discovered))
            metadata["checks"].append(check(
                "expected_row_counts", expected.keys() == set(PERIODS)
                and all(type(rows) is int and rows > 0 for rows in expected.values()),
            ))
            metadata["passed"] = not metadata["failures"] and all(item["passed"] for item in metadata["checks"])
            report["expected_coverage"] = {"per_period": expected, "total": sum(expected.values())}
            atomic_json(attempt / "attempt.json", report)
            if metadata["passed"]:
                for name, mode, role in PHASES:
                    print(f"Running {name}", flush=True)
                    phases[name] = record = run_worker(
                        kit, attempt, name, mode, role, artifacts[role], before["sha256"], settings_sha,
                    )
                    evaluate_phase(name, record, expected, metadata["runtime"])
                    atomic_json(attempt / "attempt.json", report)
                    if record.get("interrupted"):
                        break
    except (Exception, KeyboardInterrupt) as exc:
        report["errors"].append({"phase": "collection", "error": f"{type(exc).__name__}: {exc}",
                                 "interrupted": isinstance(exc, KeyboardInterrupt)})
    finally:
        stable = False
        if before is not None:
            try:
                after = input_manifest(root)
                atomic_json(attempt / "inputs-after.json", after)
                report["inputs_after"] = {"path": "inputs-after.json", "sha256": after["sha256"]}
                changed = changed_inputs(before, after)
                checks.append(check("input_drift", not changed, changed=changed))
                stable = not changed
            except (OSError, ValueError, KeyboardInterrupt) as exc:
                if isinstance(exc, KeyboardInterrupt):
                    report["errors"].append({"phase": "input_drift", "error": "KeyboardInterrupt", "interrupted": True})
                checks.append(check("input_drift", False, error=f"{type(exc).__name__}: {exc}"))
        for name, artifact in artifacts.items():
            if "sha256" not in artifact:
                continue
            try:
                source = fingerprint(root / artifact["source"])["sha256"]
                retained = fingerprint(attempt / artifact["staged"])["sha256"]
                matches = source == retained == artifact["sha256"]
                checks.append(check("artifact_drift", matches, artifact=name,
                                    source_sha256=source, staged_sha256=retained))
                stable = stable and matches
            except (OSError, ValueError, KeyboardInterrupt) as exc:
                if isinstance(exc, KeyboardInterrupt):
                    report["errors"].append({"phase": "artifact_drift", "error": "KeyboardInterrupt", "interrupted": True})
                checks.append(check("artifact_drift", False, artifact=name, error=f"{type(exc).__name__}: {exc}"))
                stable = False
        report["comparison"] = comparison(phases, stable)
        checks.append(check("all_phases_ran", {"metadata", *(name for name, _, _ in PHASES)} == phases.keys(),
                            completed=list(phases)))
        passed = (not report["errors"] and all(item["passed"] for item in checks)
                  and all(record.get("passed") for record in phases.values()))
        report.update(status="passed" if passed else "failed", finished_at=utc_now())
        atomic_json(attempt / "attempt.json", report)
    novelty = collect_novelty(root, attempt, report, before, city_map)
    decision = record_decision(root, attempt, report, novelty, before, confirm_current_map)
    report.update(novelty={"path": "novelty.json"},
                  decision={"path": "decision.json", "status": decision["status"]})
    atomic_json(attempt / "attempt.json", report)
    return attempt, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-map", type=Path, help="Extracted Season 2 city map Parquet for novelty scoring")
    parser.add_argument("--confirm-current-map", action="store_true",
                        help="Offer to confirm a fresh official map download when the candidate meets the other prerequisites")
    args = parser.parse_args()
    attempt, report = collect(city_map=args.city_map, confirm_current_map=args.confirm_current_map)
    print(f"Evidence collection {report['status']}: {attempt / 'attempt.json'}")
    print(json.dumps(report["comparison"], allow_nan=False))
    print(f"Decision {report['decision']['status']}: {attempt / 'decision.json'}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
