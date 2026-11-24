import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from unittest.mock import patch
from uuid import UUID, uuid4

from scripts import record_cycle_outcome as outcome
from scripts.record_cycle_outcome import HANDOFF_ROOT, ROOT
from scripts.validate_walk_forward import CITY_MAP_PAGE, PHASES, utc_now


SCRIPT = ROOT / "scripts" / "record_cycle_outcome.py"
EVIDENCE = ROOT / "_bmad-output" / "evidence"
CANDIDATE = b"# Retain the submitted bytes.\r\nVALUE = 42\r\n"
CANDIDATE_SHA256 = hashlib.sha256(CANDIDATE).hexdigest()
MAP_SHA256 = hashlib.sha256(b"scored map").hexdigest()
INPUTS_SHA256 = hashlib.sha256(b"scoring inputs").hexdigest()
SCORES = {"candidate": 0.81, "baseline": 0.75, "control": 0.79}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, report):
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


class CycleOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = EVIDENCE / f"test-cycle-outcome-{uuid4().hex}"
        self.attempt = self.fixture / f"attempt-{uuid4().hex}"
        self.archive = HANDOFF_ROOT / self.attempt.name
        self.addCleanup(shutil.rmtree, self.fixture, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.archive, ignore_errors=True)
        self.observations = self.archive / "observations"
        self.candidate = self.write_ready_attempt()

    def write_ready_attempt(self, attempt=None, *, scores=None):
        attempt = self.attempt if attempt is None else attempt
        scores = SCORES if scores is None else scores
        candidate_path = attempt / "artifacts" / "weighted candidate.py"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        now = utc_now()
        candidate = {"staged": candidate_path.relative_to(attempt).as_posix(),
                     "sha256": CANDIDATE_SHA256}
        common = {"schema_version": 1, "attempt_id": attempt.name}
        manifest = {"path": "inputs-before.json", "sha256": INPUTS_SHA256}
        map_evidence = {"staged": "city-map.parquet", "sha256": MAP_SHA256, "verified_at": now,
                        "verified_source": CITY_MAP_PAGE}
        city = {"city_novelty_deg": 60.0, "nearest_city": "Bucharest", "map_sha256": MAP_SHA256}
        worker = {"passed": True, "checks": [], "failures": [], "input_manifest": manifest}
        phases = {name: dict(worker) for name in ("metadata", *(name for name, _, _ in PHASES))}
        for role, score in scores.items():
            phases[f"{role}-full"]["runner_result"] = {
                "net_sharpe": score, "n_periods": 2, "n_observations": 1600,
                "per_period": {"001": {"n_val": 800}, "002": {"n_val": 800}},
            }
        phases["candidate-full"].update(artifact=candidate, checks=[{
            "name": "worker_artifact", "passed": True,
            "expected_sha256": CANDIDATE_SHA256, "actual_sha256": CANDIDATE_SHA256,
        }])
        delta = scores["candidate"] - scores["control"]
        candidate_path.write_bytes(CANDIDATE)
        write_json(attempt / "attempt.json", {
            **common, "status": "passed", "finished_at": now, "artifacts": {"candidate": candidate},
            "novelty": {"path": "novelty.json"}, "decision": {"path": "decision.json", "status": "ready"},
            "phases": phases, "inputs_before": manifest, "checks": [], "errors": [],
            "inputs_after": {"path": "inputs-after.json", "sha256": INPUTS_SHA256},
            "comparison": {"net_sharpe": scores, "candidate_minus_control": delta, "candidate_finite": True,
                           "candidate_positive": True, "candidate_strictly_above_baseline": True,
                           "candidate_performance_passed": True},
        })
        write_json(attempt / "novelty.json", {
            **common, "cap2_report": "attempt.json", "candidate": candidate, "finished_at": now,
            "phase": {**phases["candidate-full"], "novelty": city}, "map": map_evidence,
            "checks": [], "failures": [], "staged_map_after": {"sha256": MAP_SHA256},
            "confirmation": {"status": "confirmed", "sha256": MAP_SHA256,
                             "source": CITY_MAP_PAGE, "verified_at": now},
            "input_manifest": manifest,
            "inputs_after": {"path": "inputs-after-novelty.json", "sha256": INPUTS_SHA256},
            "inputs_at_decision": {"path": "inputs-at-decision.json", "sha256": INPUTS_SHA256},
        })
        write_json(attempt / "decision.json", {
            **common, "status": "ready", "decided_at": now, "reasons": [],
            "cap2_report": "attempt.json", "novelty_report": "novelty.json",
            "candidate_sha256": CANDIDATE_SHA256, "candidate_at_decision": {"sha256": CANDIDATE_SHA256},
            "metrics": {**scores, "candidate_minus_control": delta,
                        "city_novelty_deg": city["city_novelty_deg"], "nearest_city": city["nearest_city"]},
            "scored_map_sha256": MAP_SHA256, "inputs_sha256": INPUTS_SHA256,
        })
        return candidate_path

    def run_cli(self, *args, env=None):
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], cwd=ROOT,
                              capture_output=True, encoding="utf-8", env=env, timeout=30)

    def run_main(self, *args):
        argv = list(map(str, args))
        with redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()) as error:
            status = outcome.main(argv)
        return subprocess.CompletedProcess(argv, status, output.getvalue(), error.getvalue())

    def prepare(self):
        return self.run_cli("prepare", self.attempt)

    def record(self, verdict, *fields, digest=CANDIDATE_SHA256):
        return self.run_cli("record", self.archive, "--submitted-sha256", digest, "--verdict", verdict, *fields)

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)

    def assert_rejected(self, result, cause):
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(cause, result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def update_json(self, path, **fields):
        write_json(path, {**read_json(path), **fields})

    def update_report_field(self, filename, keys, value):
        path = self.attempt / filename
        report = read_json(path)
        section = report
        for key in keys[:-1]:
            section = section[key]
        section[keys[-1]] = value
        write_json(path, report)

    def read_observation(self, result):
        self.assert_success(result)
        path = next((path for path in self.observations.glob("*.json") if str(path) in result.stdout), None)
        self.assertIsNotNone(path, result.stdout)
        return path, read_json(path)

    def assert_utc(self, timestamp):
        self.assertEqual(datetime.fromisoformat(timestamp).utcoffset(), timedelta(0))

    def assert_json_identity(self, report, **expected):
        for key, value in expected.items():
            with self.subTest(field=key):
                self.assertEqual(report[key], value)
                self.assertIs(type(report[key]), type(value))

    def test_prepare_preserves_exact_bytes_and_reuses_archive(self):
        sources = {path: path.read_bytes() for path in self.attempt.rglob("*") if path.is_file()}
        result = self.prepare()
        self.assert_success(result)
        self.assertIn(str(self.archive / self.candidate.name), result.stdout)
        self.assertIn(f"Handoff directory: {self.archive}", result.stdout)
        self.assertIn(CANDIDATE_SHA256, result.stdout)
        self.assertEqual((self.archive / self.candidate.name).read_bytes(), CANDIDATE)
        decision_bytes = sources[self.attempt / "decision.json"]
        self.assertEqual((self.archive / "decision.json").read_bytes(), decision_bytes)
        handoff_path = self.archive / "handoff.json"
        handoff_bytes = handoff_path.read_bytes()
        handoff = read_json(handoff_path)
        self.assert_json_identity(
            handoff, season=2, cycle=1, schema_version=1, attempt_id=self.attempt.name,
            source_attempt=str(self.attempt), candidate_filename=self.candidate.name,
            candidate_sha256=CANDIDATE_SHA256, decision_sha256=hashlib.sha256(decision_bytes).hexdigest(),
        )
        self.assert_utc(handoff["prepared_at"])
        self.assertEqual(list(self.observations.iterdir()), [])
        self.assertEqual({path: path.read_bytes() for path in sources}, sources)

        observation, _ = self.read_observation(self.record("pending"))
        self.assert_success(self.prepare())
        self.assertEqual(handoff_path.read_bytes(), handoff_bytes)
        self.assertEqual(list(self.observations.iterdir()), [observation])

    def test_prepare_retries_a_transient_windows_rename_failure(self):
        denied = PermissionError("simulated temporary access denial")
        denied.winerror = 5
        rename = Path.rename
        attempts = []

        def transient_rename(source, destination):
            attempts.append(destination)
            if len(attempts) == 1:
                raise denied
            return rename(source, destination)

        with patch.object(Path, "rename", transient_rename):
            candidate, digest = outcome.prepare(self.attempt, HANDOFF_ROOT)
        self.assertGreaterEqual(len(attempts), 2)
        self.assertTrue(all(destination == self.archive for destination in attempts))
        self.assertEqual(candidate.read_bytes(), CANDIDATE)
        self.assertEqual(digest, CANDIDATE_SHA256)

    def test_prepare_publishes_only_the_complete_archive(self):
        ready = Event()
        resume = Event()
        publish = outcome.publish_archive

        def pause_publication(pending, archive):
            ready.set()
            if not resume.wait(timeout=10):
                raise RuntimeError("test did not release preparation")
            return publish(pending, archive)

        with patch.object(outcome, "publish_archive", side_effect=pause_publication) as publication:
            with ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(outcome.prepare, self.attempt, HANDOFF_ROOT)
                try:
                    self.assertTrue(ready.wait(timeout=10))
                    pending, destination = publication.call_args.args
                    self.assertEqual(destination, self.archive)
                    self.assertFalse(self.archive.exists())
                    self.assertEqual((pending / self.candidate.name).read_bytes(), CANDIDATE)
                    self.assertEqual((pending / "decision.json").read_bytes(),
                                     (self.attempt / "decision.json").read_bytes())
                    self.assertEqual(read_json(pending / "handoff.json")["candidate_sha256"], CANDIDATE_SHA256)
                    self.assertEqual(list((pending / "observations").iterdir()), [])
                finally:
                    resume.set()
                candidate, digest = result.result(timeout=10)
        self.assertEqual(candidate, self.archive / self.candidate.name)
        self.assertEqual(candidate.read_bytes(), CANDIDATE)
        self.assertEqual(digest, CANDIDATE_SHA256)
        self.assertFalse(pending.exists())

    def test_prepare_rejects_hold_with_the_validator_reason(self):
        self.update_json(self.attempt / "decision.json", status="hold", reasons=[{"message": "Below novelty floor."}])
        result = self.prepare()
        self.assert_rejected(result, "Decision is hold. Below novelty floor.")
        self.assertFalse(self.archive.exists())

    def test_prepare_rejects_incomplete_or_mismatched_reports(self):
        for filename, fields, cause in (
            ("attempt.json", {"decision": None}, "Missing or invalid decision"),
            ("attempt.json", {"status": "in-progress"}, "finalized ready decision"),
            ("attempt.json", {"decision": {"path": "other.json", "status": "ready"}}, "finalized ready decision"),
            ("attempt.json", {"decision": {"path": "decision.json", "status": "hold"}}, "finalized ready decision"),
            ("attempt.json", {"novelty": {"path": "other.json"}}, "finalized ready decision"),
            ("novelty.json", {"cap2_report": "other.json"}, "novelty.json must link to attempt.json"),
            ("decision.json", {"cap2_report": "other.json"}, "decision.json must link to"),
            ("decision.json", {"novelty_report": "other.json"}, "decision.json must link to"),
            ("novelty.json", {"attempt_id": "other-attempt"}, "Attempt ID mismatch"),
            ("decision.json", {"attempt_id": "other-attempt"}, "ready decision for attempt"),
            ("novelty.json", {"candidate": {"sha256": "0" * 64}}, "SHA-256 mismatch in novelty.json"),
            ("decision.json", {"candidate_at_decision": {"sha256": "0" * 64}}, "SHA-256 mismatch in decision.json"),
            ("decision.json", {"candidate_sha256": "0" * 64}, "SHA-256 mismatch in decision.json"),
        ):
            with self.subTest(filename=filename, fields=fields):
                self.write_ready_attempt()
                self.update_json(self.attempt / filename, **fields)
                self.assert_rejected(self.prepare(), cause)
                self.assertFalse(self.archive.exists())

        self.write_ready_attempt()
        (self.attempt / "novelty.json").unlink()
        self.assert_rejected(self.prepare(), "novelty.json")
        self.assertFalse(self.archive.exists())

    def test_prepare_requires_complete_versioned_ready_reports(self):
        for filename, fields, cause in (
            ("attempt.json", {"schema_version": True}, "schema_version"),
            ("novelty.json", {"schema_version": 1.0}, "schema_version"),
            ("decision.json", {"schema_version": 2}, "schema_version"),
            ("attempt.json", {"finished_at": None}, "valid UTC timestamp"),
            ("novelty.json", {"finished_at": "2026-11-24T00:00:00+02:00"}, "valid UTC timestamp"),
            ("decision.json", {"decided_at": "2026-11-24T00:00:00"}, "valid UTC timestamp"),
            ("decision.json", {"reasons": [{"code": "contradiction"}]}, "empty reasons list"),
            ("novelty.json", {"candidate": {"staged": "other.py", "sha256": CANDIDATE_SHA256}},
             "Staged candidate path mismatch"),
        ):
            with self.subTest(filename=filename, fields=fields):
                self.write_ready_attempt()
                self.update_json(self.attempt / filename, **fields)
                self.assert_rejected(self.prepare(), cause)
                self.assertFalse(self.archive.exists())

    def test_prepare_rejects_escaping_and_changed_candidates(self):
        outside = self.fixture / "outside.py"
        outside.write_bytes(CANDIDATE)
        candidate = {"staged": "../outside.py", "sha256": CANDIDATE_SHA256}
        self.update_json(self.attempt / "attempt.json", artifacts={"candidate": candidate})
        self.update_json(self.attempt / "novelty.json", candidate=candidate)
        self.assert_rejected(self.prepare(), "escapes the attempt directory")
        self.assertFalse(self.archive.exists())

        self.write_ready_attempt()
        self.candidate.write_bytes(b"changed after decision")
        self.assert_rejected(self.prepare(), "Staged candidate SHA-256 mismatch")
        self.assertFalse(self.archive.exists())

    def test_prepare_rejects_inconsistent_ready_metrics(self):
        for filename, keys, value, cause in (
            ("attempt.json", ("comparison", "net_sharpe", "candidate"), 0, "above zero and the baseline"),
            ("attempt.json", ("comparison", "net_sharpe", "candidate"), SCORES["baseline"], "above zero"),
            ("attempt.json", ("comparison", "net_sharpe", "candidate"), True, "must be a finite number"),
            ("attempt.json", ("comparison", "net_sharpe", "candidate"), 10 ** 500, "must be a finite number"),
            ("attempt.json", ("comparison", "candidate_strictly_above_baseline"), 1, "must be true"),
            ("attempt.json", ("phases", "baseline-full", "runner_result", "net_sharpe"), 0.7,
             "baseline Sharpe mismatch"),
            ("novelty.json", ("phase", "runner_result", "net_sharpe"), 0.81000000001, "Candidate Sharpe mismatch"),
            ("novelty.json", ("phase", "runner_result", "per_period", "001", "n_val"), 799,
             "Candidate runner result mismatch"),
            ("decision.json", ("metrics", "candidate"), None, "must be a finite number"),
            ("decision.json", ("metrics", "control"), 0.5, "control Sharpe mismatch"),
            ("decision.json", ("metrics", "candidate_minus_control"), 0, "Candidate-minus-control Sharpe mismatch"),
            ("novelty.json", ("phase", "novelty", "city_novelty_deg"), 59.9999, "Unrounded city distance"),
            ("novelty.json", ("phase", "novelty", "city_novelty_deg"), 180.0001, "Unrounded city distance"),
            ("novelty.json", ("phase", "novelty", "nearest_city"), "N/A", "named nearest_city"),
            ("decision.json", ("metrics", "city_novelty_deg"), 61, "City distance or nearest_city mismatch"),
            ("decision.json", ("metrics", "nearest_city"), "Other City", "City distance or nearest_city mismatch"),
        ):
            with self.subTest(filename=filename, keys=keys, value=value):
                self.write_ready_attempt()
                self.update_report_field(filename, keys, value)
                with self.assertRaisesRegex(ValueError, cause):
                    outcome.prepare(self.attempt, HANDOFF_ROOT)
                self.assertFalse(self.archive.exists())

    def test_prepare_rejects_zero_candidate_above_negative_baseline(self):
        self.write_ready_attempt(scores={**SCORES, "candidate": 0.0, "baseline": -0.75})
        with self.assertRaisesRegex(ValueError, "above zero"):
            outcome.prepare(self.attempt, HANDOFF_ROOT)
        self.assertFalse(self.archive.exists())

    def test_prepare_rejects_matching_but_wrong_control_delta(self):
        self.update_report_field("attempt.json", ("comparison", "candidate_minus_control"), 0.01)
        self.update_report_field("decision.json", ("metrics", "candidate_minus_control"), 0.01)
        with self.assertRaisesRegex(ValueError, "Candidate-minus-control Sharpe mismatch"):
            outcome.prepare(self.attempt, HANDOFF_ROOT)
        self.assertFalse(self.archive.exists())

    def test_prepare_rejects_inconsistent_map_inputs_and_completion_evidence(self):
        for filename, keys, value, cause in (
            ("novelty.json", ("confirmation", "status"), "map_changed", "must be confirmed"),
            ("novelty.json", ("phase", "novelty", "map_sha256"), "0" * 64, "Scored map SHA-256 mismatch"),
            ("novelty.json", ("map", "sha256"), "0" * 64, "Scored map SHA-256 mismatch"),
            ("novelty.json", ("staged_map_after", "sha256"), "0" * 64, "Scored map SHA-256 mismatch"),
            ("novelty.json", ("confirmation", "sha256"), "0" * 64, "Scored map SHA-256 mismatch"),
            ("decision.json", ("scored_map_sha256",), "invalid digest", "scored_map_sha256 SHA-256"),
            ("attempt.json", ("inputs_before", "sha256"), "0" * 64, "Input SHA-256 mismatch"),
            ("attempt.json", ("inputs_after", "sha256"), "0" * 64, "Input SHA-256 mismatch"),
            ("attempt.json", ("phases", "metadata", "input_manifest", "sha256"), "0" * 64, "Input SHA-256 mismatch"),
            ("attempt.json", ("phases", "candidate-full", "input_manifest", "sha256"), "0" * 64,
             "Input SHA-256 mismatch"),
            ("novelty.json", ("input_manifest", "sha256"), "0" * 64, "Input SHA-256 mismatch"),
            ("novelty.json", ("inputs_after", "sha256"), "0" * 64, "Input SHA-256 mismatch"),
            ("novelty.json", ("phase", "input_manifest", "sha256"), "0" * 64, "Input SHA-256 mismatch"),
            ("novelty.json", ("inputs_at_decision", "sha256"), "0" * 64, "Input SHA-256 mismatch"),
            ("decision.json", ("inputs_sha256",), "invalid digest", "inputs_sha256 SHA-256"),
            ("attempt.json", ("finished_at",), "9999-01-01T00:00:00+00:00", "Completion timestamps"),
            ("decision.json", ("decided_at",), "2000-01-01T00:00:00+00:00", "Completion timestamps"),
        ):
            with self.subTest(filename=filename, keys=keys, value=value):
                self.write_ready_attempt()
                self.update_report_field(filename, keys, value)
                with self.assertRaisesRegex(ValueError, cause):
                    outcome.prepare(self.attempt, HANDOFF_ROOT)
                self.assertFalse(self.archive.exists())

    def test_prepare_rejects_contradictory_worker_and_check_evidence(self):
        for filename, keys, value, cause in (
            ("attempt.json", ("phases", "candidate-full", "artifact", "sha256"), "0" * 64,
             "Candidate worker artifact SHA-256 mismatch"),
            ("attempt.json", ("phases", "candidate-full", "checks", 0, "actual_sha256"), "0" * 64,
             "Candidate worker artifact SHA-256 mismatch"),
            ("novelty.json", ("phase", "artifact", "sha256"), "0" * 64, "Candidate worker artifact SHA-256 mismatch"),
            ("novelty.json", ("phase", "checks", 0, "expected_sha256"), "0" * 64,
             "Candidate worker artifact SHA-256 mismatch"),
            ("attempt.json", ("errors",), [{"error": "interrupted"}], "empty errors list"),
            ("attempt.json", ("checks",), [{"name": "input_drift", "passed": False}], "input_drift"),
            ("attempt.json", ("phases", "candidate-quick", "passed"), False, "candidate-quick.*passed phase"),
            ("attempt.json", ("phases", "candidate-quick", "failures"), [{"error": "timeout"}], "empty failures list"),
            ("novelty.json", ("failures",), ["scoring error"], "empty failures list"),
            ("novelty.json", ("checks",), [{"name": "map_staged", "passed": False}], "map_staged"),
            ("novelty.json", ("phase", "checks", 0, "passed"), False, "worker_artifact"),
            ("novelty.json", ("phase", "exit_code"), 1, "novelty.json phase.*passed phase"),
        ):
            with self.subTest(filename=filename, keys=keys, value=value):
                self.write_ready_attempt()
                self.update_report_field(filename, keys, value)
                with self.assertRaisesRegex(ValueError, cause):
                    outcome.prepare(self.attempt, HANDOFF_ROOT)
                self.assertFalse(self.archive.exists())

    def test_prepare_rejects_duplicate_top_level_and_nested_json_keys(self):
        for filename, key in (("attempt.json", "schema_version"), ("novelty.json", "nearest_city"),
                              ("decision.json", "candidate_sha256")):
            with self.subTest(filename=filename):
                self.write_ready_attempt()
                path = self.attempt / filename
                content = path.read_text(encoding="utf-8").replace(f'"{key}":', f'"{key}": null, "{key}":', 1)
                path.write_text(content, encoding="utf-8")
                self.assert_rejected(self.prepare(), f"Duplicate JSON key: {key}")
                self.assertFalse(self.archive.exists())

    def test_prepare_rejects_a_filesystem_without_safe_publication(self):
        with patch.object(outcome, "publish_observation", side_effect=OSError("publication unsupported")):
            result = self.run_main("prepare", self.attempt)
        self.assert_rejected(result, "Safe observation publication is unavailable")
        self.assertEqual(result.stdout, "")
        self.assertFalse(self.archive.exists())

    def test_prepare_rechecks_publication_support_on_archive_reuse(self):
        self.assert_success(self.run_main("prepare", self.attempt))
        handoff = self.archive / "handoff.json"
        original = handoff.read_bytes()
        with patch.object(outcome, "publish_observation", side_effect=OSError("publication unsupported")):
            result = self.run_main("prepare", self.attempt)
        self.assert_rejected(result, "Safe observation publication is unavailable")
        self.assertEqual(result.stdout, "")
        self.assertEqual(handoff.read_bytes(), original)
        self.assertEqual(list(self.observations.iterdir()), [])

    def test_prepare_does_not_publish_when_probe_cleanup_fails(self):
        unlink = Path.unlink

        def fail_probe_cleanup(path, **kwargs):
            if path.suffix == ".published":
                raise PermissionError("simulated probe cleanup failure")
            return unlink(path, **kwargs)

        with patch.object(Path, "unlink", fail_probe_cleanup), patch.object(outcome, "publish_archive") as publish:
            result = self.run_main("prepare", self.attempt)
        self.assert_rejected(result, "simulated probe cleanup failure")
        publish.assert_not_called()
        self.assertEqual(result.stdout, "")
        self.assertFalse(self.archive.exists())

    @unittest.skipUnless(os.name == "nt", "Windows publication")
    def test_windows_prepare_and_record_use_no_hard_links(self):
        with patch.object(outcome.os, "link", side_effect=OSError("hard links unavailable")) as link:
            self.assert_success(self.run_main("prepare", self.attempt))
            _, observation = self.read_observation(self.run_main(
                "record", self.archive, "--submitted-sha256", CANDIDATE_SHA256, "--verdict", "pending",
            ))
        link.assert_not_called()
        self.assertEqual(observation["verdict"], "pending")

    @unittest.skipUnless(os.name == "nt", "Windows locking")
    def test_windows_lock_retries_only_confirmed_contention(self):
        path = self.attempt / outcome.HANDOFF_REPORT
        path.write_bytes(b"{}")
        with (patch.object(outcome, "lock_file", side_effect=[False] * 12 + [True]) as lock,
              patch.object(outcome.ctypes, "get_last_error", return_value=outcome.LOCK_VIOLATION),
              patch.object(outcome, "sleep") as wait):
            with outcome.lock_handoff(self.attempt):
                self.assertEqual(lock.call_count, 13)
        self.assertEqual(wait.call_count, 12)

        with (patch.object(outcome, "lock_file", return_value=False) as lock,
              patch.object(outcome.ctypes, "get_last_error", return_value=5),
              patch.object(outcome, "sleep") as wait):
            with self.assertRaises(PermissionError) as caught, outcome.lock_handoff(self.attempt):
                self.fail("acquired a lock despite access denial")
        self.assertEqual(caught.exception.winerror, 5)
        lock.assert_called_once()
        wait.assert_not_called()

    def test_duplicate_attempt_id_and_detached_archive_cannot_fork_history(self):
        self.assert_success(self.prepare())
        self.read_observation(self.record("pending"))
        original = {path: path.read_bytes() for path in self.archive.rglob("*") if path.is_file()}
        duplicate = self.fixture / "duplicate-source" / self.attempt.name
        shutil.copytree(self.attempt, duplicate)
        self.assert_rejected(self.run_cli("prepare", duplicate), "existing handoff does not match")
        self.assertEqual({path: path.read_bytes() for path in self.archive.rglob("*") if path.is_file()}, original)

        detached = self.fixture / "detached" / self.archive.name
        shutil.copytree(self.archive, detached)
        self.assert_rejected(self.run_cli("record", detached, "--submitted-sha256", CANDIDATE_SHA256,
                                         "--verdict", "accepted"), "Use the prepared handoff directory")
        self.assertEqual(len(list((detached / "observations").iterdir())), 1)
        self.assertEqual({path: path.read_bytes() for path in self.archive.rglob("*") if path.is_file()}, original)

    def test_redirected_output_handles_unicode_paths(self):
        attempt = self.fixture / f"{self.attempt.name}-候補"
        candidate = self.write_ready_attempt(attempt)
        archive = HANDOFF_ROOT / attempt.name
        self.addCleanup(shutil.rmtree, archive, ignore_errors=True)
        env = {**os.environ, "PYTHONIOENCODING": "ascii:strict"}
        prepared = self.run_cli("prepare", attempt, env=env)
        self.assert_success(prepared)
        self.assertIn(str(archive / candidate.name), prepared.stdout)
        result = self.run_cli("record", archive, "--submitted-sha256", CANDIDATE_SHA256, "--verdict", "pending", env=env)
        self.assert_success(result)
        observation, = (archive / "observations").glob("*.json")
        self.assertIn(str(observation), result.stdout)

    def test_record_preserves_partial_and_later_official_observations(self):
        self.assert_success(self.prepare())
        shutil.rmtree(self.attempt)
        pending_path, pending = self.read_observation(self.record("pending", digest=CANDIDATE_SHA256.upper()))
        pending_bytes = pending_path.read_bytes()
        accepted_path, accepted = self.read_observation(self.record(
            "accepted", "--official-entry-id", "entry-123", "--verdict-detail", "Passed entry checks",
        ))
        settled_path, settled = self.read_observation(self.record(
            "accepted", "--settled-sharpe=-1.2345678901234e-3", "--quality-set-membership", "no",
        ))
        rejected_path, rejected = self.read_observation(self.record(
            "rejected", "--verdict-detail", "Official verdict revised", "--settled-sharpe=0", "--quality-set-membership", "no",
        ))
        self.assertEqual(set(self.observations.iterdir()), {pending_path, accepted_path, settled_path, rejected_path})
        self.assertEqual(pending_path.read_bytes(), pending_bytes)
        self.assertEqual((pending["verdict"], accepted["verdict"], rejected["verdict"]),
                         ("pending", "accepted", "rejected"))
        for observation in (pending, accepted, settled, rejected):
            self.assert_json_identity(observation, season=2, cycle=1, schema_version=1, handoff_id=self.attempt.name,
                                      candidate_sha256=CANDIDATE_SHA256, source=CITY_MAP_PAGE)
            self.assert_utc(observation["observed_at"])
        for field in ("official_entry_id", "verdict_detail", "settled_sharpe", "quality_set_membership"):
            self.assertIsNone(pending[field])
        self.assert_json_identity(accepted, official_entry_id="entry-123", verdict_detail="Passed entry checks")
        self.assertIsNone(settled["official_entry_id"])
        self.assertEqual(settled["settled_sharpe"], -1.2345678901234e-3)
        self.assertEqual(rejected["settled_sharpe"], 0.0)
        self.assertIs(type(rejected["settled_sharpe"]), float)
        self.assertIs(rejected["quality_set_membership"], False)

    def test_record_supports_source_path_flavors_without_the_source(self):
        self.assert_success(self.prepare())
        shutil.rmtree(self.attempt)
        path = self.archive / "handoff.json"
        for source in (f"/missing/attempts/{self.attempt.name}", f"C:\\missing\\attempts\\{self.attempt.name}",
                       f"\\\\host\\share\\{self.attempt.name}"):
            with self.subTest(source=source):
                self.update_json(path, source_attempt=source)
                _, observation = self.read_observation(self.record("pending"))
                self.assert_json_identity(observation, handoff_id=self.attempt.name,
                                          candidate_sha256=CANDIDATE_SHA256)

    def test_record_preserves_entry_ids_and_keeps_unidentified_observations_unassigned(self):
        self.assert_success(self.prepare())
        unassigned, observation = self.read_observation(self.record("pending"))
        self.assertIsNone(observation["official_entry_id"])
        original = unassigned.read_bytes()
        for _ in range(2):
            _, observation = self.read_observation(self.record("accepted", "--official-entry-id", "first"))
            self.assertEqual(observation["official_entry_id"], "first")
        _, observation = self.read_observation(self.record("accepted"))
        self.assertIsNone(observation["official_entry_id"])
        self.read_observation(self.record("accepted", "--official-entry-id", "second"))
        before = {path: path.read_bytes() for path in self.observations.iterdir()}
        self.assert_rejected(self.record("accepted"), "Use --official-entry-id")
        self.assertEqual({path: path.read_bytes() for path in self.observations.iterdir()}, before)
        _, observation = self.read_observation(self.record("accepted", "--official-entry-id", "first"))
        self.assertEqual(observation["official_entry_id"], "first")
        self.assertEqual(unassigned.read_bytes(), original)

    def test_record_ignores_unassigned_and_unrelated_history(self):
        self.assert_success(self.run_main("prepare", self.attempt))
        argv = ["record", self.archive, "--submitted-sha256", CANDIDATE_SHA256, "--verdict", "pending"]
        _, existing = self.read_observation(self.run_main(*argv, "--official-entry-id", "first"))
        unrelated = self.observations / "unrelated.json"
        unrelated.write_bytes(b"not a recorder observation")
        for fields in (
            {"official_entry_id": "  "},
            {"handoff_id": "different archive", "official_entry_id": "second"},
            {"candidate_sha256": "0" * 64, "official_entry_id": "second"},
            {"season": 2.0, "official_entry_id": "second"},
        ):
            write_json(self.observations / f"{uuid4().hex}.json", {**existing, **fields})
        _, unassigned = self.read_observation(self.run_main(*argv))
        self.assertIsNone(unassigned["official_entry_id"])
        self.read_observation(self.run_main(*argv, "--official-entry-id", "second"))
        before = set(self.observations.iterdir())
        for blank in ("", "   "):
            self.assert_rejected(self.run_main(*argv, "--official-entry-id", blank), "must be nonblank")
        self.assertEqual(set(self.observations.iterdir()), before)

    def test_record_does_not_treat_a_failed_history_read_as_empty(self):
        self.assert_success(self.run_main("prepare", self.attempt))
        with patch.object(Path, "iterdir", side_effect=PermissionError("simulated history enumeration failure")):
            result = self.run_main("record", self.archive, "--submitted-sha256", CANDIDATE_SHA256,
                                   "--verdict", "pending")
        self.assert_rejected(result, "simulated history enumeration failure")
        self.assertEqual(list(self.observations.iterdir()), [])

    def test_record_checks_multi_entry_history_under_the_publication_lock(self):
        self.assert_success(self.prepare())
        first, observation = self.read_observation(self.record("accepted", "--official-entry-id", "first"))
        pending = self.observations / ".second.tmp"
        second = self.observations / f"{uuid4().hex}.json"
        write_json(pending, {**observation, "official_entry_id": "second"})

        # Signal only after the contender observes that the publisher holds the lock.
        script = """\
import os
import sys
from pathlib import Path
from scripts import record_cycle_outcome as outcome

marker = Path(sys.argv[1])
if os.name == "nt":
    wait = outcome.sleep

    def signal_contention(delay):
        marker.touch()
        wait(delay)

    outcome.sleep = signal_contention
else:
    flock = outcome.fcntl.flock

    def signal_contention(file, operation):
        try:
            flock(file, operation | outcome.fcntl.LOCK_NB)
        except BlockingIOError:
            marker.touch()
            flock(file, operation)

    outcome.fcntl.flock = signal_contention
raise SystemExit(outcome.main(sys.argv[2:]))
"""
        marker = self.fixture / "record-waiting"
        argv = [sys.executable, "-c", script, str(marker), "record", str(self.archive),
                "--submitted-sha256", CANDIDATE_SHA256, "--verdict", "accepted"]
        with ThreadPoolExecutor(max_workers=1) as pool:
            with outcome.lock_handoff(self.archive):
                writer = pool.submit(subprocess.run, argv, cwd=ROOT, capture_output=True, encoding="utf-8", timeout=30)
                deadline = monotonic() + 8
                while not marker.exists() and not writer.done() and monotonic() < deadline:
                    sleep(0.02)
                self.assertTrue(marker.exists())
                self.assertFalse(writer.done())
                self.assertEqual(set(self.observations.iterdir()), {first, pending})
                outcome.publish_observation(pending, second)
                pending.unlink(missing_ok=True)
            self.assert_rejected(writer.result(), "Use --official-entry-id")
        self.assertEqual(set(self.observations.iterdir()), {first, second})

    def test_record_rejects_invalid_handoff_metadata(self):
        self.assert_success(self.prepare())
        path = self.archive / "handoff.json"
        original = path.read_bytes()
        for field, value, cause in (
            ("schema_version", True, "schema_version"), ("season", 2.0, "season"), ("cycle", True, "cycle"),
            ("prepared_at", "not a date", "prepared_at"), ("source_attempt", self.attempt.name, "source_attempt"),
            ("source_attempt", str(self.attempt.with_name("wrong-attempt")), "source_attempt"),
            ("source_attempt", f"C:relative\\{self.attempt.name}", "source_attempt"),
            ("attempt_id", "wrong-attempt", "does not identify this Season 2 Cycle 1 handoff"),
        ):
            with self.subTest(field=field, value=value):
                path.write_bytes(original)
                self.update_json(path, **{field: value})
                self.assert_rejected(self.record("pending"), cause)
                self.assertEqual(list(self.observations.iterdir()), [])

    def test_record_rejects_duplicate_handoff_key(self):
        self.assert_success(self.prepare())
        path = self.archive / "handoff.json"
        content = path.read_text(encoding="utf-8").replace('"attempt_id":', '"attempt_id": null, "attempt_id":')
        path.write_text(content, encoding="utf-8")
        self.assert_rejected(self.record("pending"), "Duplicate JSON key: attempt_id")
        self.assertEqual(list(self.observations.iterdir()), [])

    def test_record_write_failure_or_collision_preserves_existing_observation(self):
        self.assert_success(self.prepare())
        existing, _ = self.read_observation(self.record("pending"))
        original = existing.read_bytes()
        argv = ["record", str(self.archive), "--submitted-sha256", CANDIDATE_SHA256, "--verdict", "accepted"]

        with patch.object(outcome, "uuid4", return_value=UUID(existing.stem)), redirect_stderr(io.StringIO()) as error:
            self.assertEqual(outcome.main(argv), 1)
            self.assertIn("observation already exists", error.getvalue())
        self.assertEqual(list(self.observations.iterdir()), [existing])
        self.assertEqual(existing.read_bytes(), original)

        def interrupt_write(path, _contents):
            path.write_bytes(b'{"incomplete":')
            raise OSError("simulated write failure")

        with patch.object(outcome, "atomic_json", side_effect=interrupt_write), redirect_stderr(io.StringIO()) as error:
            self.assertEqual(outcome.main(argv), 1)
            self.assertIn("simulated write failure", error.getvalue())
        self.assertEqual(list(self.observations.iterdir()), [existing])
        self.assertEqual(existing.read_bytes(), original)

    def test_record_reports_the_commit_after_cleanup_and_stdout_fail(self):
        self.assert_success(self.prepare())
        argv = ["record", str(self.archive), "--submitted-sha256", CANDIDATE_SHA256, "--verdict", "accepted"]
        unlink = Path.unlink

        def fail_cleanup(path, **kwargs):
            if path.name.startswith(".outcome-") and path.suffix == ".tmp":
                raise PermissionError("simulated post-commit cleanup failure")
            return unlink(path, **kwargs)

        class BrokenOutput(io.StringIO):
            def flush(self):
                raise BrokenPipeError("simulated stdout failure")

        with (patch.object(Path, "unlink", fail_cleanup), redirect_stdout(BrokenOutput()),
              redirect_stderr(io.StringIO()) as error):
            self.assertEqual(outcome.main(argv), 0)
        observation, = self.observations.glob("*.json")
        self.assertIn("simulated post-commit cleanup failure", error.getvalue())
        self.assertIn(f"Recorded outcome: {observation}", error.getvalue())
        self.assertEqual(read_json(observation)["verdict"], "accepted")

    def test_record_subprocess_exits_success_when_the_stdout_pipe_is_closed(self):
        self.assert_success(self.prepare())
        reader, writer = os.pipe()
        os.close(reader)
        try:
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "record", str(self.archive), "--submitted-sha256", CANDIDATE_SHA256,
                 "--verdict", "accepted"], cwd=ROOT, stdout=writer, stderr=subprocess.PIPE,
                encoding="utf-8", timeout=30,
            )
        finally:
            os.close(writer)
        self.assert_success(result)
        observation, = self.observations.glob("*.json")
        self.assertIn(f"Recorded outcome: {observation}", result.stderr)
        self.assertEqual(read_json(observation)["verdict"], "accepted")

    def test_record_escapes_a_surrogate_path_for_console_output(self):
        path = "observations/attempt-\udcff.json"
        argv = ["record", str(self.archive), "--submitted-sha256", CANDIDATE_SHA256, "--verdict", "pending"]
        buffer = io.BytesIO()
        with io.TextIOWrapper(buffer, encoding="ascii", errors="strict") as output:
            with patch.object(outcome, "record", return_value=path), redirect_stdout(output):
                self.assertEqual(outcome.main(argv), 0)
            self.assertIn(b"Recorded outcome: observations/attempt-\\udcff.json", buffer.getvalue())

    def test_record_rejects_invalid_digest_sharpe_and_tampered_archive(self):
        self.assert_success(self.prepare())
        observation, _ = self.read_observation(self.record("pending"))
        original = observation.read_bytes()
        self.assert_rejected(self.record("accepted", digest="0" * 64), "Submitted SHA-256 does not match")
        for value in ("nan", "inf", "-inf", "1e999"):
            with self.subTest(value=value):
                self.assert_rejected(self.record("accepted", f"--settled-sharpe={value}"), "Sharpe must be finite")
        for value in ("1e-999", "-3e-9999"):
            with self.subTest(value=value):
                self.assert_rejected(self.record("accepted", f"--settled-sharpe={value}"), "Sharpe underflows to zero")
        self.assert_rejected(self.record("rejected", "--quality-set-membership", "yes"),
                             "A rejected verdict cannot have quality-set membership yes")

        archived_candidate = self.archive / self.candidate.name
        archived_candidate.write_bytes(b"tampered candidate")
        self.assert_rejected(self.record("accepted"), "Archived candidate SHA-256 mismatch")
        self.assert_rejected(self.prepare(), "Archived candidate SHA-256 mismatch")
        self.assertEqual(archived_candidate.read_bytes(), b"tampered candidate")
        archived_candidate.write_bytes(CANDIDATE)

        decision = self.archive / "decision.json"
        self.update_json(decision, status="hold")
        self.assert_rejected(self.record("accepted"), "Archived decision SHA-256 mismatch")
        self.update_json(self.archive / "handoff.json", decision_sha256=hashlib.sha256(decision.read_bytes()).hexdigest())
        self.assert_rejected(self.record("accepted"), "must contain a ready decision")
        self.assertEqual(list(self.observations.iterdir()), [observation])
        self.assertEqual(observation.read_bytes(), original)

    def test_concurrent_prepare_and_record_keep_all_evidence(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            prepares = [pool.submit(self.prepare) for _ in range(4)]
            for future in prepares:
                self.assert_success(future.result())
            records = [pool.submit(self.record, "accepted", "--official-entry-id", str(i),
                                   "--quality-set-membership", "yes") for i in range(4)]
        for future in records:
            self.assert_success(future.result())
        self.assertEqual((self.archive / self.candidate.name).read_bytes(), CANDIDATE)
        observations = [read_json(path) for path in self.observations.iterdir()]
        self.assertEqual(len(observations), 4)
        self.assertEqual({value["official_entry_id"] for value in observations}, {"0", "1", "2", "3"})
        self.assertTrue(all(value["quality_set_membership"] is True for value in observations))
        self.assert_rejected(self.record("accepted"), "Use --official-entry-id")
        self.assertEqual(len(list(self.observations.iterdir())), 4)


if __name__ == "__main__":
    unittest.main()
