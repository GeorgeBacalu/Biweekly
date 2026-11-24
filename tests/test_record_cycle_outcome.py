import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

from scripts import record_cycle_outcome as outcome
from scripts.record_cycle_outcome import HANDOFF_ROOT, ROOT
from scripts.validate_walk_forward import CITY_MAP_PAGE, atomic_json, utc_now


SCRIPT = ROOT / "scripts" / "record_cycle_outcome.py"
EVIDENCE = ROOT / "_bmad-output" / "evidence"
CANDIDATE = b"# Retain the submitted bytes.\r\nVALUE = 42\r\n"
CANDIDATE_SHA256 = hashlib.sha256(CANDIDATE).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class CycleOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = EVIDENCE / f"test-cycle-outcome-{uuid4().hex}"
        self.attempt = self.fixture / f"attempt-{uuid4().hex}"
        self.archive = HANDOFF_ROOT / self.attempt.name
        self.addCleanup(shutil.rmtree, self.fixture, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.archive, ignore_errors=True)
        self.observations = self.archive / "observations"
        self.candidate = self.write_ready_attempt()

    def write_ready_attempt(self, attempt=None):
        attempt = self.attempt if attempt is None else attempt
        candidate_path = attempt / "artifacts" / "weighted candidate.py"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        now = utc_now()
        candidate = {"staged": candidate_path.relative_to(attempt).as_posix(),
                     "sha256": CANDIDATE_SHA256}
        common = {"schema_version": 1, "attempt_id": attempt.name}
        candidate_path.write_bytes(CANDIDATE)
        atomic_json(attempt / "attempt.json", {
            **common, "status": "passed", "finished_at": now, "artifacts": {"candidate": candidate},
            "novelty": {"path": "novelty.json"}, "decision": {"path": "decision.json", "status": "ready"},
        })
        atomic_json(attempt / "novelty.json", {
            **common, "cap2_report": "attempt.json", "candidate": candidate, "finished_at": now,
        })
        atomic_json(attempt / "decision.json", {
            **common, "status": "ready", "decided_at": now, "reasons": [],
            "cap2_report": "attempt.json", "novelty_report": "novelty.json",
            "candidate_sha256": CANDIDATE_SHA256, "candidate_at_decision": {"sha256": CANDIDATE_SHA256},
        })
        return candidate_path

    def run_cli(self, *args, env=None):
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], cwd=ROOT,
                              capture_output=True, encoding="utf-8", env=env, timeout=30)

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
        atomic_json(path, {**read_json(path), **fields})

    def read_observation(self, result):
        self.assert_success(result)
        path = next((path for path in self.observations.glob("*.json") if str(path) in result.stdout), None)
        self.assertIsNotNone(path, result.stdout)
        return path, read_json(path)

    def assert_utc(self, timestamp):
        self.assertEqual(datetime.fromisoformat(timestamp).utcoffset(), timedelta(0))

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
        self.assertEqual((handoff["season"], handoff["cycle"], handoff["schema_version"]), (2, 1, 1))
        self.assertEqual(handoff["attempt_id"], self.attempt.name)
        self.assertEqual(handoff["source_attempt"], str(self.attempt))
        self.assertEqual(handoff["candidate_filename"], self.candidate.name)
        self.assertEqual(handoff["candidate_sha256"], CANDIDATE_SHA256)
        self.assertEqual(handoff["decision_sha256"], hashlib.sha256(decision_bytes).hexdigest())
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
        self.candidate.unlink()
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
            self.assertEqual((observation["season"], observation["cycle"]), (2, 1))
            self.assertEqual(observation["handoff_id"], self.attempt.name)
            self.assertEqual(observation["candidate_sha256"], CANDIDATE_SHA256)
            self.assertEqual(observation["source"], CITY_MAP_PAGE)
            self.assert_utc(observation["observed_at"])
        for field in ("official_entry_id", "verdict_detail", "settled_sharpe", "quality_set_membership"):
            self.assertIsNone(pending[field])
        self.assertEqual(accepted["official_entry_id"], "entry-123")
        self.assertEqual(accepted["verdict_detail"], "Passed entry checks")
        self.assertEqual(settled["settled_sharpe"], -1.2345678901234e-3)
        self.assertEqual(rejected["settled_sharpe"], 0.0)
        self.assertIs(type(rejected["settled_sharpe"]), float)
        self.assertIs(rejected["quality_set_membership"], False)

    def test_record_rejects_invalid_handoff_metadata(self):
        self.assert_success(self.prepare())
        path = self.archive / "handoff.json"
        original = path.read_bytes()
        for field, value in (
            ("schema_version", True), ("season", 2.0), ("cycle", True),
            ("prepared_at", "not a date"), ("source_attempt", self.attempt.name),
            ("source_attempt", str(self.attempt.with_name("wrong-attempt"))),
        ):
            with self.subTest(field=field, value=value):
                path.write_bytes(original)
                self.update_json(path, **{field: value})
                self.assert_rejected(self.record("pending"), field)
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

    def test_record_rejects_invalid_digest_sharpe_and_tampered_archive(self):
        self.assert_success(self.prepare())
        observation, _ = self.read_observation(self.record("pending"))
        original = observation.read_bytes()
        self.assert_rejected(self.record("accepted", digest="0" * 64), "Submitted SHA-256 does not match")
        for value in ("nan", "inf", "-inf", "1e999"):
            with self.subTest(value=value):
                self.assert_rejected(self.record("accepted", f"--settled-sharpe={value}"), "Sharpe must be finite")

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


if __name__ == "__main__":
    unittest.main()
