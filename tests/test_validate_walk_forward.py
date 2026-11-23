import io
import importlib
import json
import os
import py_compile
import subprocess
import sys
import unittest
from contextlib import chdir, contextmanager, redirect_stdout
from pathlib import Path
from shutil import rmtree
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import cloudpickle
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts import _walk_forward_worker as worker
from scripts import validate_walk_forward as validator
from scripts._walk_forward_worker import Journal, peak_rss_bytes, position_contract, summarize_result
from scripts.materialize_control import derive_control


EXPECTED = dict.fromkeys(validator.PERIODS, 800)
KIT = validator.ROOT / "requirements"
RUNTIME = {"python": "test", "dependencies": {"xgboost": "3.4.1"}}
SCORES = {"candidate": 0.12345678901234566, "control": 0.12345678901234441, "baseline": 0.12}
SOURCE = b"class WeightedVotePredictor:\n    _USE_WEIGHTED_VOTE = True\n"


@contextmanager
def workspace():
    directory = validator.ROOT / "_bmad-output" / "evidence" / f"test-{uuid4().hex}"
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        rmtree(directory)


def fixture(root):
    for relative in ("runner.py", "walkforward.py", "evaluation.py", "data_loader.py",
                     "predictor.py", "requirements.txt"):
        path = root / "requirements" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"shipped source")
    data = root / "requirements" / validator.DATA_DIR
    data.mkdir(parents=True)
    for period in EXPECTED:
        for kind in ("features", "returns", "target", "dispersion"):
            (data / f"{period}.small.{kind}.parquet").write_bytes(f"{period}:{kind}".encode())
    for role, relative in validator.ARTIFACTS.items():
        path = root / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(derive_control(SOURCE) if role == "control" else SOURCE)
    return data


def result(score):
    return {
        **validator.score_value(score),
        "n_periods": len(EXPECTED), "n_observations": sum(EXPECTED.values()),
        "per_period": {period: {"n_val": rows} for period, rows in EXPECTED.items()},
    }


def scored_calls(periods):
    return [{"period": period, "phase": phase, "seconds": 0.2}
            for period in periods for phase in ("train", "predict")]


def fake_worker(_, attempt, name, mode, role, artifact, manifest_sha, settings_sha):
    directory = attempt / name
    directory.mkdir()
    validator.atomic_json(directory / "runtime.json", RUNTIME)
    (directory / "stdout.log").write_text("ran shipped runner\n", encoding="utf-8")
    (directory / "stderr.log").touch()
    phase = {
        "mode": mode, "artifact": artifact, "runtime": RUNTIME, "runner_result": {},
        "checks": [], "failures": [], "peak_rss_bytes": 1_000_000,
        "input_manifest": {"path": "inputs-before.json", "sha256": manifest_sha},
        "settings": {"path": "settings.json", "sha256": settings_sha},
        "files": {filename: f"{name}/{filename}" for filename in ("runtime.json", "stdout.log", "stderr.log")},
        "calls": [],
    }
    if mode == "metadata":
        phase.update(expected_rows=EXPECTED, discovered_periods=list(EXPECTED))
    elif mode == "quick":
        phase["runner_result"] = {**validator.score_value(0.1), "demean_check": True, "val_rows": 800}
    elif mode == "behavior":
        phase["checks"].extend(validator.check("reload", True, period=period)
                               for period in validator.REPRESENTATIVE_PERIODS)
    elif mode == "full":
        phase["checks"].append(validator.check("worker_artifact", True))
        phase["runner_result"] = result(SCORES[role])
        phase["calls"] = scored_calls(EXPECTED)
        if role == "candidate":
            phase["observed_scored_rows"] = EXPECTED
            phase["exposure"] = {"passed": True}
    return phase


class CollectionTests(unittest.TestCase):
    def collect(self, root, worker=fake_worker):
        with patch.object(validator, "run_worker", side_effect=worker), redirect_stdout(io.StringIO()):
            return validator.collect(root)

    def test_missing_input_and_stale_control_preserve_preflight_evidence(self):
        with workspace() as root:
            data = fixture(root)
            missing = data / "016.small.returns.parquet"
            missing.unlink()
            (root / validator.ARTIFACTS["control"]).write_bytes(SOURCE)
            attempt, report = self.collect(root)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["phases"], {})
            checks = {entry["name"]: entry for entry in report["checks"]}
            self.assertFalse(checks["matched_control"]["passed"])
            self.assertIn("requirements/data/train/016.small.returns.parquet", checks["input_files"]["errors"])
            self.assertEqual(json.loads((attempt / "attempt.json").read_text()), report)
            self.assertTrue((attempt / report["artifacts"]["candidate"]["staged"]).is_file())

    def test_all_models_missing_same_period_and_wrong_counts_fail_coverage(self):
        def incomplete(*args):
            phase = fake_worker(*args)
            if phase["mode"] == "full":
                scored = phase["runner_result"]
                del scored["per_period"]["016"]
                scored["per_period"]["017"]["n_val"] = 801
                scored["n_periods"] = 30
                scored["n_observations"] = 24001
            return phase

        with workspace() as root:
            fixture(root)
            _, report = self.collect(root, incomplete)
        self.assertEqual(report["status"], "failed")
        self.assertIsNone(report["comparison"]["candidate_minus_control"])
        for role in validator.ARTIFACTS:
            phase = report["phases"][f"{role}-full"]
            checks = {entry["name"]: entry for entry in phase["checks"]}
            self.assertEqual(checks["period_keys"]["missing"], ["016"])
            self.assertFalse(checks["period_row_counts"]["passed"])
            self.assertFalse(checks["period_count"]["passed"])
            self.assertFalse(checks["observation_count"]["passed"])
            self.assertEqual(phase["runner_result"]["net_sharpe"], SCORES[role])

    def test_optional_dispersion_is_hashed_and_addition_and_removal_are_drift(self):
        with workspace() as root:
            data = fixture(root)
            path = data / "016.small.dispersion.parquet"
            key = "requirements/data/train/016.small.dispersion.parquet"
            path.unlink()
            absent = validator.input_manifest(root)
            self.assertNotIn(key, absent["files"])
            self.assertFalse(any("error" in entry for entry in absent["files"].values()))
            path.write_bytes(b"available dispersion")
            present = validator.input_manifest(root)
            self.assertEqual(present["files"][key], validator.fingerprint(path))
            self.assertEqual(validator.changed_inputs(absent, present), [key])
            path.unlink()
            self.assertEqual(validator.changed_inputs(present, validator.input_manifest(root)), [key])

    def test_weak_and_invalid_scores_are_distinct(self):
        for value, valid in ((-0.1234567891234567, True), (0.0, True), (None, False),
                             (float("nan"), False), (float("inf"), False), (-float("inf"), False)):
            with self.subTest(score=value):
                candidate = {
                    "mode": "full", "runner_result": result(value),
                    "checks": [validator.check("worker_artifact", True)], "failures": [],
                    "calls": scored_calls(EXPECTED),
                    "runtime": RUNTIME, "observed_scored_rows": EXPECTED, "exposure": {"passed": True},
                }
                validator.evaluate_phase("candidate-full", candidate, EXPECTED, RUNTIME)
                control = {"runner_result": result(SCORES["control"]), "comparable": True}
                phases = {"candidate-full": candidate, "control-full": control, "baseline-full": control}
                comparison = validator.comparison(phases, inputs_stable=True)
                self.assertIs(candidate["passed"], valid)
                self.assertIs(candidate["comparable"], valid)
                self.assertIs(comparison["candidate_finite"], valid)
                self.assertFalse(comparison["candidate_positive"])
                self.assertFalse(comparison["candidate_performance_passed"])
                self.assertEqual(comparison["net_sharpe"]["candidate"], value if valid else None)
                if valid:
                    self.assertEqual(comparison["candidate_minus_control"], value - SCORES["control"])
                else:
                    self.assertIsNone(comparison["candidate_minus_control"])
                    self.assertTrue(candidate["runner_result"]["net_sharpe_reason"])
                    json.dumps(candidate, allow_nan=False)

    def test_complete_weaker_score_keeps_comparison_after_exposure_and_resource_failure(self):
        def diagnosed(*args):
            phase = fake_worker(*args)
            if args[2] == "candidate-full":
                phase["exposure"] = {"passed": False}
                phase["checks"].append(validator.check("peak_resident_bytes", False))
            return phase

        with workspace() as root, patch.dict(SCORES, candidate=0.09):
            fixture(root)
            _, report = self.collect(root, diagnosed)
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["phases"]["candidate-full"]["passed"])
            self.assertTrue(report["phases"]["candidate-full"]["comparable"])
            self.assertTrue(report["comparison"]["candidate_positive"])
            self.assertFalse(report["comparison"]["candidate_strictly_above_baseline"])
            self.assertEqual(report["comparison"]["candidate_minus_control"], 0.09 - SCORES["control"])

    def test_interrupt_retains_the_current_worker_and_stops_dispatch(self):
        started = []

        def interrupted(arguments, **kwargs):
            directory = Path(arguments[-1])
            started.append(directory.name)
            validator.atomic_json(directory / "runtime.json", RUNTIME)
            if directory.name == "metadata":
                validator.atomic_json(directory / "result.json", {
                    "expected_rows": EXPECTED, "discovered_periods": list(EXPECTED), "peak_rss_bytes": 1000,
                })
                Journal(directory).emit("worker_exit", peak_rss_bytes=1000)
                return subprocess.CompletedProcess(arguments, 0)
            kwargs["stdout"].write(b"partial worker output\n")
            journal = Journal(directory)
            journal.period = "031"
            journal.model_call("train", lambda: None)
            journal.enter("predict")
            journal.emit("scored_period", observation={"rows": 800})
            raise KeyboardInterrupt

        with workspace() as root, redirect_stdout(io.StringIO()):
            fixture(root)
            with patch.object(validator.subprocess, "run", side_effect=interrupted):
                attempt, report = validator.collect(root)
            self.assertEqual(started, ["metadata", "candidate-quick"])
            phase = report["phases"]["candidate-quick"]
            self.assertEqual(report["status"], "failed")
            self.assertTrue(phase["interrupted"])
            self.assertEqual(phase["failures"][-1]["phase"], "predict")
            self.assertEqual(phase["observed_periods"], {"031": {"rows": 800}})
            self.assertEqual(phase["calls"][0]["phase"], "train")
            self.assertTrue((attempt / phase["files"]["events.ndjson"]).is_file())
            self.assertIn("partial worker output", (attempt / phase["files"]["stdout.log"]).read_text())
            self.assertEqual(json.loads((attempt / "attempt.json").read_text()), report)

    def test_result_errors_and_contract_failure_retain_other_observations(self):
        def failed_checks(*args):
            phase = fake_worker(*args)
            if args[2] == "candidate-quick":
                phase["runner_result"] = {}
            if args[2] == "candidate-behavior":
                for name in ("positions_hour_axis", "positions_finite", "equivalent_padding"):
                    phase["checks"].append(validator.check(name, False))
            if args[2] == "baseline-full":
                phase["runner_result"] = summarize_result(
                    {"sharpe": 0.0, "error": "train failed", "error_phase": "train"})
            return phase

        with workspace() as root:
            fixture(root)
            _, report = self.collect(root, failed_checks)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["phases"]["candidate-quick"]["passed"])
        self.assertFalse(report["phases"]["candidate-behavior"]["passed"])
        self.assertTrue(report["phases"]["candidate-full"]["passed"])
        baseline = report["phases"]["baseline-full"]["runner_result"]
        self.assertEqual(baseline["error_phase"], "train")
        self.assertIsNone(baseline["net_sharpe"])

    def test_precision_links_reruns_and_drift(self):
        with workspace() as root:
            data = fixture(root)
            first, completed = self.collect(root)
            self.assertEqual(completed["status"], "passed", completed["errors"])
            self.assertTrue(completed["comparison"]["candidate_strictly_above_baseline"])
            self.assertEqual(completed["comparison"]["candidate_minus_control"],
                             SCORES["candidate"] - SCORES["control"])
            for role, source in validator.ARTIFACTS.items():
                phase = completed["phases"][f"{role}-full"]
                self.assertEqual(phase["runner_result"]["net_sharpe"], SCORES[role])
                self.assertEqual(phase["runner_result"]["n_observations"], 24800)
                self.assertEqual(phase["input_manifest"], completed["inputs_before"])
                self.assertEqual(phase["settings"], completed["settings"])
                self.assertEqual(phase["artifact"]["source"], source)
                for relative in phase["files"].values():
                    self.assertTrue((first / relative).is_file())
                for key, path in (("staged", first), ("source", root)):
                    fingerprint = validator.fingerprint(path / phase["artifact"][key])
                    self.assertEqual(fingerprint["sha256"], phase["artifact"]["sha256"])

            def drift(*args):
                if args[2] == "control-full":
                    (data / "002.small.target.parquet").write_bytes(b"mutated data")
                    (root / "requirements" / "evaluation.py").write_bytes(b"mutated kit")
                return fake_worker(*args)

            second, unstable = self.collect(root, drift)
            self.assertNotEqual(first, second)
            self.assertEqual(json.loads((first / "attempt.json").read_text()), completed)
            self.assertEqual(unstable["status"], "failed")
            self.assertIsNone(unstable["comparison"]["candidate_minus_control"])
            check = next(entry for entry in unstable["checks"] if entry["name"] == "input_drift")
            self.assertEqual(check["changed"], ["requirements/data/train/002.small.target.parquet",
                                               "requirements/evaluation.py"])


class ExposureAndWorkerTests(unittest.TestCase):
    def test_timeout_preserves_partial_evidence_and_child_environment_is_forced(self):
        def timeout(arguments, **kwargs):
            self.assertEqual(kwargs["env"]["COMP6_DISABLE_PIP_INSTALL"], "1")
            self.assertEqual(kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "-1")
            self.assertEqual(kwargs["cwd"], KIT)
            self.assertEqual(kwargs["timeout"], validator.STARTUP_ALLOWANCE + len(EXPECTED) *
                             (validator.TRAIN_LIMIT + validator.PREDICT_LIMIT))
            kwargs["stderr"].write(b"worker last message\n")
            journal = Journal(Path(arguments[-1]))
            journal.period = "002"
            journal.emit("artifact_hashed", sha256="sha")
            journal.enter("predict")
            journal.emit("call_start", id=0, limit_seconds=60.0)
            journal.emit("scored_period", observation={"rows": 800})
            raise subprocess.TimeoutExpired(arguments, kwargs["timeout"])

        with workspace() as attempt:
            with patch.dict(validator.os.environ, COMP6_DISABLE_PIP_INSTALL="0"):
                with patch.object(validator.subprocess, "run", side_effect=timeout):
                    phase = validator.run_worker(KIT, attempt, "candidate-full", "full", "candidate",
                                                 {"staged": "staged.py", "sha256": "sha"}, "inputs", "settings")
            self.assertFalse(phase["interrupted"])
            self.assertIsNone(phase["exit_code"])
            self.assertEqual(phase["observed_periods"]["002"]["rows"], 800)
            self.assertEqual(phase["calls"][0]["phase"], "predict")
            self.assertEqual(phase["failures"][-1]["phase"], "predict")
            self.assertIn("exceeded the phase deadline", phase["failures"][-1]["error"])
            self.assertTrue(next(check["passed"] for check in phase["checks"] if check["name"] == "worker_artifact"))
            self.assertIn("worker last message", (attempt / phase["files"]["stderr.log"]).read_text())

    def test_positions_require_the_exact_axes_finite_numbers_and_a_zero_sum(self):
        hours = pd.Index([9, 10], name="hour")
        assets = pd.Index(["b", "a"], name="asset")
        features = pd.DataFrame(0.0, index=hours,
                                columns=pd.MultiIndex.from_product([["Feature.1"], assets]))
        valid = pd.DataFrame([[1.0, -1.0], [-2.0, 2.0]], index=hours, columns=assets)
        changes = {
            "positions_hour_axis": valid.iloc[::-1],
            "positions_asset_axis": valid.iloc[:, ::-1],
            "positions_finite": valid.mask(valid == 1, np.inf),
            "positions_zero_sum": valid + 0.01,
        }
        for name, positions in changes.items():
            with self.subTest(check=name), workspace() as directory:
                journal = Journal(directory)
                position_contract(journal, features, positions)
                checks = {entry["check"]["name"]: entry["check"]
                          for entry in validator.read_events(journal.path)}
                self.assertFalse(checks[name]["passed"])

    def test_exposure_threshold_fraction_streak_gaps_and_all_flat(self):
        diagnostic = validator.exposure_diagnostic
        fraction_boundary = [(i, 0.02 if i < 5 else 2.0) for i in range(100)]
        exact = diagnostic({"001": fraction_boundary})
        self.assertEqual(exact["threshold"], 0.02)
        self.assertEqual(exact["flagged_rows"], 5)
        self.assertTrue(exact["passed"])
        fraction_boundary[5] = (5, 0.0)
        self.assertFalse(diagnostic({"001": fraction_boundary})["passed"])

        streak = [(i, 0.0 if i < 24 else 2.0) for i in range(600)]
        self.assertEqual(diagnostic({"001": streak})["longest_streak"], 24)
        self.assertFalse(diagnostic({"001": streak})["passed"])
        split = diagnostic({"001": streak[:12], "002": streak[12:]})
        self.assertEqual(split["longest_streak"], 12)
        self.assertTrue(split["passed"])
        gapped = diagnostic({"001": streak[:12] + [(hour + 1, gross) for hour, gross in streak[12:]]})
        self.assertEqual(gapped["longest_streak"], 12)
        self.assertTrue(gapped["passed"])

        flat = diagnostic({"001": [(i, 0.0) for i in range(3)]})
        self.assertEqual(flat["threshold"], 1e-12)
        self.assertTrue(flat["all_flat"])
        self.assertFalse(flat["passed"])
        self.assertFalse(diagnostic({})["passed"])

    def test_worker_failure_retains_model_calls_observed_period_and_phase(self):
        with workspace() as attempt:
            artifact = {"staged": "staged.py", "sha256": "expected"}

            def failed_worker(arguments, **_):
                journal = Journal(Path(arguments[-1]))
                journal.period = "001"
                journal.model_call("train", lambda: None)
                journal.model_call("predict", lambda: None)
                journal.emit("scored_period", observation={"rows": 800, "net_sharpe": 0.22})
                journal.period = "002"
                with self.assertRaisesRegex(ValueError, "bad fit"):
                    journal.model_call("train", lambda: int("bad fit"), probe="test")
                journal.emit("call_start", id=3, phase="predict", limit_seconds=60,
                             peak_rss_bytes=validator.PEAK_RSS_LIMIT)
                return subprocess.CompletedProcess(arguments, 17)

            with patch.object(validator.subprocess, "run", side_effect=failed_worker):
                phase = validator.run_worker(attempt, attempt, "candidate-full", "full", "candidate",
                                             artifact, "inputs", "settings")
            self.assertIsNone(phase["runner_result"].get("net_sharpe"))
            self.assertEqual(phase["exit_code"], 17)
            self.assertEqual(phase["failures"][-1]["phase"], "predict")
            self.assertEqual(phase["observed_periods"]["001"]["rows"], 800)
            self.assertEqual(phase["calls"][2]["phase"], "train")
            self.assertEqual(phase["calls"][2]["period"], "002")
            self.assertGreaterEqual(phase["calls"][2]["seconds"], 0)
            self.assertIn("ValueError", phase["calls"][2]["error"])
            checks = {entry["name"]: entry for entry in phase["checks"]}
            self.assertFalse(checks["model_calls"]["passed"])
            self.assertFalse(checks["peak_resident_bytes"]["passed"])
            self.assertGreater(peak_rss_bytes(), 0)

    def test_worker_time_limits_are_inclusive_and_memory_limit_is_exclusive(self):
        def completed_worker(arguments, **_):
            directory = Path(arguments[-1])
            journal = Journal(directory)
            journal.period = "031"
            for index, (phase, limit, elapsed) in enumerate((
                ("train", validator.TRAIN_LIMIT, validator.TRAIN_LIMIT),
                ("predict", validator.PREDICT_LIMIT, predict_seconds),
            )):
                journal.enter(phase)
                journal.emit("call_start", id=index, limit_seconds=limit)
                journal.emit("call_finish", id=index, seconds=elapsed)
            peak = validator.PEAK_RSS_LIMIT - 1
            journal.emit("worker_exit", peak_rss_bytes=peak)
            validator.atomic_json(directory / "result.json", {"peak_rss_bytes": peak, "artifact_sha256": "sha"})
            validator.atomic_json(directory / "runtime.json", RUNTIME)
            return subprocess.CompletedProcess(arguments, 0)

        for predict_seconds, passed in ((validator.PREDICT_LIMIT, True), (validator.PREDICT_LIMIT + 0.01, False)):
            with self.subTest(predict_seconds=predict_seconds), workspace() as attempt:
                with patch.object(validator.subprocess, "run", side_effect=completed_worker):
                    phase = validator.run_worker(attempt, attempt, "candidate-quick", "quick", "candidate",
                                                 {"staged": "staged.py", "sha256": "sha"}, "inputs", "settings")
                checks = {entry["name"]: entry for entry in phase["checks"]}
                self.assertEqual(checks["model_calls"]["passed"], passed)
                self.assertTrue(checks["peak_resident_bytes"]["passed"])


class ScorerObserverTests(unittest.TestCase):
    def test_kit_import_uses_source_even_when_matching_bytecode_is_stale(self):
        with workspace() as kit:
            source = kit / "evaluation.py"
            source.write_text("score = 1\n", encoding="utf-8")
            stamp = source.stat()
            py_compile.compile(str(source), doraise=True,
                               invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
            source.write_text("score = 2\n", encoding="utf-8")
            os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            finder = worker.KitSourceFinder(kit)
            spec = finder.find_spec("evaluation")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.score, 2)
            self.assertIsNone(finder.find_spec("demo_engineered"))

    def test_materializer_rejects_class_body_override_but_allows_method_local_assignment(self):
        override = SOURCE + b"    if True:\n        _USE_WEIGHTED_VOTE = False\n"
        with self.assertRaisesRegex(ValueError, "exactly one enabled vote selector"):
            derive_control(override)
        local = SOURCE + b"    def train(self):\n        _USE_WEIGHTED_VOTE = False\n"
        self.assertEqual(derive_control(local), local.replace(b"= True", b"= False"))
        candidate = (validator.ROOT / validator.ARTIFACTS["candidate"]).read_bytes()
        control = (validator.ROOT / validator.ARTIFACTS["control"]).read_bytes()
        self.assertEqual(derive_control(candidate), control)

    def test_normalization_keeps_malformed_extra_period_keys(self):
        normalized = summarize_result({
            "sharpe": 0.3, "n_periods": 1, "n_observations": 800,
            "per_period": {"001": {"n_val": 800, "sharpe": 0.3}, "032": None, 33: "invalid"},
        })
        self.assertEqual(set(normalized["per_period"]), {"001", "032", "<int: 33>"})
        self.assertIsNone(normalized["per_period"]["032"]["n_val"])
        self.assertIsNone(normalized["per_period"]["032"]["net_sharpe"])
        self.assertIn("invalid", normalized["per_period"]["032"]["net_sharpe_reason"])
        keys = validator.coverage_checks(normalized, {"001": 800})[0]
        self.assertFalse(keys["passed"])
        self.assertEqual(keys["extra"], ["032", "<int: 33>"])

    def test_runtime_preserves_available_versions_when_a_dependency_import_fails(self):
        with workspace() as directory:
            path = directory / "runtime.json"
            imports = [SimpleNamespace(__version__="1.2"), SimpleNamespace(__version__="3.4"),
                       ModuleNotFoundError("missing pyarrow")]
            with patch.object(worker.importlib, "import_module", side_effect=imports):
                with self.assertRaisesRegex(ModuleNotFoundError, "missing pyarrow"):
                    worker.load_runtime(path)
            runtime = validator.read_json(path)
            self.assertEqual(runtime["interpreter"], sys.executable)
            self.assertTrue(runtime["python"])
            self.assertEqual(runtime["dependencies"], {"numpy": "1.2", "pandas": "3.4"})

    def test_metadata_uses_shipped_uncapped_and_capped_splits(self):
        with workspace() as directory, chdir(directory):
            data = directory / validator.DATA_DIR
            data.mkdir(parents=True)
            for period, rows in (("001", 29), ("002", 4_500)):
                pq.write_table(pa.table({"feature": range(rows)}), data / f"{period}.small.features.parquet")
            output = {}
            with patch.object(sys, "path", [str(KIT), *sys.path]), patch.object(worker, "PERIODS", ("001", "002")):
                worker.collect_metadata(Journal(directory), output)
            self.assertEqual(output["discovered_periods"], ["001", "002"])
            self.assertEqual(output["expected_rows"], {"001": 5, "002": 800})

    def test_full_scorer_observes_raw_positions_and_rejects_mutated_input_axes(self):
        with patch.object(sys, "path", [str(KIT), *sys.path]):
            runner = importlib.import_module("runner")
            walkforward = importlib.import_module("walkforward")
            predictor = importlib.import_module("predictor")

        def fixed_positions(mutate):
            class FixedPositions(predictor.Predictor):
                def train(self, features, target):
                    pass

                def predict(self, features):
                    index = features.index.copy(deep=True)
                    assets = features.columns.get_level_values(1).unique()
                    if mutate:
                        features.index.name = "changed hour"
                        features.columns = features.columns.set_names(["changed feature", "changed asset"])
                    return pd.DataFrame([[1.0, -1.0]] * len(features), index=index, columns=assets)

            return FixedPositions

        for mutate in (False, True):
            with self.subTest(mutate=mutate), workspace() as directory:
                hours = pd.Index(np.arange(15), name="hour")
                assets = pd.Index(["a", "b"], name="asset")
                columns = pd.MultiIndex.from_product([["Feature.1"], assets], names=["feature", "asset"])
                features = pd.DataFrame(0.0, index=hours, columns=columns)
                targets = pd.DataFrame(np.arange(30).reshape(15, 2), index=hours.copy(), columns=assets)
                data = {"features": features, "target": targets, "returns": targets}
                journal, output = Journal(directory), {}
                with (patch.object(runner, "load_predictor_class_from_file", return_value=fixed_positions(mutate)),
                      patch.object(walkforward, "load_period", return_value=data),
                      patch.object(walkforward, "list_periods", return_value=["001"]), redirect_stdout(io.StringIO())):
                    worker.score(journal, directory / "staged.py", directory, "full", "candidate", output)
                events = validator.read_events(journal.path)
                checks = {event["check"]["name"]: event["check"]["passed"]
                          for event in events if event["event"] == "check"}
                self.assertEqual(output["runner_result"]["n_observations"], 3)
                self.assertIs(checks["input_hour_axis_unchanged"], not mutate)
                self.assertIs(checks["input_feature_axis_unchanged"], not mutate)
                self.assertTrue(checks["positions_hour_axis"])
                self.assertTrue(checks["positions_asset_axis"])
                observed = [event["rows"] for event in events if event["event"] == "exposure"]
                self.assertEqual(observed, [] if mutate else [[[12, 2.0], [13, 2.0], [14, 2.0]]])

    def test_reloads_use_fresh_processes_and_record_failure_for_the_period(self):
        module = ModuleType("cap2_standalone_fixture_" + uuid4().hex)
        exec(
            "import os\n"
            "import pandas as pd\n"
            "class FixturePredictor:\n"
            "    def train(self, features, target):\n"
            "        self.pid = os.getpid()\n"
            "        self.fail_in_child = target.iloc[0, 0] < 0\n"
            "    def predict(self, features):\n"
            "        if self.pid != os.getpid() and self.fail_in_child:\n"
            "            raise RuntimeError('deterministic reload failure')\n"
            "        return pd.DataFrame(0.0, index=features.index, columns=features.columns.get_level_values(1))\n",
            module.__dict__,
        )
        hours = pd.Index(np.arange(160), name="hour")
        assets = pd.Index(["a", "b"], name="asset")
        columns = pd.MultiIndex.from_product([["Feature.1"], assets])
        features = pd.DataFrame(1.0, index=hours, columns=columns)
        with workspace() as directory, chdir(KIT):
            journal = Journal(directory)
            with patch.dict(sys.modules, {module.__name__: module}):
                cloudpickle.register_pickle_by_value(module)
            try:
                for period in validator.REPRESENTATIVE_PERIODS:
                    journal.period = period
                    target = pd.DataFrame(-1.0 if period == "016" else 1.0, index=hours, columns=assets)
                    worker.behavior_panel(journal, module.FixturePredictor, features, target, 80)
            finally:
                cloudpickle.unregister_pickle_by_value(module)
            events = validator.read_events(journal.path)
            reload_checks = {event["period"]: event["check"]["passed"] for event in events
                             if event["event"] == "check" and event["check"]["name"] == "reload"}
            self.assertEqual(reload_checks, {"001": True, "016": False, "031": True})
            standalone = [event["period"] for event in events if event["event"] == "check"
                          and event["check"]["name"] == "standalone_reload" and event["check"]["passed"]]
            self.assertEqual(standalone, list(validator.REPRESENTATIVE_PERIODS))
            for name in ("short_frame", "future_perturbation"):
                checks = [event for event in events if event["event"] == "check"
                          and event["check"]["name"] == name and event["check"]["passed"]]
                self.assertEqual([event["period"] for event in checks], list(validator.REPRESENTATIVE_PERIODS))
            reloads = [event for event in events if event["event"] == "reload_worker"]
            self.assertEqual([event["exit_code"] for event in reloads], [0, 1, 0])
            self.assertTrue(all(event["peak_rss_bytes"] > 0 and event["measured_at_worker_exit"] for event in reloads))
            reload_calls = [event for event in events
                            if event["event"] == "call_finish" and event.get("process") == "reload"]
            self.assertEqual([event["period"] for event in reload_calls], list(validator.REPRESENTATIVE_PERIODS))
            self.assertTrue(all(event["seconds"] >= 0 for event in reload_calls))
            self.assertIn("deterministic reload failure", reload_calls[1]["error"])

    def test_behavior_detects_reused_output_and_future_only_dependence(self):
        class ProbePredictor:
            def train(self, features, target):
                self.calls = 0

            def predict(self, frame):
                self.calls += 1
                assets = frame.columns.get_level_values(1).unique()
                scale = frame.iloc[-1, 0] if uses_future and len(frame) else 1.0
                result = pd.DataFrame(np.tile([scale, -scale], (len(frame), 1)),
                                      index=frame.index, columns=assets)
                if not uses_future and self.calls == 1:
                    self.buffer = result
                if not uses_future and self.calls == 2:
                    self.buffer.iloc[:, :] = -result.to_numpy()
                    return self.buffer
                return result

        hours = pd.Index(np.arange(160), name="hour")
        assets = pd.Index(["a", "b"], name="asset")
        columns = pd.MultiIndex.from_product([["Feature.1"], assets])
        features = pd.DataFrame(1.0, index=hours, columns=columns)
        target = pd.DataFrame(0.0, index=hours, columns=assets)
        for uses_future, expected_failure in ((False, "repeat"), (True, "future_perturbation")):
            with self.subTest(expected_failure=expected_failure), workspace() as directory:
                journal = Journal(directory)
                journal.period = "001"
                with patch.object(worker, "reload_in_child", side_effect=lambda _, model, frame: model.predict(frame)):
                    worker.behavior_panel(journal, ProbePredictor, features, target, 80)
                checks = {event["check"]["name"]: event["check"]["passed"]
                          for event in validator.read_events(journal.path) if event["event"] == "check"}
                self.assertFalse(checks[expected_failure])


if __name__ == "__main__":
    unittest.main()
