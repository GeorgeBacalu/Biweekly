"""Observe the shipped scorer from one isolated CAP-2 worker process."""

import ctypes
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from importlib.machinery import SourceFileLoader
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

if __package__:
    from .validate_walk_forward import (
        CITY_WORLD, CPU_ENV, DATA_DIR, GAUGE_DIMENSIONS, PERIODS, PREDICT_LIMIT, REPRESENTATIVE_PERIODS,
        STARTUP_ALLOWANCE, TRAIN_LIMIT, UNIVERSE, ZERO_SUM_TOLERANCE, atomic_json, check, finite_number,
        fingerprint, read_events, read_json, score_value,
    )
else:
    from validate_walk_forward import (
        CITY_WORLD, CPU_ENV, DATA_DIR, GAUGE_DIMENSIONS, PERIODS, PREDICT_LIMIT, REPRESENTATIVE_PERIODS,
        STARTUP_ALLOWANCE, TRAIN_LIMIT, UNIVERSE, ZERO_SUM_TOLERANCE, atomic_json, check, finite_number,
        fingerprint, read_events, read_json, score_value,
    )


class KitSourceLoader(SourceFileLoader):
    def get_code(self, fullname):
        filename = self.get_filename(fullname)
        return self.source_to_code(self.get_data(filename), filename)


class KitSourceFinder:
    # The staged predictor and baseline still load through the shipped runner.
    modules = frozenset(("runner", "walkforward", "data_loader", "evaluation", "predictor", "city_tools"))

    def __init__(self, kit):
        self.kit = kit

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.modules:
            source = str(self.kit / f"{fullname}.py")
            return importlib.util.spec_from_file_location(fullname, source, loader=KitSourceLoader(fullname, source))
        return None


def peak_rss_bytes():
    if os.name != "nt":
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)

    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("size", wintypes.DWORD), ("page_faults", wintypes.DWORD),
            ("peak_working_set", ctypes.c_size_t), ("working_set", ctypes.c_size_t),
            ("peak_paged_pool", ctypes.c_size_t), ("paged_pool", ctypes.c_size_t),
            ("peak_nonpaged_pool", ctypes.c_size_t), ("nonpaged_pool", ctypes.c_size_t),
            ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t),
        ]

    current_process = ctypes.windll.kernel32.GetCurrentProcess
    current_process.argtypes = []
    current_process.restype = wintypes.HANDLE
    memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
    memory_info.restype = wintypes.BOOL
    counters = ProcessMemoryCounters()
    counters.size = ctypes.sizeof(counters)
    if not memory_info(current_process(), ctypes.byref(counters), counters.size):
        raise ctypes.WinError()
    return counters.peak_working_set


class Journal:
    def __init__(self, directory):
        self.path = directory / "events.ndjson"
        self.period = None
        self.phase = "startup"
        self.next_call = 0

    def emit(self, event, **fields):
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"event": event, "phase": self.phase, "period": self.period,
                                     **fields}, allow_nan=False) + "\n")

    def enter(self, phase):
        self.phase = phase
        self.emit("phase", peak_rss_bytes=peak_rss_bytes())

    def check(self, name, passed, **details):
        self.emit("check", check=check(name, passed, period=self.period, phase=self.phase, **details))

    def fail(self, exc):
        self.emit("failure", error=exception_text(exc), peak_rss_bytes=peak_rss_bytes())

    def model_call(self, method, function, *args, probe=None, **kwargs):
        self.phase = method
        call_id = self.next_call
        self.next_call += 1
        limit = TRAIN_LIMIT if method == "train" else PREDICT_LIMIT
        self.emit("call_start", id=call_id, limit_seconds=limit, probe=probe,
                  started_monotonic=time.perf_counter(), peak_rss_bytes=peak_rss_bytes())
        started = time.perf_counter()
        error = None
        try:
            return function(*args, **kwargs)
        except BaseException as exc:
            error = exception_text(exc)
            raise
        finally:
            elapsed = time.perf_counter() - started
            self.emit("call_finish", id=call_id, seconds=elapsed, error=error, probe=probe,
                      peak_rss_bytes=peak_rss_bytes())


def exception_text(exc):
    try:
        message = str(exc)
    except Exception:
        message = "<unprintable message>"
    return f"{type(exc).__name__}: {message}"


def load_runtime(path):
    dependencies = ("numpy", "pandas", "pyarrow", "matplotlib", "scipy", "xgboost", "cloudpickle")
    runtime = {
        "python": platform.python_version(), "interpreter": sys.executable,
        "platform": platform.platform(),
        "dependencies": {},
        "environment": {key: os.environ.get(key) for key in CPU_ENV},
    }
    try:
        for name in dependencies:
            runtime["dependencies"][name] = importlib.import_module(name).__version__
    finally:
        atomic_json(path, runtime)
    return runtime


def collect_metadata(journal, output):
    import pyarrow.parquet as pq
    from data_loader import list_periods
    from walkforward import compute_val_rows

    journal.enter("parquet_metadata")
    output["discovered_periods"] = list_periods(DATA_DIR, UNIVERSE)
    expected = output["expected_rows"] = {}
    for period in PERIODS:
        journal.period = period
        filename = Path(DATA_DIR) / f"{period}.{UNIVERSE}.features.parquet"
        expected[period] = compute_val_rows(pq.read_metadata(filename).num_rows)


def assert_probe(journal, name, assertion, **details):
    try:
        assertion()
    except (AssertionError, TypeError, ValueError) as exc:
        journal.check(name, False, error=exception_text(exc), **details)
        return False
    journal.check(name, True, **details)
    return True


def snapshot_axes(frame):
    return frame.index.copy(deep=True), frame.columns.copy(deep=True)


def position_contract(journal, features, positions, probe=None, *, axes=None):
    import numpy as np
    import pandas as pd

    expected_hours, expected_columns = axes if axes is not None else snapshot_axes(features)
    input_hours_ok = assert_probe(journal, "input_hour_axis_unchanged", lambda: pd.testing.assert_index_equal(
        features.index, expected_hours, exact=True), probe=probe)
    input_columns_ok = assert_probe(journal, "input_feature_axis_unchanged", lambda: pd.testing.assert_index_equal(
        features.columns, expected_columns, exact=True), probe=probe)
    frame_ok = isinstance(positions, pd.DataFrame)
    journal.check("positions_dataframe", frame_ok, probe=probe)
    if not frame_ok:
        return False
    assets = expected_columns.get_level_values(1).unique()
    index_ok = assert_probe(journal, "positions_hour_axis", lambda: pd.testing.assert_index_equal(
        positions.index, expected_hours, exact=True), probe=probe)
    columns_ok = assert_probe(journal, "positions_asset_axis", lambda: pd.testing.assert_index_equal(
        positions.columns, assets, exact=True), probe=probe)
    numeric = all(getattr(dtype, "kind", "O") in "iuf" for dtype in positions.dtypes)
    journal.check("positions_real_numeric", numeric, probe=probe)
    if not numeric:
        return False
    values = positions.to_numpy(dtype=float, na_value=np.nan)
    finite = bool(np.isfinite(values).all())
    journal.check("positions_finite", finite, probe=probe)
    max_sum = float(np.abs(values.sum(axis=1)).max(initial=0)) if finite else None
    journal.check("positions_zero_sum", max_sum is not None and np.isfinite(max_sum)
                  and max_sum <= ZERO_SUM_TOLERANCE,
                  max_abs_sum=max_sum if max_sum is not None and np.isfinite(max_sum) else None,
                  tolerance=ZERO_SUM_TOLERANCE, probe=probe)
    return input_hours_ok and input_columns_ok and index_ok and columns_ok and finite


def summarize_result(result):
    if not result:
        return {}
    summary = score_value(result.get("sharpe"))
    for key in ("val_rows", "train_rows", "n_periods", "n_observations"):
        if key in result:
            value = result[key]
            summary[key] = value if type(value) in (int, str) else None
    if "demean_check" in result:
        summary["demean_check"] = bool(result["demean_check"])
    for key in ("error", "error_phase"):
        if result.get(key):
            summary[key] = str(result[key])
    if "error" in summary:
        summary.update(net_sharpe=None, net_sharpe_reason="runner reported an error")
    if isinstance(result.get("per_period"), dict):
        periods = result["per_period"]
        normalized = summary["per_period"] = {}
        for key, entry in periods.items():
            label = key
            if not isinstance(key, str):
                label = f"<{type(key).__name__}: {key!r}>"
                while label in periods or label in normalized:
                    label = f"<{label}>"
            if isinstance(entry, dict):
                normalized[label] = {"n_val": entry["n_val"] if type(entry.get("n_val")) is int else None,
                                     **score_value(entry.get("sharpe"))}
            else:
                normalized[label] = {"n_val": None, "net_sharpe": None,
                                     "net_sharpe_reason": "invalid runner period record"}
    return summary


def validate_city_map(path):
    import numpy as np
    import pandas as pd

    table = pd.read_parquet(path)
    dimensions = [f"city_dim_{number}" for number in range(GAUGE_DIMENSIONS[0])]
    actual_dimensions = {column for column in table.columns if column.startswith("city_dim_")}
    if (table.empty or not table.columns.is_unique or not {"predictor_id", "world"}.issubset(table.columns)
            or actual_dimensions != set(dimensions)):
        raise ValueError("City map requires predictor_id, world and exactly 19 coordinate columns")
    if not bool(table["world"].fillna("").eq(CITY_WORLD).all()):
        raise ValueError(f"City map must contain only {CITY_WORLD} rows")
    ids = table["predictor_id"]
    if not ids.is_unique or not all(isinstance(value, str) and value.strip() for value in ids):
        raise ValueError("City map identifiers must be unique nonempty strings")
    if not all(dtype.kind in "iuf" for dtype in table[dimensions].dtypes):
        raise ValueError("City map coordinates must be real numbers")
    vectors = table[dimensions].to_numpy()
    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or not (norms >= 1e-12).all():
        raise ValueError("City map vectors must have finite coordinates and a finite norm of at least 1e-12")
    return set(ids)


class GaugeLoader:
    """Observe NumPy loads in the shipped walk-forward module only."""

    def __init__(self, numpy, load):
        self.numpy = numpy
        self.load = load

    def __getattr__(self, name):
        return getattr(self.numpy, name)


class NoveltyObserver:
    def __init__(self, journal, request, city_tools):
        self.journal = journal
        self.expected = request["expected_rows"]
        self.map_path = Path(request["map_path"])
        self.map_sha = request["map_sha256"]
        self.expected_gauges = request["gauge_sha256"]
        self.compute = city_tools.compute_gauge_fixed
        self.compare = city_tools.compare_to_cities
        self.identifiers = set()
        self.compared_sha = None
        self.fixed = {}
        self.observed_gauges = {}

    def validate_map(self):
        journal = self.journal
        journal.enter("city_map_validation")
        try:
            if fingerprint(self.map_path)["sha256"] != self.map_sha:
                raise ValueError("Staged map digest does not match the novelty request")
            identifiers = validate_city_map(self.map_path)
            if fingerprint(self.map_path)["sha256"] != self.map_sha:
                raise ValueError("Staged map changed during validation")
            self.identifiers = identifiers
            journal.check("city_map_valid", True, rows=len(identifiers), sha256=self.map_sha, world=CITY_WORLD)
        except Exception as exc:
            journal.check("city_map_valid", False, error=exception_text(exc))

    def load_gauge(self, filename):
        import numpy as np

        path = Path(filename)
        journal = self.journal
        journal.enter("raw_gauge_validation")
        period = journal.period
        expected_sha = self.expected_gauges.get(path.as_posix())
        details = {"path": path.as_posix(), "expected_shape": (self.expected.get(period), *GAUGE_DIMENSIONS),
                   "expected_sha256": expected_sha, "sha256": None}
        self.observed_gauges[period] = None
        try:
            contents = path.read_bytes()
            sha = hashlib.sha256(contents).hexdigest()
            self.observed_gauges[period] = details["sha256"] = sha
            if sha != expected_sha:
                raise ValueError("Raw gauge digest differs from the before-scoring manifest")
            gauge = np.load(BytesIO(contents), allow_pickle=False)
            details.update(shape=gauge.shape, dtype=str(gauge.dtype))
            nonfinite = int(np.count_nonzero(~np.isfinite(gauge))) if gauge.dtype.kind in "iuf" else None
            details.update(nonfinite_values=nonfinite)
            valid = gauge.shape == details["expected_shape"] and nonfinite == 0
            journal.check("raw_gauge", valid, **details)
            return gauge
        except Exception as exc:
            journal.check("raw_gauge", False, error=exception_text(exc), **details)
            raise

    def gauge_fixed(self, predictions, matrices):
        import numpy as np

        journal = self.journal
        journal.enter("gauge_fix")
        expected = (self.expected.get(journal.period), GAUGE_DIMENSIONS[0])
        try:
            vectors = self.compute(predictions, matrices)
        except Exception as exc:
            journal.check("period_gauge_fixed", False, expected_shape=expected, error=exception_text(exc))
            raise
        valid = vectors.shape == expected and bool(np.isfinite(vectors).all())
        self.fixed.setdefault(journal.period, []).append(vectors.shape[0] if valid else None)
        journal.check("period_gauge_fixed", valid, shape=vectors.shape, expected_shape=expected)
        return vectors

    def compare_to_map(self, city):
        import numpy as np

        journal = self.journal
        journal.period = None
        journal.enter("city_comparison")
        try:
            if not self.identifiers:
                raise ValueError("No validated city map is available")
            if fingerprint(self.map_path)["sha256"] != self.map_sha:
                raise ValueError("Staged map changed before comparison")
            comparison = self.compare(city, city_db_path=str(self.map_path))
            if fingerprint(self.map_path)["sha256"] != self.map_sha:
                raise ValueError("Staged map changed during comparison")
            distances = comparison["angular_distance_deg"].to_numpy()
            covers_map = len(comparison) == len(self.identifiers) and set(comparison["signal_id"]) == self.identifiers
            valid = covers_map and bool(np.isfinite(distances).all()) and bool(((0 <= distances) & (distances <= 180)).all())
            journal.check("map_comparison", valid, rows=len(comparison), expected_rows=len(self.identifiers))
            self.compared_sha = self.map_sha
            return comparison
        except Exception as exc:
            journal.check("map_comparison", False, error=exception_text(exc))
            raise

    def summarize(self, result):
        import numpy as np

        journal = self.journal
        journal.enter("novelty_result")
        for period in sorted(self.expected.keys() - self.observed_gauges.keys()):
            journal.period = period
            path = f"{DATA_DIR}/gauge/{period}.{UNIVERSE}.gauge.npy"
            journal.check("raw_gauge", False, path=path, expected_sha256=self.expected_gauges.get(path),
                          sha256=None, error="Native scorer did not load the expected gauge file")
        journal.period = None
        vectors = result.get("gauge_fixed")
        shape = list(vectors.shape) if isinstance(vectors, np.ndarray) else None
        expected_shape = [sum(self.expected.values()), GAUGE_DIMENSIONS[0]]
        journal.check("gauge_aggregate", shape == expected_shape and bool(np.isfinite(vectors).all()),
                      shape=shape, expected_shape=expected_shape)
        expected_periods = {period: [rows] for period, rows in self.expected.items()}
        journal.check("gauge_period_coverage", self.fixed == expected_periods,
                      actual=self.fixed, expected=expected_periods)
        distance = finite_number(result.get("city_novelty_deg"))
        nearest = result.get("nearest_city")
        table = result.get("city_comparison")
        compared_rows = len(table) if table is not None else 0
        journal.check("nearest_city", isinstance(nearest, str) and nearest in self.identifiers
                      and distance is not None and 0 <= distance <= 180 and compared_rows > 0
                      and nearest == table.iloc[0]["signal_id"] and distance == table.iloc[0]["angular_distance_deg"])
        journal.emit("novelty_result", observation={
            "city_novelty_deg": distance, "nearest_city": nearest if isinstance(nearest, str) else None,
            "map_sha256": self.compared_sha, "map_rows": len(self.identifiers), "comparison_rows": compared_rows,
            "gauge_fixed_shape": shape, "gauge_fixed_rows": shape[0] if shape else 0,
            "gauge_fixed_per_period": self.fixed, "gauge_sha256_per_period": self.observed_gauges,
            "gauge_fixed_observed_rows": sum(rows for counts in self.fixed.values() for rows in counts if rows),
        })


def score(journal, artifact, directory, mode, role, output):
    import numpy as np
    import runner
    import walkforward

    exposure = None
    walkforward_result = {}
    shipped_load = runner.load_predictor_class_from_file
    shipped_period = walkforward.load_period
    shipped_backtest = walkforward.backtest
    full = mode in ("full", "novelty")
    entry_point = "walkforward_full" if full else "walkforward_backtest"
    shipped_walkforward = getattr(runner, entry_point)
    novelty = None
    if mode == "novelty":
        import city_tools

        novelty = NoveltyObserver(journal, read_json(directory / "novelty-inputs.json"), city_tools)
        novelty.validate_map()

    def load_period(period, *args, **kwargs):
        nonlocal exposure
        journal.period = period
        exposure = None
        journal.enter("load_period")
        return shipped_period(period, *args, **kwargs)

    def load_predictor(path):
        journal.enter("load_predictor")
        try:
            predictor = shipped_load(path)
        except BaseException as exc:
            journal.fail(exc)
            raise
        original_init = predictor.__init__
        original_train = predictor.train
        original_predict = predictor.predict

        def initialize(instance, *args, **kwargs):
            journal.enter("init")
            return original_init(instance, *args, **kwargs)

        def train(instance, *args, **kwargs):
            return journal.model_call("train", original_train, instance, *args, **kwargs)

        def predict(instance, features, *args, **kwargs):
            nonlocal exposure
            axes = snapshot_axes(features) if role == "candidate" else None
            positions = journal.model_call("predict", original_predict, instance, features, *args, **kwargs)
            if role == "candidate" and position_contract(journal, features, positions, axes=axes) and mode == "full":
                gross = np.abs(positions.to_numpy(dtype=float)).sum(axis=1)
                valid = bool(np.isfinite(gross).all()) and positions.index.is_monotonic_increasing
                valid = valid and positions.index.is_unique and positions.index.dtype.kind in "iu"
                journal.check("exposure_inputs", valid)
                if valid:
                    exposure = [[int(hour), float(value)] for hour, value in zip(positions.index, gross)]
            return positions

        predictor.__init__ = initialize
        predictor.train = train
        predictor.predict = predict
        original_generate = getattr(predictor, "generate_features", None)
        if original_generate is not None:
            def generate_features(instance, *args, **kwargs):
                journal.enter("generate_features")
                return original_generate(instance, *args, **kwargs)

            predictor.generate_features = generate_features
        return predictor

    def backtest(predictions, returns):
        journal.enter("backtest")
        pnl = shipped_backtest(predictions, returns)
        std = pnl.std()
        sharpe = float(pnl.mean() / std) if std > 0 else 0.0
        journal.emit("scored_period", observation={"rows": len(pnl), **score_value(sharpe)})
        if exposure is not None:
            journal.check("exposure_scored_rows", len(exposure) == len(pnl))
            if len(exposure) == len(pnl):
                journal.emit("exposure", rows=exposure)
        return pnl

    def walkforward_with_result(*args, **kwargs):
        nonlocal walkforward_result
        result = shipped_walkforward(*args, **kwargs)
        walkforward_result = result
        output["runner_result"] = summarize_result(result)
        journal.emit("runner_result", result=output["runner_result"])
        journal.enter("runner_output")
        return result

    with (ExitStack() as hooks,
          patch.object(runner, "load_predictor_class_from_file", load_predictor),
          patch.object(runner, entry_point, walkforward_with_result),
          patch.object(walkforward, "load_period", load_period),
          patch.object(walkforward, "backtest", backtest)):
        if novelty is not None:
            hooks.enter_context(patch.object(city_tools, "compute_gauge_fixed", novelty.gauge_fixed))
            hooks.enter_context(patch.object(city_tools, "compare_to_cities", novelty.compare_to_map))
            hooks.enter_context(patch.object(walkforward, "np", GaugeLoader(np, novelty.load_gauge)))
        result = {}
        try:
            result = runner.validate_submission(
                str(artifact), output_file=str(directory / "runner.csv"), data_dir=DATA_DIR,
                universe=UNIVERSE, full=full, gauge_fix=novelty is not None,
                max_train_time=TRAIN_LIMIT, max_predict_time=PREDICT_LIMIT,
            )
        finally:
            if novelty is not None:
                period, phase = journal.period, journal.phase
                novelty.summarize(walkforward_result or result)
                journal.period, journal.phase = period, phase
    output["runner_result"] = summarize_result(result)
    if not result:
        journal.check("predictor_loaded", False, error="runner returned an empty result")


def behavior(journal, artifact):
    from data_loader import load_period
    from runner import load_predictor_class_from_file
    from walkforward import compute_val_rows

    journal.enter("load_predictor")
    predictor = load_predictor_class_from_file(str(artifact))
    for period in REPRESENTATIVE_PERIODS:
        journal.period = period
        journal.enter("behavior_panel")
        try:
            data = load_period(period, DATA_DIR, UNIVERSE)
            boundary = len(data["features"]) - compute_val_rows(len(data["features"]))
            behavior_panel(journal, predictor, data["features"], data["target"], boundary)
        except Exception as exc:
            journal.fail(exc)
            traceback.print_exc()


def reload_in_child(journal, model, frame):
    import cloudpickle

    journal.enter("reload")
    directory = journal.path.parent / f"reload-{journal.period}"
    directory.mkdir()
    stderr_path = directory / "stderr.log"
    arguments = [sys.executable, "-u", str(Path(__file__).resolve()), "reload",
                 type(model).__module__, "", str(directory)]
    exit_code = None
    try:
        with stderr_path.open("wb") as stderr:
            process = subprocess.run(
                arguments, input=cloudpickle.dumps((model, frame)), stdout=subprocess.PIPE, stderr=stderr,
                timeout=STARTUP_ALLOWANCE + PREDICT_LIMIT,
            )
        exit_code = process.returncode
        if exit_code != 0:
            raise RuntimeError(f"Standalone reload worker exited with code {exit_code}")
        return cloudpickle.loads(process.stdout)
    finally:
        events = read_events(directory / "events.ndjson")
        for event in events:
            if event["event"] in ("call_start", "call_finish"):
                details = {key: value for key, value in event.items() if key not in ("event", "period", "id")}
                journal.emit(event["event"], id=journal.next_call, process="reload", **details)
                if event["event"] == "call_finish":
                    journal.next_call += 1
            elif event["event"] == "check":
                journal.emit("check", check={**event["check"], "period": journal.period, "process": "reload"})
        if any(event["event"] == "call_start" for event in events) and not any(
            event["event"] == "call_finish" for event in events
        ):
            journal.next_call += 1
        peak = max((event["peak_rss_bytes"] for event in events if "peak_rss_bytes" in event), default=None)
        journal.emit("reload_worker", exit_code=exit_code, peak_rss_bytes=peak,
                     measured_at_worker_exit=any(event["event"] == "worker_exit" for event in events),
                     journal_path=journal.path.parent.name + f"/{directory.name}/events.ndjson",
                     stderr_path=journal.path.parent.name + f"/{directory.name}/stderr.log")


def reload_prediction(journal, module_name):
    import cloudpickle

    journal.enter("reload")
    standalone = importlib.util.find_spec(module_name) is None
    journal.check("standalone_reload", standalone, submission_module=module_name)
    if not standalone:
        raise RuntimeError(f"Submission module {module_name} is importable in the reload worker")
    model, frame = cloudpickle.load(sys.stdin.buffer)
    axes = snapshot_axes(frame)
    positions = journal.model_call("predict", model.predict, frame, probe="reload")
    position_contract(journal, frame, positions, "reload", axes=axes)
    cloudpickle.dump(positions, sys.stdout.buffer)


def behavior_panel(journal, predictor, features, target, boundary):
    import pandas as pd

    train = features.iloc[:boundary]
    labels = target.iloc[:boundary]
    evaluation = features.iloc[boundary - 48:boundary + 48]
    core = evaluation.iloc[24:-24]
    if len(core) != 48:
        raise ValueError("Representative behavior panel needs 48 hours either side of the split")
    journal.emit("behavior_panel", train_rows=len(train), probe_rows=len(evaluation),
                 first_probe_hour=int(evaluation.index[0]), last_probe_hour=int(evaluation.index[-1]),
                 first_holdout_hour=int(features.index[boundary]))

    def fit(probe, training=train, outcomes=labels):
        journal.enter("init")
        model = predictor()
        journal.model_call("train", model.train, training, outcomes, probe=probe)
        return model

    def predict(model, frame, probe):
        axes = snapshot_axes(frame)
        result = journal.model_call("predict", model.predict, frame, probe=probe)
        position_contract(journal, frame, result, probe, axes=axes)
        if isinstance(result, pd.DataFrame):
            hours, assets = snapshot_axes(result)
            return result.copy(deep=True).set_axis(hours).set_axis(assets, axis=1)
        return result

    def same(name, actual, expected):
        return assert_probe(journal, name, lambda: pd.testing.assert_frame_equal(
            actual, expected, check_exact=False, atol=ZERO_SUM_TOLERANCE, rtol=0,
        ))

    model = fit("reference")
    reference = predict(model, evaluation, "reference")
    same("repeat", predict(model, evaluation, "repeat"), reference)
    try:
        restored = reload_in_child(journal, model, evaluation)
        same("reload", restored, reference)
    except Exception as exc:
        journal.check("reload", False, error=exception_text(exc))

    padded = features.iloc[boundary - 72:boundary + 72]
    same("equivalent_padding", predict(model, padded, "equivalent_padding").loc[core.index],
         reference.loc[core.index])
    changed_future = evaluation.copy(deep=True)
    changed_future.iloc[-24:] = -changed_future.iloc[-24:].to_numpy()
    same("future_perturbation", predict(model, changed_future, "future_perturbation").loc[:core.index[-1]],
         reference.loc[:core.index[-1]])

    for axis in (0, 1):
        probe = f"predict_axis_{axis}_permutation"
        reordered = evaluation.sample(frac=1, axis=axis, random_state=41 + axis)
        actual = predict(model, reordered, probe).reindex(index=reference.index, columns=reference.columns)
        same(probe, actual, reference)
        for frame_name, frame in (("features", train), ("target", labels)):
            probe = f"train_{frame_name}_axis_{axis}_permutation"
            reordered = frame.sample(frac=1, axis=axis, random_state=73 + axis)
            reordered_model = (fit(probe, training=reordered) if frame_name == "features"
                               else fit(probe, outcomes=reordered))
            same(probe, predict(reordered_model, evaluation, probe), reference)

    empty = evaluation.iloc[:0]
    predict(model, empty, "empty_frame")
    short = core.iloc[:3]
    short_result = predict(model, short, "short_frame")
    single_hours = pd.concat([predict(model, short.iloc[i:i + 1], "single_hour") for i in range(len(short))])
    expected = pd.DataFrame([single_hours.loc[hour - 23:hour].mean(axis=0).to_numpy()
                             for hour in short.index], index=short.index, columns=single_hours.columns)
    same("short_frame", short_result, expected)

    overlap = train.iloc[-48:]
    shift = int(evaluation.index[-1]) - int(overlap.index[0]) + 1
    outside = overlap.set_axis(overlap.index + shift)
    observed = predict(model, overlap, "training_overlap")
    relabeled = predict(model, outside, "outside_training").set_axis(overlap.index)
    same("training_overlap", relabeled, observed)


def main():
    mode, role, artifact_path, directory = sys.argv[1:]
    artifact = Path(artifact_path) if artifact_path else None
    directory = Path(directory)
    journal = Journal(directory)
    output = {}
    exit_code = 0
    kit = Path.cwd()
    sys.path.insert(0, str(kit))
    sys.meta_path.insert(0, KitSourceFinder(kit))
    try:
        if mode == "reload":
            reload_prediction(journal, role)
        else:
            journal.enter("runtime")
            runtime = load_runtime(directory / "runtime.json")
            version = runtime["dependencies"]["xgboost"]
            supports_baseline = tuple(int(part) for part in version.split(".")[:2]) >= (3, 2)
            journal.check("xgboost_compatible", supports_baseline, version=version, minimum="3.2")
            if not supports_baseline:
                raise RuntimeError("The shipped baseline requires XGBoost 3.2 or newer")
            if mode == "metadata":
                collect_metadata(journal, output)
            else:
                journal.enter("staged_artifact")
                contents = artifact.read_bytes()
                snapshot = directory / "scored-artifact" / artifact.name
                snapshot.parent.mkdir()
                snapshot.write_bytes(contents)
                output["artifact_sha256"] = hashlib.sha256(contents).hexdigest()
                output["artifact_snapshot"] = snapshot.relative_to(directory).as_posix()
                journal.emit("artifact_hashed", sha256=output["artifact_sha256"], snapshot=output["artifact_snapshot"])
                if mode == "behavior":
                    behavior(journal, snapshot)
                else:
                    score(journal, snapshot, directory, mode, role, output)
    except BaseException as exc:
        journal.fail(exc)
        traceback.print_exc()
        exit_code = 1
    finally:
        output["peak_rss_bytes"] = peak_rss_bytes()
        journal.emit("worker_exit", peak_rss_bytes=output["peak_rss_bytes"], exit_code=exit_code)
        atomic_json(directory / "result.json", output)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
