import codecs
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import chdir
from pathlib import Path
from unittest.mock import patch

import cloudpickle
import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "requirements"
sys.path.insert(0, str(KIT))

from data_loader import list_periods
from runner import load_predictor_class_from_file, validate_submission


SUBMISSIONS = ROOT / "submissions"
VARIANTS = ("weighted_vote.py", "weighted_vote_control.py")
FEATURES = tuple(f"Feature.{number}" for number in range(1, 7))
ASSETS = tuple(f"asset.{number:02}" for number in range(20))
PIP_DISABLED = {"COMP6_DISABLE_PIP_INSTALL": "1"}


def panel(hours, seed):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(len(hours), len(FEATURES), len(ASSETS)))
    index = pd.Index(hours, name="hour")
    columns = pd.MultiIndex.from_product([FEATURES, ASSETS], names=["feature", "asset"])
    features = pd.DataFrame(values.reshape(len(index), len(columns)), index=index, columns=columns)
    weights = [3.0, -2.5, 1.5, 1.5, -1.0, -0.8]
    signal = np.einsum("hfa,f->ha", values, weights) / 10
    signal -= signal.mean(axis=1, keepdims=True)
    return features, pd.DataFrame(signal, index=index, columns=pd.Index(ASSETS, name="asset"))


class WeightedVoteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, PIP_DISABLED):
            cls.variants = {
                filename: load_predictor_class_from_file(str(SUBMISSIONS / filename))
                for filename in VARIANTS
            }

    def assert_position_contract(self, positions, features):
        pd.testing.assert_index_equal(positions.index, features.index, exact=True)
        assets = features.columns.get_level_values(1).unique()
        pd.testing.assert_index_equal(positions.columns, assets, exact=True)
        self.assertTrue(np.isfinite(positions.to_numpy()).all())
        np.testing.assert_allclose(positions.sum(axis=1), 0.0, atol=1e-12, rtol=0)

    def assert_same_positions(self, actual, expected):
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, atol=1e-12, rtol=0)

    def test_centered_ranks_average_ties_and_neutralize_non_finite_values(self):
        model = self.variants[VARIANTS[0]]()
        block = pd.DataFrame([
            [1.0, 1.0, 5.0, 9.0, np.inf, -np.inf, pd.NA] + [np.nan] * 13,
            [np.nan] * 20,
            [1.0] + [np.nan] * 19,
        ])
        expected = np.zeros((3, 20))
        expected[0, :4] = [-2 / 3, -2 / 3, 1 / 3, 1]
        np.testing.assert_allclose(model._centered_ranks(block), expected, atol=1e-12, rtol=0)

    def test_aligned_fit_keeps_zero_targets_and_builds_the_expected_vote(self):
        features, target = panel(range(56), seed=941)
        target.iloc[0] = 0.0
        target.iloc[2, 3] = np.nan
        target.iloc[4, 5] = np.inf
        target = target.drop(index=7, columns=ASSETS[6])
        expected_labels = target.to_numpy().ravel()
        expected_labels = expected_labels[np.isfinite(expected_labels)]
        target = target.astype(object)
        target.iloc[2, 3] = pd.NA
        target = target.sample(frac=1, random_state=45).sample(frac=1, axis=1, random_state=83)
        direction = np.array([1, -1, 1, 1, -1, -1])
        probe = np.array([
            [1, 0.5, 0.75, -1, 0.25, -0.5],
            [0.25, -1, 0, -0.25, -0.75, 1],
        ])

        for filename, predictor_type in self.variants.items():
            with self.subTest(filename=filename):
                model = predictor_type()
                with patch.object(xgb, "DMatrix", wraps=xgb.DMatrix) as dmatrix:
                    model.train(features, target)
                np.testing.assert_array_equal(dmatrix.call_args.kwargs["label"], expected_labels)
                self.assertEqual(model._fit[1], dict(zip(FEATURES, direction)))
                design = model._model_input(probe, model._fit[1])
                np.testing.assert_array_equal(design[:, :6], probe)
                expected_vote = [1 / 12, 1 / 8] if filename == VARIANTS[0] else [0.0, 0.0]
                np.testing.assert_allclose(design[:, 6], expected_vote, atol=1e-12, rtol=0)

                model.train(features, -target)
                self.assertEqual(model._fit[1], dict(zip(FEATURES, -direction)))

    def test_exactly_unassociated_ranks_get_neutral_orientations(self):
        base, labels = panel(range(32), seed=104)
        opposite = (-base).set_axis(base.index + len(base))
        features = pd.concat([base, opposite])
        target = pd.concat([labels, labels.set_axis(opposite.index)])
        reordered = features.sample(frac=1, random_state=8).sample(frac=1, axis=1, random_state=9)
        reordered_target = target.sample(frac=1, random_state=10).sample(frac=1, axis=1, random_state=11)

        for filename, predictor_type in self.variants.items():
            with self.subTest(filename=filename):
                model = predictor_type()
                for frame, aligned_labels in ((features, target), (reordered, reordered_target)):
                    model.train(frame, aligned_labels)
                    self.assertEqual(model._fit[1], dict.fromkeys(FEATURES, 0.0))

    def test_vote_reaches_both_booster_fit_and_inference(self):
        train, _ = panel(range(96), seed=762)
        evaluation, _ = panel([100, 124, 148], seed=208)
        directions = [1, -1, 1, 1, -1, -1]

        def design(frame, use_vote):
            ranks = np.column_stack([
                (frame[label].rank(axis=1).to_numpy().ravel() - 10.5) / 9.5
                for label in FEATURES
            ])
            vote = np.zeros(len(ranks))
            if use_vote:
                for column, direction in zip(ranks.T, directions):
                    vote += column * direction
                vote /= len(FEATURES)
            return np.column_stack((ranks, vote))

        labels = np.sign(design(train, True)[:, 6])
        target = pd.DataFrame(labels.reshape(len(train), len(ASSETS)), index=train.index, columns=ASSETS)
        for filename, predictor_type in self.variants.items():
            with self.subTest(filename=filename):
                model = predictor_type()
                with patch.object(xgb, "train", wraps=xgb.train) as fit:
                    model.train(train, target)
                use_vote = filename == VARIANTS[0]
                self.assertEqual("f6" in model._fit[0].get_score(), use_vote)
                reference = xgb.train(
                    fit.call_args.args[0], xgb.DMatrix(design(train, use_vote), label=labels, nthread=4),
                    **fit.call_args.kwargs,
                )
                raw = reference.inplace_predict(design(evaluation, use_vote))
                if use_vote:
                    omitted = reference.inplace_predict(design(evaluation, False))
                    self.assertGreater(np.max(np.abs(raw - omitted)), 0.0)
                expected = raw.astype(np.float64).reshape(len(evaluation), len(ASSETS))
                expected -= expected.mean(axis=1, keepdims=True)
                np.testing.assert_allclose(model.predict(evaluation), expected, rtol=0, atol=1e-12)

    def test_independent_axis_permutations_and_local_ticker_labels(self):
        train, target = panel(range(56), seed=302)
        evaluation, _ = panel([39, 40, 45, 46, 55, 68, 69, 79], seed=71)
        reordered_train = train.sample(frac=1, random_state=5).sample(frac=1, axis=1, random_state=77)
        reordered_target = target.sample(frac=1, random_state=95).sample(frac=1, axis=1, random_state=11)
        reordered_eval = evaluation.sample(frac=1, random_state=48).sample(frac=1, axis=1, random_state=74)
        renamed_assets = {asset: f"unseen.{asset}" for asset in ASSETS}

        for filename, predictor_type in self.variants.items():
            with self.subTest(filename=filename):
                reference = predictor_type()
                reference.train(train, target)
                expected = reference.predict(evaluation)
                model = predictor_type()
                model.train(reordered_train, reordered_target)
                self.assertEqual(reference._fit[1], model._fit[1])

                predictions = model.predict(reordered_eval)
                self.assert_position_contract(predictions, reordered_eval)
                self.assert_same_positions(
                    predictions.reindex(index=expected.index, columns=expected.columns), expected
                )

                unseen = evaluation.rename(columns=renamed_assets, level="asset")
                predictions = model.predict(unseen)
                self.assert_position_contract(predictions, unseen)
                self.assert_same_positions(predictions, expected.rename(columns=renamed_assets))

    def test_causal_hour_windows_repeat_and_reload(self):
        train, target = panel(range(56), seed=176)
        evaluation, _ = panel([*range(2, 28), 48, 49, 50, 74, 75], seed=80)
        evaluation.iloc[2, 0:4] = [np.nan, np.inf, -np.inf, np.nan]
        evaluation.iloc[6, :20] = np.nan
        snapshot = evaluation.copy(deep=True)
        recent = evaluation.loc[24:]
        with_changed_future = evaluation.copy()
        with_changed_future.loc[50:] *= -5
        before, _ = panel([-60, -25], seed=81)
        after, _ = panel([76, 77], seed=82)
        padded = pd.concat([before, evaluation, after])

        for filename, predictor_type in self.variants.items():
            with self.subTest(filename=filename):
                model = predictor_type()
                model.train(train, target)
                booster_before = model._fit[0].save_raw()
                directions_before = model._fit[1].copy()
                full = model.predict(evaluation)
                self.assert_position_contract(full, evaluation)
                self.assertGreater(np.abs(full.to_numpy()).max(), 0.0)
                single_hour_frames = []
                for hour in evaluation.index:
                    one_row = evaluation.loc[[hour]]
                    prediction = model.predict(one_row)
                    self.assert_position_contract(prediction, one_row)
                    single_hour_frames.append(prediction)
                single_hours = pd.concat(single_hour_frames)
                means = [single_hours.loc[hour - 23:hour].mean(axis=0) for hour in evaluation.index]
                expected = pd.DataFrame(means, index=full.index, columns=full.columns)
                self.assert_same_positions(full, expected)

                recent_predictions = model.predict(recent)
                self.assert_position_contract(recent_predictions, recent)
                self.assert_same_positions(recent_predictions.loc[48:], full.loc[48:])
                self.assert_same_positions(model.predict(with_changed_future).loc[:49], full.loc[:49])
                padded_predictions = model.predict(padded)
                self.assert_position_contract(padded_predictions, padded)
                self.assert_same_positions(padded_predictions.loc[evaluation.index], full)
                empty = evaluation.iloc[:0]
                self.assert_position_contract(model.predict(empty), empty)
                self.assert_same_positions(model.predict(evaluation), full)
                pd.testing.assert_frame_equal(evaluation, snapshot)
                self.assertEqual(model._fit[1], directions_before)
                self.assertEqual(model._fit[0].save_raw(), booster_before)

                self.assertNotIn(predictor_type.__module__, sys.modules)
                restored = cloudpickle.loads(cloudpickle.dumps(model))
                self.assert_same_positions(restored.predict(evaluation), full)
                self.assert_same_positions(restored.predict(recent), recent_predictions)

                reload_script = (
                    "import importlib.util, sys, cloudpickle\n"
                    "assert importlib.util.find_spec(sys.argv[1]) is None\n"
                    "model, features = cloudpickle.load(sys.stdin.buffer)\n"
                    "cloudpickle.dump(model.predict(features), sys.stdout.buffer)\n"
                )
                process = subprocess.run(
                    [sys.executable, "-c", reload_script, predictor_type.__module__],
                    input=cloudpickle.dumps((model, evaluation)), capture_output=True, cwd=KIT,
                )
                self.assertEqual(process.returncode, 0, process.stderr.decode(errors="replace"))
                self.assert_same_positions(cloudpickle.loads(process.stdout), full)

    def test_sparse_narrow_integer_hours_do_not_wrap(self):
        train, target = panel(range(56), seed=318)
        hours = pd.Index([-128, -120, 0, 125, 126], dtype=np.int8)
        evaluation, _ = panel(hours, seed=319)
        for filename, predictor_type in self.variants.items():
            with self.subTest(filename=filename):
                model = predictor_type()
                model.train(train, target)
                positions = model.predict(evaluation)
                self.assert_position_contract(positions, evaluation)
                for block in (evaluation.iloc[:2], evaluation.iloc[2:3], evaluation.iloc[3:]):
                    self.assert_same_positions(positions.loc[block.index], model.predict(block))

    def test_prediction_snapshots_the_fit_before_retraining(self):
        train, target = panel(range(56), seed=501)
        evaluation, _ = panel([100, 101], seed=502)
        model = self.variants[VARIANTS[0]]()
        model.train(train, target)
        initial = model.predict(evaluation)
        rank_matrix = model._rank_matrix

        def rank_with_retrain(features, assets):
            if features is evaluation:
                model.train(train, -target)
            return rank_matrix(features, assets)

        with patch.object(model, "_rank_matrix", side_effect=rank_with_retrain):
            self.assert_same_positions(model.predict(evaluation), initial)
        updated = model.predict(evaluation)
        self.assertGreater(np.max(np.abs(updated.to_numpy() - initial.to_numpy())), 0.0)

        fitted = model._fit
        with self.assertRaisesRegex(ValueError, "No finite targets after aligning hours and assets"):
            model.train(train, target * np.nan)
        self.assertIs(model._fit, fitted)

    def test_materializer_verifies_bytes_and_keeps_check_read_only(self):
        script = ROOT / "scripts" / "materialize_control.py"
        candidate = (SUBMISSIONS / VARIANTS[0]).read_bytes()
        control = (SUBMISSIONS / VARIANTS[1]).read_bytes()
        for source in (candidate, control):
            self.assertFalse(source.startswith(codecs.BOM_UTF8))
            source.decode("utf-8")
        selector = b"    _USE_WEIGHTED_VOTE = True"
        self.assertEqual(candidate.count(selector), 1)
        self.assertEqual(control, candidate.replace(selector, b"    _USE_WEIGHTED_VOTE = False"))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "submissions").mkdir()
            copied_script = root / "scripts" / script.name
            shutil.copyfile(script, copied_script)
            canonical = root / "submissions" / VARIANTS[0]
            canonical.write_bytes(candidate)
            generated = root / "submissions" / VARIANTS[1]
            command = [sys.executable, str(copied_script)]
            check = [*command, "--check"]

            self.assertNotEqual(subprocess.run(check, capture_output=True).returncode, 0)
            self.assertFalse(generated.exists())
            generated.write_bytes(control + b"\n")
            self.assertNotEqual(subprocess.run(check, capture_output=True).returncode, 0)
            self.assertEqual(generated.read_bytes(), control + b"\n")
            subprocess.run(command, check=True, capture_output=True)
            subprocess.run(check, check=True, capture_output=True)
            self.assertEqual(generated.read_bytes(), control)

            decoy = b'    """\n' + selector + b'\n    """\n    _USE_WEIGHTED_VOTE: bool = True'
            canonical.write_bytes(candidate.replace(selector, decoy))
            rejected = subprocess.run(command, capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(b"exactly one enabled vote selector", rejected.stderr)
            self.assertEqual(generated.read_bytes(), control)

    def test_shipped_quick_runner(self):
        with tempfile.TemporaryDirectory() as directory, chdir(KIT), patch.dict(os.environ, PIP_DISABLED):
            latest = list_periods()[-1]
            features = pd.read_parquet(Path("data/train") / f"{latest}.small.features.parquet")
            for filename in VARIANTS:
                with self.subTest(filename=filename):
                    output = Path(directory) / f"{filename}.csv"
                    results = validate_submission(str(SUBMISSIONS / filename), output_file=str(output))
                    self.assertIsNone(results.get("error"), results.get("error"))
                    self.assertIn("sharpe", results)
                    self.assertTrue(results["demean_check"])
                    self.assertTrue(output.is_file())
                    self.assert_position_contract(results["predictions"], features.iloc[-results["val_rows"]:])


if __name__ == "__main__":
    unittest.main()
