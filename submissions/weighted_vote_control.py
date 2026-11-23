"""Season 2 cross-sectional predictor with a weighted feature vote."""

import numpy as np
import pandas as pd
import xgboost as xgb

from predictor import Predictor


class WeightedVotePredictor(Predictor):
    _USE_WEIGHTED_VOTE = False
    _FEATURES = ("Feature.1", "Feature.2", "Feature.3", "Feature.4", "Feature.5", "Feature.6")
    _TRAILING_HOURS = 24
    _THREADS = 4

    def __init__(self):
        self._fit = None

    @staticmethod
    def _asset_order(features: pd.DataFrame) -> pd.Index:
        return features.columns.get_level_values(1).unique()

    @staticmethod
    def _centered_ranks(block: pd.DataFrame) -> np.ndarray:
        finite = block.replace([np.inf, -np.inf], np.nan)
        ranks = finite.rank(axis=1, method="average").to_numpy()
        count = finite.count(axis=1).to_numpy()[:, None]
        centered = (ranks - (count + 1) / 2) / (np.maximum(count - 1, 1) / 2)
        return np.nan_to_num(centered, nan=0.0, copy=False)

    def _rank_matrix(self, features: pd.DataFrame, assets: pd.Index) -> np.ndarray:
        ranks = np.empty((len(features) * len(assets), len(self._FEATURES)))
        for position, label in enumerate(self._FEATURES):
            block = features[label].reindex(columns=assets)
            ranks[:, position] = self._centered_ranks(block).ravel()
        return ranks

    def _model_input(self, ranks: np.ndarray, orientations: dict[str, float]) -> np.ndarray:
        if self._USE_WEIGHTED_VOTE:
            direction = np.array([orientations[label] for label in self._FEATURES])
            # A centered rank is its direction from the median times its distance.
            vote = (ranks * direction).mean(axis=1)
        else:
            vote = np.zeros(len(ranks))
        return np.column_stack((ranks, vote))

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        # Canonical fit order keeps both the association and the learner independent
        # of the order in which the gate supplies rows, blocks, or tickers.
        features = features.sort_index()
        assets = self._asset_order(features).sort_values()
        ranks = self._rank_matrix(features, assets)
        aligned = target.reindex(index=features.index, columns=assets)
        labels = aligned.fillna(np.nan).to_numpy(dtype=np.float64, na_value=np.nan).ravel()
        observed = np.isfinite(labels)
        if not observed.any():
            raise ValueError("No finite targets after aligning hours and assets")
        ranks, labels = ranks[observed], labels[observed]

        centered_labels = labels - labels.mean()
        centered_ranks = ranks - ranks.mean(axis=0)
        association = centered_labels @ centered_ranks
        magnitude = np.abs(centered_labels) @ np.abs(centered_ranks)
        roundoff = np.finfo(np.float64).eps * len(labels) * magnitude
        association[np.abs(association) <= roundoff] = 0.0
        orientations = dict(zip(self._FEATURES, np.sign(association)))
        design = self._model_input(ranks, orientations)
        training = xgb.DMatrix(design, label=labels, nthread=self._THREADS)
        model = xgb.train(
            {
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "device": "cpu",
                "max_depth": 2,
                "eta": 0.05,
                "min_child_weight": 8,
                "reg_lambda": 5.0,
                "seed": 42,
                "nthread": self._THREADS,
            },
            training,
            num_boost_round=30,
        )
        self._fit = (model, orientations)

    def _shape_positions(self, raw: np.ndarray, hours: np.ndarray) -> np.ndarray:
        # Unsigned differences are exact for sorted integer hours across the signed boundary.
        hours = hours.astype(np.uint64, copy=False)
        smoothed = raw.astype(np.float64)
        count = np.ones(len(hours), dtype=np.intp)
        # With distinct integer hours, at most 23 earlier rows can be in the window.
        for lag in range(1, min(len(hours), self._TRAILING_HOURS)):
            within_window = hours[lag:] - hours[:-lag] < self._TRAILING_HOURS
            np.add(smoothed[lag:], raw[:-lag], out=smoothed[lag:], where=within_window[:, None])
            count[lag:] += within_window
        smoothed /= count[:, None]
        smoothed -= smoothed.mean(axis=1, keepdims=True)
        return smoothed

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        fit = self._fit
        if fit is None:
            raise RuntimeError("Call train() before predict()")
        model, orientations = fit
        assets = self._asset_order(features)
        if not len(features):
            return pd.DataFrame(index=features.index, columns=assets, dtype=np.float64)

        ranks = self._rank_matrix(features, assets)
        design = self._model_input(ranks, orientations)
        raw = model.inplace_predict(design).reshape(len(features), len(assets))
        hours = features.index.to_numpy()
        order = np.argsort(hours)
        positions = np.empty(raw.shape, dtype=np.float64)
        positions[order] = self._shape_positions(raw[order], hours[order])
        return pd.DataFrame(positions, index=features.index, columns=assets)
