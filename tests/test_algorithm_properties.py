"""Behavioral checks of the ensemble algorithms, independent of random seeds."""

import numpy as np

from ensembles.boosting import GradientBoostingMSE
from ensembles.random_forest import RandomForestMSE


class RecordingTree:
    def __init__(self, scale=0.0):
        self.scale = scale
        self.calls = []

    def fit(self, X, y):
        self.calls.append((np.asarray(X).copy(), np.asarray(y).copy()))
        return self

    def predict(self, X):
        return self.scale * np.asarray(X)[:, 0]


def test_forest_bootstraps_rows_and_preserves_feature_target_alignment():
    # Unique IDs allow us to inspect the sampled rows without fixing an RNG API.
    ids = np.arange(40, dtype=float)
    X = np.column_stack((ids, ids * 2 + 1, -ids))
    y = 3 * ids + 7
    model = RandomForestMSE(n_estimators=20)
    trees = [RecordingTree() for _ in model.forest]
    model.forest = trees
    model.fit(X, y)

    samples = []
    for tree in trees:
        assert len(tree.calls) == 1
        sampled_X, sampled_y = tree.calls[0]
        assert sampled_X.shape == X.shape
        sampled_ids = sampled_X[:, 0].astype(int)
        assert np.all((0 <= sampled_ids) & (sampled_ids < len(X)))
        np.testing.assert_array_equal(sampled_X, X[sampled_ids])
        np.testing.assert_array_equal(sampled_y, y[sampled_ids])
        samples.append(sampled_ids)
    # Across 20 samples, accidental failure of either check is negligible.
    assert any(len(np.unique(sample)) < len(X) for sample in samples)
    assert any(not np.array_equal(samples[0], sample) for sample in samples[1:])


def test_forest_predict_averages_all_trees():
    X = np.array([[1., 2.], [3., 4.], [-2., 0.]])
    model = RandomForestMSE(n_estimators=3)
    model.forest = [RecordingTree(scale=1.), RecordingTree(scale=2.),
                    RecordingTree(scale=-0.5)]
    predictions = model.predict(X)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(X),)
    np.testing.assert_allclose(predictions, (1. + 2. - 0.5) / 3 * X[:, 0])


def test_forest_passes_feature_sampling_parameter_to_each_tree():
    model = RandomForestMSE(n_estimators=4, tree_params={"max_features": 2})
    assert len(model.forest) == 4
    assert all(tree.get_params()["max_features"] == 2 for tree in model.forest)


def test_boosting_fits_successive_residuals_and_scales_predictions():
    X = np.array([[1., 2.], [2., 1.], [3., 0.], [4., -1.]])
    y = np.array([2., 5., 4., 9.])
    rate = 0.25
    model = GradientBoostingMSE(n_estimators=2, learning_rate=rate)
    trees = [RecordingTree(scale=1.0), RecordingTree(scale=-0.5)]
    model.forest = trees
    model.fit(X, y)

    assert len(trees[0].calls) == len(trees[1].calls) == 1
    np.testing.assert_array_equal(trees[0].calls[0][0], X)
    np.testing.assert_array_equal(trees[1].calls[0][0], X)
    initial = np.mean(y)
    np.testing.assert_allclose(model.const_prediction, initial)
    np.testing.assert_allclose(trees[0].calls[0][1], y - initial)
    np.testing.assert_allclose(trees[1].calls[0][1], y - initial - rate * X[:, 0])
    expected = initial + rate * (X[:, 0] - 0.5 * X[:, 0])
    np.testing.assert_allclose(model.predict(X), expected)
