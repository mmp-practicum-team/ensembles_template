from __future__ import annotations

import numpy as np
import pytest

from ensembles.boosting import GradientBoostingMSE
from ensembles.random_forest import RandomForestMSE


@pytest.fixture
def regression_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 3))
    y = 2 * X[:, 0] - X[:, 1] + 0.25 * X[:, 2] ** 2
    return X, y


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda: RandomForestMSE(n_estimators=5, tree_params={"max_depth": 3, "random_state": 42}),
        lambda: GradientBoostingMSE(
            n_estimators=5,
            tree_params={"max_depth": 2, "random_state": 42},
            learning_rate=0.1,
        ),
    ],
    ids=["random-forest", "gradient-boosting"],
)
def test_model_fit_predict_smoke(model_factory, regression_data):
    X, y = regression_data
    model = model_factory()

    model.fit(X, y)
    predictions = model.predict(X[:7])

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (7,)
    assert np.isfinite(predictions).all()


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda: RandomForestMSE(n_estimators=3, tree_params={"max_depth": 3, "random_state": 42}),
        lambda: GradientBoostingMSE(
            n_estimators=3,
            tree_params={"max_depth": 2, "random_state": 42},
            learning_rate=0.1,
        ),
    ],
    ids=["random-forest", "gradient-boosting"],
)
def test_model_dump_load_preserves_predictions(model_factory, regression_data, tmp_path):
    X, y = regression_data
    model = model_factory()
    model.fit(X, y)
    expected = model.predict(X[:5])
    assert isinstance(expected, np.ndarray)
    assert expected.shape == (5,)
    assert np.isfinite(expected).all()

    model_path = tmp_path / "model"
    model.dump(str(model_path))
    loaded = type(model).load(str(model_path))

    actual = loaded.predict(X[:5])
    assert isinstance(actual, np.ndarray)
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, equal_nan=False)
