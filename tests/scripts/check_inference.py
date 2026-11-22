"""Check inference through the supplied frontend client on trained UI experiments."""

from io import BytesIO
import os

import numpy as np

from ensembles.frontend import Client


client = Client(os.environ["BASE_URL"])
experiments = [name for name in client.get_names() if name.startswith("ui_test_")]
for suffix in ("_rf", "_gb"):
    matches = [name for name in experiments if name.endswith(suffix)]
    if len(matches) != 1:
        raise AssertionError(f"Ожидался один эксперимент {suffix}, найдено: {matches}")
    data = BytesIO(b"feature_1,feature_2\n51,2\n52,3\n53,4\n")
    data.name = "test.csv"
    predictions = np.asarray(client.predict(matches[0], data))
    if predictions.shape != (3,):
        raise AssertionError(f"{matches[0]}: ожидалось 3 предсказания, форма: {predictions.shape}")
    if not np.isfinite(predictions).all():
        raise AssertionError(f"{matches[0]}: предсказания содержат NaN или бесконечность")
    print(f"OK: {matches[0]}: 3 конечных предсказания для 3 строк CSV")
