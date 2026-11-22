"""Exercise the provided Streamlit UI through registration, training and inference."""

import json
import time
import uuid
from urllib.error import URLError
from urllib.request import urlopen

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import expect, sync_playwright


TRAIN_CSV = ("feature_1,feature_2,target\n" + "".join(
    f"{i},{i % 7},{2 * i + 3}\n" for i in range(1, 51)
)).encode()
TEST_CSV = b"feature_1,feature_2\n51,2\n52,3\n53,4\n"


def wait_for_http(url: str, timeout: int = 90) -> bytes:
    """Wait for a service to become available."""
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=3) as response:
                return response.read()
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"Сервис {url} недоступен: {last_error}")


def select_option(page, label: str, value: str) -> None:
    """Select a value in a Streamlit selectbox, not a native HTML select."""
    box = page.get_by_test_id("stSelectbox").filter(has=page.get_by_text(label, exact=True))
    control = box.get_by_role("combobox")
    for attempt in range(3):
        control.click()
        try:
            page.get_by_role("option", name=value, exact=True).click(timeout=5000)
            break
        except PlaywrightTimeoutError:
            if attempt == 2:
                raise
    # Streamlit 1.45 renders the selection as text and leaves the search input
    # empty; 1.67 renders it as the combobox input value.
    if box.locator('[data-baseweb="select"]').count():
        expect(box.get_by_text(value, exact=True)).to_be_visible()
    else:
        expect(control).to_have_value(value)
    # The selection is reflected client-side before Streamlit finishes its rerun.
    page.wait_for_timeout(1000)
    expect(page.get_by_test_id("stStatusWidget")).to_be_hidden(timeout=30000)


def upload(page, label: str, name: str, data: bytes) -> None:
    """Upload an in-memory CSV to the named Streamlit file uploader."""
    uploader = page.get_by_test_id("stFileUploader").filter(has=page.get_by_text(label, exact=True))
    uploader.locator('input[type="file"]').set_input_files({
        "name": name, "mimeType": "text/csv", "buffer": data,
    })
    # Wait for the upload-triggered Streamlit rerun before interacting again.
    expect(page.get_by_test_id("stStatusWidget")).to_be_hidden(timeout=30000)


def check_experiment(page, model: str, name: str) -> None:
    """Register one model, select it after reload, train it and request predictions."""
    print(f"Проверка {model}: {name}", flush=True)
    page.get_by_label("Give a name to your experiment").fill(name)
    select_option(page, "Choose a model", model)
    upload(page, "Upload your training CSV file", "train.csv", TRAIN_CSV)
    expect(page.get_by_text("Select target column", exact=True)).to_be_visible(timeout=30000)
    select_option(page, "Select target column", "target")
    page.get_by_role("button", name="Register Experiment").click()
    expect(page.get_by_text(f"Experiment {name} was successfully created!", exact=False)).to_be_visible(timeout=60000)

    # The UI explicitly requires a refresh to populate the experiment selector.
    page.reload(wait_until="domcontentloaded")
    expect(page.get_by_text("Model Training and Prediction", exact=True)).to_be_visible(timeout=60000)
    select_option(page, "Experiment", name)
    expect(page.get_by_role("button", name="Train Model")).to_be_visible(timeout=60000)
    page.get_by_role("button", name="Train Model").click()
    expect(page.get_by_text("Learning Curves", exact=True)).to_be_visible(timeout=120000)
    expect(page.get_by_test_id("stPlotlyChart")).to_be_visible(timeout=30000)
    expect(page.get_by_text("Inference on your data", exact=True)).to_be_visible()
    upload(page, "Upload your test CSV file", "test.csv", TEST_CSV)
    # Client.predict is specified to return a NumPy array; st.write renders it as a dataframe.
    expect(page.get_by_test_id("stDataFrame")).to_be_visible(timeout=60000)
    expect(page.get_by_test_id("stException")).to_have_count(0)
    expect(page.get_by_test_id("stAlert").filter(has_text="Error")).to_have_count(0)

    # A new session must still be able to load the trained experiment.
    page.reload(wait_until="domcontentloaded")
    expect(page.get_by_text("Model Training and Prediction", exact=True)).to_be_visible(timeout=60000)
    select_option(page, "Experiment", name)
    expect(page.get_by_text("Learning Curves", exact=True)).to_be_visible(timeout=60000)
    expect(page.get_by_role("button", name="Train Model")).to_have_count(0)
    select_option(page, "Experiment", "start new")
    expect(page.get_by_role("button", name="Register Experiment")).to_be_visible(timeout=30000)


def main() -> None:
    """Run the browser scenario against the Compose services."""
    response = json.loads(wait_for_http("http://localhost:8000/existing_experiments/"))
    if not isinstance(response.get("experiment_names"), list):
        raise AssertionError("API не вернул список experiment_names")
    wait_for_http("http://localhost:8501/_stcore/health")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        try:
            page.goto("http://localhost:8501", wait_until="domcontentloaded")
            expect(page.get_by_text("Model Training and Prediction", exact=True)).to_be_visible(timeout=60000)
            run_id = uuid.uuid4().hex[:12]
            for suffix, model in (("rf", "Random Forest"), ("gb", "Gradient Boosting")):
                check_experiment(page, model, f"ui_test_{run_id}_{suffix}")
        except Exception:
            print("Streamlit exceptions:", page.get_by_test_id("stException").all_text_contents(), flush=True)
            print("Streamlit alerts:", page.get_by_test_id("stAlert").all_text_contents(), flush=True)
            raise
        finally:
            browser.close()
    print("OK: регистрация, обучение, графики, инференс и повторная загрузка обеих моделей.")


if __name__ == "__main__":
    main()
