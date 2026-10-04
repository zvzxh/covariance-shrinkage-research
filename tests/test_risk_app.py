from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


APP = Path(__file__).parents[1] / "src" / "shrinkage_research" / "risk_app.py"


def test_browser_demo_and_downloads_and_stale_results_are_cleared():
    app = AppTest.from_file(APP, default_timeout=20).run()
    assert not app.exception
    app.button(key="analyze").click().run()
    assert not app.exception
    assert app.success
    assert "125" in app.success[0].value
    assert len(app.dataframe) >= 1
    assert len(app.get("download_button")) == 4
    app.number_input(key="annualization").set_value(12).run()
    assert not app.dataframe
    assert not app.get("download_button")
    assert any("旧结果已清空" in message.value for message in app.info)


def test_browser_prices_and_invalid_date_rejection():
    app = AppTest.from_file(APP, default_timeout=20).run()
    app.selectbox(key="kind").set_value("adjusted_prices").run()
    app.button(key="analyze").click().run()
    assert not app.exception
    assert app.success
    app.text_input(key="start").set_value("2024-99-01").run()
    app.button(key="analyze").click().run()
    assert app.error
    assert not app.success
    assert not app.get("download_button")


def test_explicit_weights_use_numeric_controls_and_reject_wrong_sum():
    app = AppTest.from_file(APP, default_timeout=20).run()
    app.checkbox(key="explicit_weights").check().run()
    controls = sorted([widget for widget in app.number_input if widget.key.startswith("asset_weight_")], key=lambda widget: widget.label)
    assert len(controls) == 4
    controls[0].set_value(50).run()
    app.button(key="analyze").click().run()
    assert app.error
    assert not app.get("download_button")
    values = [40, 30, 20, 10]
    for key, value in zip([widget.key for widget in controls], values):
        app.number_input(key=key).set_value(value)
    app.run()
    app.button(key="analyze").click().run()
    assert not app.exception
    assert app.success
    assert app.session_state["risk_result"][0]["weights"] == [0.4, 0.3, 0.2, 0.1]
