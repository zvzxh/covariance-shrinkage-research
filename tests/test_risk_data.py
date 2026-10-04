import numpy as np
import pytest
from numpy.testing import assert_allclose

from shrinkage_research.risk_data import read_risk_input, selected_returns_csv
from shrinkage_research.risk_demo import synthetic_csv


def test_prices_convert_before_period_end_selection_and_record_baseline():
    prices = b"date,A,B\n2024-01-01,100,50\n2024-01-02,110,55\n2024-01-03,99,66\n2024-01-04,99,33\n"
    result = read_risk_input(prices, input_kind="adjusted_prices", start="2024-01-03")
    assert result.dates == ("2024-01-03", "2024-01-04")
    assert_allclose(result.returns, [[-0.1, 0.2], [0, -0.5]], atol=1e-15)
    assert result.audit["source_price_baseline_date"] == "2024-01-01"
    assert result.audit["first_selected_return_price_baseline_date"] == "2024-01-02"
    assert result.audit["selected_rows"] == 2


def test_returns_and_prices_share_date_window_contract_and_export_roundtrip():
    kwargs = dict(start="2024-02-01", end="2024-05-01", last_n=20)
    returns = read_risk_input(synthetic_csv("returns"), input_kind="returns", **kwargs)
    prices = read_risk_input(synthetic_csv("adjusted_prices"), input_kind="adjusted_prices", **kwargs)
    assert returns.dates == prices.dates
    assert_allclose(returns.returns, prices.returns, rtol=1e-10, atol=1e-14)
    again = read_risk_input(selected_returns_csv(prices), input_kind="returns")
    assert again.dates == prices.dates
    assert_allclose(again.returns, prices.returns, rtol=0, atol=0)


@pytest.mark.parametrize("content,kind", [
    (b"date,A,A\n2024-01-01,0,0\n2024-01-02,0,0\n", "returns"),
    (b"date,A,B\n2024-01-01,0,0\n2024-01-01,0,0\n", "returns"),
    (b"date,A,B\n2024-01-02,0,0\n2024-01-01,0,0\n", "returns"),
    (b"date,A,B\n2024-01-01,,0\n2024-01-02,0,0\n", "returns"),
    (b"date,A,B\n2024-01-01,nan,0\n2024-01-02,0,0\n", "returns"),
    (b"date,A,B\n2024-01-01,inf,0\n2024-01-02,0,0\n", "returns"),
    (b"date,A,B\n2024-01-01,-1,0\n2024-01-02,0,0\n", "returns"),
    (b"date,A,B\n2024-01-01,0,1\n2024-01-02,1,2\n2024-01-03,1,2\n", "adjusted_prices"),
    (b"date,A,B\n2024-02-30,0,0\n2024-03-01,0,0\n", "returns"),
    (b"date,A,B\n2024-1-1,0,0\n2024-01-02,0,0\n", "returns"),
    (b"date,A,B\n2024-01-01,0\n2024-01-02,0,0\n", "returns"),
])
def test_invalid_source_is_rejected_even_if_date_filter_would_exclude_bad_rows(content, kind):
    with pytest.raises(ValueError):
        read_risk_input(content, input_kind=kind, start="2024-01-02")


def test_reject_empty_or_too_short_selection_and_invalid_window():
    content = synthetic_csv()
    for kwargs in ({"start": "2099-01-01"}, {"last_n": 1}, {"start": "2025-01-01", "end": "2024-01-01"}):
        with pytest.raises(ValueError):
            read_risk_input(content, input_kind="returns", **kwargs)
