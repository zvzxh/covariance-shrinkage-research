"""V2 strict shared date/asset contract and auditable price conversion."""

import csv
from dataclasses import dataclass
from datetime import date
import hashlib
import io
import re

import numpy as np


@dataclass(frozen=True)
class RiskInput:
    dates: tuple[str, ...]
    assets: tuple[str, ...]
    returns: np.ndarray
    audit: dict


def iso_date(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("dates must be ISO YYYY-MM-DD")
    try:
        date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"invalid calendar date: {value}") from error
    return value


def read_risk_input(content: bytes, *, input_kind: str, start: str | None = None,
                    end: str | None = None, last_n: int | None = None) -> RiskInput:
    """Validate all source rows, convert prices, then select return-end dates.

    Prices must be declared adjusted; their economic adjustment cannot be
    established from values alone. No sorting, filling or imputation occurs.
    """
    if input_kind not in {"returns", "adjusted_prices"}:
        raise ValueError("input_kind must be returns or adjusted_prices")
    try:
        rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
    except UnicodeDecodeError as error:
        raise ValueError("CSV must be UTF-8 encoded") from error
    if not rows or len(rows[0]) < 3 or rows[0][0] != "date":
        raise ValueError("CSV must start with date and at least two asset columns")
    header = rows[0]
    if len(set(header)) != len(header) or any(not h or h.strip() != h for h in header):
        raise ValueError("column names must be unique, nonempty and without surrounding whitespace")
    dates, values = [], []
    for line, row in enumerate(rows[1:], 2):
        if len(row) != len(header):
            raise ValueError(f"row {line}: column count differs from header")
        period_end = iso_date(row[0])
        if dates and period_end <= dates[-1]:
            raise ValueError(f"row {line}: dates must be unique and strictly increasing")
        try:
            vector = np.asarray([float(value) for value in row[1:]], dtype=float)
        except ValueError as error:
            raise ValueError(f"row {line}: missing or nonnumeric value") from error
        if not np.isfinite(vector).all():
            raise ValueError(f"row {line}: all values must be finite; missing values are rejected")
        if (vector <= (-1 if input_kind == "returns" else 0)).any():
            boundary = "greater than -1" if input_kind == "returns" else "strictly positive"
            raise ValueError(f"row {line}: {input_kind} values must be {boundary}")
        dates.append(period_end)
        values.append(vector)
    if len(values) < (3 if input_kind == "adjusted_prices" else 2):
        raise ValueError("at least two return observations are required (three price rows)")
    raw = np.vstack(values)
    source_dates = list(dates)
    if input_kind == "adjusted_prices":
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            raw = raw[1:] / raw[:-1] - 1.0
        if not np.isfinite(raw).all() or (raw <= -1).any():
            raise ValueError("price conversion produced unrepresentable simple returns")
        dates = dates[1:]
    if start is not None:
        start = iso_date(start)
    if end is not None:
        end = iso_date(end)
    if start is not None and end is not None and start > end:
        raise ValueError("start must not be after end")
    if last_n is not None and (type(last_n) is not int or last_n < 2):
        raise ValueError("last_n must be null or an integer >= 2")
    selected = [i for i, d in enumerate(dates) if (start is None or d >= start) and (end is None or d <= end)]
    date_filtered_rows = len(selected)
    if last_n is not None:
        selected = selected[-last_n:]
    if len(selected) < 2:
        raise ValueError("selection contains fewer than two return observations")
    selected_dates = tuple(dates[i] for i in selected)
    audit = {
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "input_kind": input_kind, "source_rows": len(values), "asset_count": len(header) - 1,
        "source_first_date": source_dates[0], "source_last_date": source_dates[-1],
        "conversion": "P_t/P_(t-1)-1; no fill" if input_kind == "adjusted_prices" else "none; decimal simple returns",
        "source_price_baseline_date": source_dates[0] if input_kind == "adjusted_prices" else None,
        "converted_return_rows": len(dates), "selection_order": "validate entire source; convert; return-end date filter; trailing last_n",
        "requested_start": start, "requested_end": end, "requested_last_n": last_n,
        "date_filtered_rows": date_filtered_rows, "selected_rows": len(selected),
        "selected_first_return_end": selected_dates[0], "selected_last_return_end": selected_dates[-1],
        "first_selected_return_price_baseline_date": source_dates[selected[0]] if input_kind == "adjusted_prices" else None,
        "price_adjustment_verified": False if input_kind == "adjusted_prices" else None,
    }
    return RiskInput(selected_dates, tuple(header[1:]), raw[selected].copy(), audit)


def selected_returns_csv(data: RiskInput) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["date", *data.assets])
    for d, row in zip(data.dates, data.returns):
        writer.writerow([d, *[format(float(value), ".17g") for value in row]])
    return buffer.getvalue().encode("utf-8")
