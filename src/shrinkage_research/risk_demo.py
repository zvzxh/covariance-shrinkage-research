"""Generated example data, available in both returns and adjusted-price schemas."""

from datetime import date, timedelta
import io
import csv

import numpy as np


def synthetic_csv(input_kind: str = "returns", *, rows: int = 125, seed: int = 20261006) -> bytes:
    if input_kind not in {"returns", "adjusted_prices"}:
        raise ValueError("unknown demo input kind")
    if type(rows) is not int or rows < 2:
        raise ValueError("rows must be an integer >= 2")
    rng = np.random.default_rng(seed)
    factors = rng.normal(0, 0.008, size=(rows, 2))
    loadings = np.array([[1.0, 0.2], [0.8, -0.3], [0.5, 0.7], [-0.2, 0.4]])
    returns = factors @ loadings.T + rng.normal(0, 0.004, size=(rows, 4))
    dates = []
    current = date(2024, 1, 1)
    while len(dates) < rows + 1:
        if current.weekday() < 5:
            dates.append(current.isoformat())
        current += timedelta(days=1)
    if input_kind == "returns":
        values, labels = returns, dates[1:]
    else:
        values = np.vstack((np.full(4, 100.0), 100 * np.cumprod(1 + returns, axis=0)))
        labels = dates
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["date", "ASSET_A", "ASSET_B", "ASSET_C", "ASSET_D"])
    writer.writerows([d, *[format(float(v), ".17g") for v in row]] for d, row in zip(labels, values))
    return stream.getvalue().encode("utf-8")
