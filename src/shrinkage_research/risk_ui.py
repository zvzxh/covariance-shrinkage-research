"""Launch the installed optional UI, bound to localhost with telemetry disabled."""

import os
from pathlib import Path
import sys


def main() -> None:
    try:
        import streamlit  # noqa: F401
    except ImportError as error:
        raise SystemExit("Install the optional UI: python -m pip install 'covariance-shrinkage-research[app]' or pip install -e '.[app]'.") from error
    arguments = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).with_name("risk_app.py")),
                 "--server.address", "127.0.0.1", "--browser.gatherUsageStats", "false", *sys.argv[1:]]
    os.execv(sys.executable, arguments)


if __name__ == "__main__":
    main()
