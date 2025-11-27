from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from the repo without installing.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinns.utils.plotting import plot_losses_from_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training losses from a metrics CSV.")
    parser.add_argument("--metrics", type=Path, required=True, help="Path to metrics.csv.")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the plot; shows interactively when omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_losses_from_csv(args.metrics, save_path=args.save)


if __name__ == "__main__":
    main()
