"""Run the data audit on every project file and save a summary.

Usage (from the project root, with .venv active):
    python main.py
"""
import json
from pathlib import Path

from ahu_oa import loading

DATA = Path("data")
FILES = [
    "AHU_annual.csv",
    "oa_bias_2_annual.csv",
    "oa_bias_4_annual.csv",
    "oa_bias_-2_annual.csv",
    "oa_bias_-4_annual.csv",
]


def main():
    results = {}
    for name in FILES:
        path = DATA / name
        if not path.exists():
            print(f"SKIP {name}: not found in data/")
            continue
        df = loading.load_ahu_csv(str(path), time_col="Datetime")
        report = loading.audit_ahu(df)
        report["n_occupied_rows"] = int(loading.occupied_mask(df).sum())
        results[name] = report
        print(name, report)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "audit_summary.json").write_text(
        json.dumps(results, indent=2, default=str)
    )


if __name__ == "__main__":
    main()