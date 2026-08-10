"""Run the complete reference pipeline from raw time series."""

from pathlib import Path

from auditable_alarm.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["run", str(Path(__file__).with_name("full_pipeline_input.json"))]))
