from pathlib import Path
r = Path(__file__).resolve().parent
for name in ["data/TEP", "data/HAI", "models", "results/llm_raw", "logs", "tmp"] + [f"experiments/{v}/{s}" for v in ["v2", "v3", "v4", "v5_review", "v6_review"] for s in ["data", "models", "results"]]:
    (r / name).mkdir(parents=True, exist_ok=True)
