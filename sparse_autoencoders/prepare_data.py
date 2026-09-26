"""Convert the constructor-stubs prompt set into the parquet file the SAE pipeline reads."""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "variance_partitioning" / "constructor_stubs.jsonl"
TARGET = HERE / "data" / "constructor_stubs.parquet"


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_json(SOURCE, lines=True)
    df.to_parquet(TARGET, index=False)
    print(f"{len(df)} prompts -> {TARGET}")


if __name__ == "__main__":
    main()
