from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from multimin.web_app import load_well_dataframe, run_inversion_on_logs


def main(method: str = "constrained") -> None:
    base_dir = Path(__file__).resolve().parent
    las_path = base_dir / "data" / "15_9-F-11A.LAS"
    dataframe = load_well_dataframe(las_path)
    dataframe_with_results, mean_misfit = run_inversion_on_logs(
        dataframe, method=method
    )

    processed_rows = int(dataframe_with_results["Quartz"].notna().sum())
    print(f"Applied multimineral inversion using the '{method}' method.")
    print(f"Rows processed: {processed_rows}")
    print(f"Average misfit: {mean_misfit:.4f}")

    preview_columns = [
        "DEPTH",
        "DT",
        "RHOB",
        "NPHI",
        "Quartz",
        "Calcite",
        "Dolomite",
        "Porosity",
    ]
    print("\nPreview of results:")
    print(dataframe_with_results.loc[:, preview_columns].head())

    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "inversion_results.parquet"
    fallback_results_path = output_dir / "inversion_results.csv"
    summary_path = output_dir / "inversion_summary.txt"

    try:
        dataframe_with_results.to_parquet(results_path, index=False)
        saved_path = results_path
    except ImportError:
        dataframe_with_results.to_csv(fallback_results_path, index=False)
        saved_path = fallback_results_path
        print(
            "pyarrow/fastparquet not available; wrote CSV instead of Parquet."
        )
    summary_path.write_text(
        "\n".join(
            [
                f"Method: {method}",
                f"Rows processed: {processed_rows}",
                f"Average misfit: {mean_misfit:.4f}",
                f"Results file: {saved_path}",
            ]
        )
    )
    print(f"\nSaved full results to {saved_path}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
