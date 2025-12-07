from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import lasio
import numpy as np
import pandas as pd

from multimin.inversion import MultiMineralInversion


LOG_COLUMNS = ("DT", "RHOB", "NPHI")


def load_well_dataframe(las_path: Path) -> pd.DataFrame:
    """Read the LAS file and return a clean dataframe with a DEPTH column."""
    well = lasio.read(str(las_path))
    data = well.df()
    index_name = data.index.name or "DEPTH"
    dataframe = data.reset_index().rename(columns={index_name: "DEPTH"})
    return dataframe


def filter_complete_logs(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where all required logs are present."""
    complete_rows = dataframe.dropna(subset=LOG_COLUMNS)
    if complete_rows.empty:
        raise ValueError(
            f"No rows contain all required logs: {', '.join(LOG_COLUMNS)}."
        )
    return complete_rows


def run_inversion_on_logs(
    dataframe: pd.DataFrame, method: str = "constrained"
) -> tuple[pd.DataFrame, float]:
    """
    Run the inversion on rows with complete logs and merge the result.

    Returns a tuple of (dataframe_with_results, mean_misfit).
    """
    inverter = MultiMineralInversion()
    complete_rows = filter_complete_logs(dataframe)

    def invert_row(row: pd.Series) -> pd.Series:
        result = inverter.invert(
            dt=float(row["DT"]),
            rhob=float(row["RHOB"]),
            nphi=float(row["NPHI"]),
            method=method,
        )
        solution = cast(dict[str, float], result["solution"])
        return pd.Series(
            {
                "Quartz": solution["Quartz"],
                "Calcite": solution["Calcite"],
                "Dolomite": solution["Dolomite"],
                "Porosity": solution["Porosity"],
                "TotalMisfit": result["total_misfit"],
            }
        )

    total_rows = len(complete_rows)
    print(f"Running inversion on {total_rows} rows using '{method}' method...")

    inversion_records: list[pd.Series] = []
    inversion_index: list[Any] = []

    progress_interval = max(1, total_rows // 10)
    for position, (row_index, row) in enumerate(complete_rows.iterrows(), start=1):
        inversion_records.append(invert_row(row))
        inversion_index.append(row_index)

        if position % progress_interval == 0 or position == total_rows:
            percent = position / total_rows * 100
            print(f"Processed {position}/{total_rows} rows ({percent:.0f}%)")

    inversion_results = pd.DataFrame(inversion_records, index=inversion_index)
    print("Inversion complete.")

    dataframe_with_results = dataframe.copy()
    for column in inversion_results.columns:
        dataframe_with_results[column] = np.nan
        dataframe_with_results.loc[
            inversion_results.index, column
        ] = inversion_results[column].to_numpy()

    mean_misfit = float(inversion_results["TotalMisfit"].mean())
    return dataframe_with_results, mean_misfit
