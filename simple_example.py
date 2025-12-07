import argparse
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

from multimin.inversion import MultiMineralInversion, MultiMineralInversion2


DEFAULT_POINT = {"dt": 75.0, "rhob": 2.4, "nphi": 0.3}
DEPTHS = np.array([100, 200, 300, 400, 500, 600], dtype=float)
DT_LOG = np.array([122.565, 166.415, 141.094, 127.176, 70.465, 185.890])
RHOB_LOG = np.array([1.537, 1.058, 2.495, 2.610, 1.240, 2.470])
NPHI_LOG = np.array([0.401, 0.086, 0.102, 0.178, 0.188, 0.139])


def plot_batch_results(
    depths: np.ndarray,
    solutions: np.ndarray,
    mineral_names: list[str],
) -> None:
    _, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=True)

    for index, mineral in enumerate(mineral_names):
        axes[index].plot(
            solutions[:, index] * 100,
            depths,
            "o-",
            linewidth=2,
            markersize=8,
        )
        axes[index].set_xlabel("Volume (%)")
        axes[index].set_title(mineral)
        axes[index].grid(True, alpha=0.3)
        axes[index].set_xlim(-5, 105)
        axes[index].axvline(x=0, color="red", linestyle="--", alpha=0.5)

    axes[0].set_ylabel("Depth")
    axes[0].invert_yaxis()
    plt.suptitle("Depth Track - Multimineral Inversion")
    plt.tight_layout()
    plt.show()


def run_batch_demo(
    depths: np.ndarray,
    dt_log: np.ndarray,
    rhob_log: np.ndarray,
    nphi_log: np.ndarray,
    mineral_names: list[str],
    solver: Callable[[float, float, float], dict[str, Any]],
    label: str,
) -> list[dict[str, Any]]:
    print("\n\n" + "=" * 60)
    print(f"BATCH PROCESSING EXAMPLE ({label})")
    print("=" * 60)

    results = [
        solver(dt_val, rhob_val, nphi_val)
        for dt_val, rhob_val, nphi_val in zip(dt_log, rhob_log, nphi_log)
    ]
    solutions = np.array([result["solution_array"] for result in results])

    plot_batch_results(depths, solutions, mineral_names)

    print(f"\nProcessed {len(depths)} depth points successfully!")
    return results


def run_classic_demo() -> None:
    inverter = MultiMineralInversion()

    print("\nMULTIMINERAL LOG INVERSION EXAMPLE (Classic)")
    print(
        f"DT={DEFAULT_POINT['dt']} µs/ft, "
        f"RHOB={DEFAULT_POINT['rhob']} g/cm³, "
        f"NPHI={DEFAULT_POINT['nphi']} v/v\n"
    )

    for method in ["standard", "constrained", "weighted", "scipy"]:
        print(f"\n{'=' * 60}")
        print(f"METHOD: {method.upper()}")
        print("=" * 60)
        results = inverter.invert(method=method, **DEFAULT_POINT)
        inverter.print_results(results)

        if method == "constrained":
            inverter.plot_results(results)

    run_batch_demo(
        DEPTHS,
        DT_LOG,
        RHOB_LOG,
        NPHI_LOG,
        inverter.mineral_names,
        solver=lambda dt, rhob, nphi: inverter.invert(
            dt=dt,
            rhob=rhob,
            nphi=nphi,
            method="weighted",
        ),
        label="Weighted Classic Method",
    )


def print_notebook_result(
    result: dict[str, Any], mineral_names: list[str]
) -> None:
    print("\nNotebook-Style Inversion Result")
    print("-" * 60)
    for mineral in mineral_names:
        value = result["solution"][mineral]
        print(f"{mineral:8s}: {value:.6f}")

    residual = result["residual"]
    print("\nResidual (A @ x - b):")
    print(f"  DT  : {residual[0]: .6f}")
    print(f"  RHOB: {residual[1]: .6f}")
    print(f"  NPHI: {residual[2]: .6f}")
    print(f"  Sum : {residual[3]: .6f}")
    print(f"\nObjective value: {result['objective_value']:.6e}")
    print(f"Optimizer success: {result['optimizer_success']}")
    print(f"Message: {result['optimizer_message']}")


def run_notebook_demo() -> None:
    inverter = MultiMineralInversion2()

    print("\nMULTIMINERAL LOG INVERSION EXAMPLE (Notebook Replica)")
    print(
        f"DT={DEFAULT_POINT['dt']} µs/ft, "
        f"RHOB={DEFAULT_POINT['rhob']} g/cm³, "
        f"NPHI={DEFAULT_POINT['nphi']} v/v\n"
    )

    result = inverter.invert(**DEFAULT_POINT)
    print_notebook_result(result, inverter.mineral_names)

    run_batch_demo(
        DEPTHS,
        DT_LOG,
        RHOB_LOG,
        NPHI_LOG,
        inverter.mineral_names,
        solver=lambda dt, rhob, nphi: inverter.invert(dt, rhob, nphi),
        label="Notebook Formulation",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run simple multimineral inversion demo."
    )
    parser.add_argument(
        "--variant",
        choices=["classic", "notebook"],
        default="classic",
        help=(
            "Select inversion backend: 'classic' (multi-method) "
            "or 'notebook' replica."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.variant == "classic":
        run_classic_demo()
    else:
        run_notebook_demo()


if __name__ == "__main__":
    main()
