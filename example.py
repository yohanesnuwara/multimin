import matplotlib.pyplot as plt
import numpy as np

from multimin.inversion import MultiMineralInversion


def main() -> None:
    inverter = MultiMineralInversion()

    print("\nMULTIMINERAL LOG INVERSION EXAMPLE")
    print("DT=75 µs/ft, RHOB=2.4 g/cm³, NPHI=0.3 v/v\n")

    methods = ["standard", "constrained", "weighted", "scipy"]

    for method in methods:
        print(f"\n{'=' * 60}")
        print(f"METHOD: {method.upper()}")
        print("=" * 60)

        results = inverter.invert(dt=75, rhob=2.4, nphi=0.3, method=method)
        inverter.print_results(results)

        if method == "constrained":
            inverter.plot_results(results)

    print("\n\n" + "=" * 60)
    print("BATCH PROCESSING EXAMPLE")
    print("=" * 60)

    depths = np.array([100, 200, 300, 400, 500, 600], dtype=float)
    dt_log = np.array([122.565, 166.415, 141.094, 127.176, 70.465, 185.89])
    rhob_log = np.array([1.537, 1.058, 2.495, 2.61, 1.24, 2.47])
    nphi_log = np.array([0.401, 0.086, 0.102, 0.178, 0.188, 0.139])

    all_results = [
        inverter.invert(dt=dt_val, rhob=rhob_val, nphi=nphi_val, method="weighted")
        for dt_val, rhob_val, nphi_val in zip(dt_log, rhob_log, nphi_log)
    ]
    solutions = np.array([result["solution_array"] for result in all_results])

    _, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=True)

    for index, mineral in enumerate(inverter.mineral_names):
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

    print(f"\nProcessed {len(depths)} depth points successfully!")


if __name__ == "__main__":
    main()
