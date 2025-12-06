from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.optimize import lsq_linear


FloatArray = NDArray[np.float64]


class MultiMineralInversion:
    """
    Multimineral inversion for petrophysical analysis
    Solves for Quartz, Calcite, Dolomite, and Porosity from DT, RHOB, NPHI logs
    """

    def __init__(self) -> None:
        # Endpoint properties: [Qtz, Cal, Dol, Fluid]
        self.endpoints = {
            "DT": np.array([55.5, 47.5, 43.5, 189.0]),
            "RHOB": np.array([2.65, 2.7, 2.8, 1.05]),
            "NPHI": np.array([-0.04, 0.0, 0.05, 1.0]),
        }

        # Normalization factors for better conditioning
        self.norm_factors = {"DT": 100.0, "RHOB": 1.0, "NPHI": 0.5}

        self.mineral_names = ["Quartz", "Calcite", "Dolomite", "Porosity"]

    def build_kernel_matrix(self, normalized: bool = False) -> FloatArray:
        """
        Build the kernel matrix G from endpoint properties.

        Returns:
            G: (3, 4) or (4, 4) matrix depending on method.
        """
        G = np.array(
            [
                self.endpoints["DT"],
                self.endpoints["RHOB"],
                self.endpoints["NPHI"],
            ]
        )

        if normalized:
            for i, log_type in enumerate(["DT", "RHOB", "NPHI"]):
                G[i, :] = G[i, :] / self.norm_factors[log_type]

        return G

    def standard_least_squares(self, dt: float, rhob: float, nphi: float) -> FloatArray:
        """
        Standard least squares with closure constraint.

        May produce negative values.
        Solves: [G; 1 1 1 1] * x = [d; 1].
        """
        G = self.build_kernel_matrix()
        d = np.array([dt, rhob, nphi])

        G_aug = np.vstack([G, np.ones(4)])  # Add closure constraint as 4th equation
        d_aug = np.append(d, 1.0)

        solution, *_ = np.linalg.lstsq(G_aug, d_aug, rcond=None)
        return solution

    def constrained_least_squares(
        self, dt: float, rhob: float, nphi: float, method: str = "scipy"
    ) -> FloatArray:
        """
        Constrained least squares with bounds and closure constraint.

        Minimizes: ||Gx - d||^2.
        Subject to: sum(x) = 1, x >= 0, x <= 1.

        Parameters:
            method: "scipy" or "custom".
        """
        G = self.build_kernel_matrix(normalized=True)
        d = np.array(
            [
                dt / self.norm_factors["DT"],
                rhob / self.norm_factors["RHOB"],
                nphi / self.norm_factors["NPHI"],
            ]
        )

        if method == "scipy":
            result = lsq_linear(
                G,
                d,
                bounds=(0, 1),
                method="bvls",  # Bounded-Variable Least Squares
            )
            solution = result.x
            solution = self._project_simplex(solution)  # Project onto simplex
        else:
            solution = self._projected_gradient_descent(G, d)

        return solution

    def _project_simplex(self, vector: FloatArray) -> FloatArray:
        """
        Project vector onto the probability simplex:
        {x: x >= 0, sum(x) = 1}

        Using Duchi et al. 2008 algorithm.
        """
        n = len(vector)
        sorted_vals = np.sort(vector)[::-1]
        cumulative_sum = np.cumsum(sorted_vals)

        rho = 0
        for j in range(n):
            if sorted_vals[j] + (1.0 - cumulative_sum[j]) / (j + 1) > 0:
                rho = j + 1

        theta = (cumulative_sum[rho - 1] - 1.0) / rho
        return np.maximum(vector - theta, 0)

    def _projected_gradient_descent(
        self,
        kernel: FloatArray,
        data: FloatArray,
        max_iter: int = 1000,
        tol: float = 1e-10,
    ) -> FloatArray:
        """
        Projected gradient descent with Nesterov acceleration.
        """
        n = kernel.shape[1]
        x_curr = np.ones(n) / n  # Initialize with equal distribution
        x_prev = x_curr.copy()
        t_curr = 1.0

        for _ in range(max_iter):
            residual = kernel @ x_curr - data
            gradient = kernel.T @ residual

            alpha = 0.1  # Adaptive step size with backtracking
            for _ in range(20):
                candidate = x_curr - alpha * gradient
                x_new = self._project_simplex(candidate)

                obj_old = np.sum((kernel @ x_curr - data) ** 2)
                obj_new = np.sum((kernel @ x_new - data) ** 2)

                if obj_new <= obj_old:
                    break
                alpha *= 0.5

            t_next = (1 + np.sqrt(1 + 4 * t_curr * t_curr)) / 2
            momentum = (t_curr - 1) / t_next

            x_next = x_new + momentum * (x_new - x_prev)
            x_next = self._project_simplex(x_next)

            diff = np.linalg.norm(x_next - x_curr)
            x_prev = x_curr
            x_curr = x_next
            t_curr = t_next

            if diff < tol:
                break

        return x_curr

    def weighted_constrained_ls(
        self, dt: float, rhob: float, nphi: float, weights: FloatArray | None = None
    ) -> FloatArray:
        """
        Weighted constrained least squares.

        Higher weights on more reliable logs.
        Default weights: [1, 2, 1] for [DT, RHOB, NPHI].
        RHOB is typically most reliable.
        """
        if weights is None:
            weights = np.array([1.0, 2.0, 1.0])

        kernel = self.build_kernel_matrix(normalized=True)
        data = np.array(
            [
                dt / self.norm_factors["DT"],
                rhob / self.norm_factors["RHOB"],
                nphi / self.norm_factors["NPHI"],
            ]
        )

        weight_matrix = np.diag(np.sqrt(weights))
        kernel_weighted = weight_matrix @ kernel
        data_weighted = weight_matrix @ data

        return self._projected_gradient_descent(kernel_weighted, data_weighted)

    def reconstruct_logs(self, solution: FloatArray) -> dict[str, float]:
        """
        Reconstruct log values from solution.
        """
        reconstruction = {}
        for log_type in ["DT", "RHOB", "NPHI"]:
            reconstruction[log_type] = np.dot(self.endpoints[log_type], solution)

        return reconstruction

    def compute_misfit(
        self, solution: FloatArray, dt: float, rhob: float, nphi: float
    ) -> tuple[dict[str, float], float]:
        """
        Compute reconstruction misfit.
        """
        recon = self.reconstruct_logs(solution)

        misfits = {
            "DT": abs(recon["DT"] - dt),
            "RHOB": abs(recon["RHOB"] - rhob),
            "NPHI": abs(recon["NPHI"] - nphi),
        }

        total_misfit = np.sqrt(
            (misfits["DT"] / 50) ** 2
            + (misfits["RHOB"] / 0.5) ** 2
            + (misfits["NPHI"] / 0.2) ** 2
        )

        return misfits, total_misfit

    def invert(
        self,
        dt: float,
        rhob: float,
        nphi: float,
        method: str = "constrained",
    ) -> dict[str, Any]:
        """
        Main inversion function.

        Parameters:
            dt: Travel time (us/ft).
            rhob: Bulk density (g/cm3).
            nphi: Neutron porosity (v/v).
            method: "standard", "constrained", "weighted", "scipy".

        Returns:
            results: dict with solution, reconstruction, and diagnostics.
        """
        if method == "standard":
            solution = self.standard_least_squares(dt, rhob, nphi)
            method_name = "Standard Least Squares"
        elif method == "constrained":
            solution = self.constrained_least_squares(dt, rhob, nphi, method="custom")
            method_name = "Constrained LS (Custom)"
        elif method == "scipy":
            solution = self.constrained_least_squares(dt, rhob, nphi, method="scipy")
            method_name = "Constrained LS (SciPy)"
        elif method == "weighted":
            solution = self.weighted_constrained_ls(dt, rhob, nphi)
            method_name = "Weighted Constrained LS"
        else:
            raise ValueError(f"Unknown method: {method}")

        reconstruction = self.reconstruct_logs(solution)
        misfits, total_misfit = self.compute_misfit(solution, dt, rhob, nphi)

        return {
            "solution": dict(zip(self.mineral_names, solution)),
            "solution_array": solution,
            "reconstruction": reconstruction,
            "misfits": misfits,
            "total_misfit": total_misfit,
            "method": method_name,
            "sum": float(np.sum(solution)),
            "has_negative": bool(np.any(solution < -0.001)),
            "inputs": {"DT": dt, "RHOB": rhob, "NPHI": nphi},
        }

    def plot_results(self, results: dict[str, Any]) -> Figure:
        """
        Plot inversion results.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax1 = axes[0]
        minerals = self.mineral_names
        values = [results["solution"][mineral] * 100 for mineral in minerals]
        colors = ["#EAB308", "#3B82F6", "#9333EA", "#06B6D4"]

        bars = ax1.barh(minerals, values, color=colors, alpha=0.8, edgecolor="black")
        ax1.set_xlabel("Volume Fraction (%)")
        ax1.set_title(f'Mineral Composition\n{results["method"]}')
        ax1.set_xlim(0, 100)
        ax1.axvline(x=0, color="black", linewidth=0.5)
        ax1.grid(axis="x", alpha=0.3)

        for index, (bar, value) in enumerate(zip(bars, values)):
            if value < 0:
                ax1.text(
                    -2,
                    index,
                    f"{value:.1f}%",
                    ha="right",
                    va="center",
                    color="red",
                    fontweight="bold",
                )
            else:
                ax1.text(value + 2, index, f"{value:.1f}%", ha="left", va="center")

        ax2 = axes[1]
        log_types = ["DT", "RHOB", "NPHI"]
        x_pos = np.arange(len(log_types))
        width = 0.35

        input_vals = [results["inputs"][log_type] for log_type in log_types]
        recon_vals = [results["reconstruction"][log_type] for log_type in log_types]

        ax2.bar(
            x_pos - width / 2,
            input_vals,
            width,
            label="Input",
            alpha=0.8,
            color="skyblue",
        )
        ax2.bar(
            x_pos + width / 2,
            recon_vals,
            width,
            label="Reconstructed",
            alpha=0.8,
            color="orange",
        )

        ax2.set_ylabel("Log Value")
        ax2.set_title("Log Reconstruction Quality")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(log_types)
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)

        for index, log_type in enumerate(log_types):
            misfit_val = results["misfits"][log_type]
            ax2.text(
                x_pos[index],
                max(input_vals[index], recon_vals[index]) * 1.05,
                f"Δ={misfit_val:.3f}",
                ha="center",
                fontsize=8,
            )

        plt.tight_layout()
        plt.show()
        return fig

    def print_results(self, results: dict[str, Any]) -> None:
        """
        Print inversion results in a formatted way.
        """
        print("=" * 60)
        print("MULTIMINERAL INVERSION RESULTS")
        print(f"Method: {results['method']}")
        print("=" * 60)

        print("\n📊 VOLUME FRACTIONS:")
        print("-" * 60)
        for mineral, value in results["solution"].items():
            bar_length = int(max(0, value * 50))
            bar = "█" * bar_length
            warning = " ⚠️ NEGATIVE!" if value < 0 else ""
            print(f"{mineral:12s}: {value * 100:6.2f}% {bar}{warning}")

        print(f"\nSum: {results['sum'] * 100:.2f}%")

        print("\n📈 LOG RECONSTRUCTION:")
        print("-" * 60)
        for log_type in ["DT", "RHOB", "NPHI"]:
            input_val = results["inputs"][log_type]
            recon_val = results["reconstruction"][log_type]
            misfit = results["misfits"][log_type]

            match = "✓" if misfit < 0.1 else "✗"
            print(
                f"{log_type:6s}: Input={input_val:7.3f}  "
                f"Recon={recon_val:7.3f}  Δ={misfit:6.3f} {match}"
            )

        print(f"\nTotal Misfit: {results['total_misfit']:.4f}")

        if results["has_negative"]:
            print("\n⚠️  WARNING: Solution contains negative values!")
            print("   This indicates the mineral model may be inappropriate")
            print("   or the input logs are inconsistent.")

        print("=" * 60)
