import sys
import importlib
import matplotlib.pyplot as plt
import numpy as np


def run_experiment(module_name, fn_name="c51", extra_args=None):

    if extra_args is None:
        extra_args = []

    # Backup argv (tyro reads from sys.argv)
    old_argv = sys.argv.copy()
    sys.argv = [module_name] + extra_args

    try:
        module = importlib.import_module(module_name)
        fn = getattr(module, fn_name)
        returns = fn()
    finally:
        sys.argv = old_argv

    return np.asarray(returns)


def smooth(x, window=10):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


if __name__ == "__main__":

    BASE_MODULE = "baselines.c51"         
    HEUR_MODULE = "h_c51_product"   

    SMOOTH_WINDOW = 10

    print("Running C51 baseline...")
    returns_base = run_experiment(BASE_MODULE, fn_name="c51")

    print("Running Heuristic Guided C51...")
    returns_shift = run_experiment(HEUR_MODULE, fn_name="h_c51")

    plt.figure(figsize=(10, 6))

    plt.plot(
        smooth(returns_base, SMOOTH_WINDOW),
        label="C51 baseline",
        linewidth=2,
    )
    plt.plot(
        smooth(returns_shift, SMOOTH_WINDOW),
        label="C51 heur",
        linewidth=2,
    )

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("C51 vs H-C51 — Episodic Returns")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("c51_vs_hc51_returns.png")
