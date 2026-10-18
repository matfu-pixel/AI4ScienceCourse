"""Boyle's law analysis: p*V = const under isothermal conditions.

Reads data/measurements.csv, computes p*V, fits p = C/V, flags outliers,
writes results/ (CSV, JSON summary, PNG figures).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "measurements.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["pv"] = df["pressure_kpa"] * df["volume_l"]
    return df


def detect_outliers(df: pd.DataFrame) -> pd.Series:
    """Flag rows whose p*V deviates from the median by > 3*MAD."""
    med = df["pv"].median()
    mad = (df["pv"] - med).abs().median()
    return (df["pv"] - med).abs() > 3 * (1.4826 * mad + 1e-9)


def fit_boyle(df: pd.DataFrame) -> tuple[float, float]:
    """Fit p = C/V by linear regression of p vs 1/V (no intercept)."""
    x = 1.0 / df["volume_l"].to_numpy()
    y = df["pressure_kpa"].to_numpy()
    C = np.sum(x * y) / np.sum(x * x)
    resid = y - C * x
    r2 = 1.0 - np.sum(resid**2) / np.sum((y - y.mean())**2)
    return float(C), float(r2)


def main() -> None:
    df = load()
    outlier = detect_outliers(df)
    clean = df[~outlier]

    C, r2 = fit_boyle(clean)

    pv_clean = clean["pv"]
    stats = {
        "n_total": int(len(df)),
        "n_outliers": int(outlier.sum()),
        "outlier_rows": df.loc[outlier, "volume_l"].tolist(),
        "pv_mean_kpa_l": float(pv_clean.mean()),
        "pv_std_kpa_l": float(pv_clean.std(ddof=1)),
        "pv_median_kpa_l": float(pv_clean.median()),
        "pv_min_kpa_l": float(pv_clean.min()),
        "pv_max_kpa_l": float(pv_clean.max()),
        "pv_rel_std_pct": float(pv_clean.std(ddof=1) / pv_clean.mean() * 100),
        "fit_C_kpa_l": C,
        "fit_r2": r2,
        "temperature_k": float(df["temperature_k"].iloc[0]),
    }

    df.to_csv(OUT / "pv_table.csv", index=False)
    with open(OUT / "summary.json", "w") as f:
        json.dump(stats, f, indent=2)

    # --- Figure 1: p vs V with Boyle fit ---
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    v = np.linspace(df["volume_l"].min(), df["volume_l"].max(), 200)
    ax.plot(v, C / v, "-", color="C0", label=f"fit $p=C/V$, $C={C:.1f}$")
    ax.scatter(clean["volume_l"], clean["pressure_kpa"], s=18, color="C1",
               label="data")
    ax.scatter(df.loc[outlier, "volume_l"], df.loc[outlier, "pressure_kpa"],
               s=40, facecolors="none", edgecolors="C3", label="outlier")
    ax.set_xlabel("Volume $V$ (L)")
    ax.set_ylabel("Pressure $p$ (kPa)")
    ax.legend(fontsize=7)
    ax.set_title("Isotherm $T=293.15$ K")
    fig.tight_layout()
    fig.savefig(OUT / "fig_pv.pdf")
    plt.close(fig)

    # --- Figure 2: p*V product vs V ---
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.axhline(pv_clean.mean(), color="C0", ls="--",
               label=f"mean $pV={pv_clean.mean():.1f}$")
    ax.scatter(clean["volume_l"], clean["pv"], s=18, color="C1", label="data")
    ax.scatter(df.loc[outlier, "volume_l"], df.loc[outlier, "pv"], s=40,
               facecolors="none", edgecolors="C3", label="outlier")
    ax.set_xlabel("Volume $V$ (L)")
    ax.set_ylabel("$p\\cdot V$ (kPa$\\cdot$L)")
    ax.legend(fontsize=7)
    ax.set_title("Boyle product $pV$")
    fig.tight_layout()
    fig.savefig(OUT / "fig_pv_product.pdf")
    plt.close(fig)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()