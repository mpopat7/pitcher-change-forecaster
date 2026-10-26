"""Plot validation accuracy and calibration, and export workload and error reviews."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pcf.model import ARTIFACTS, ROWS, row_loss

COLORS = {"A": "#777777", "B": "#2879a1", "C": "#186646"}


def validation_report():
    report = json.loads((ARTIFACTS / "report_validation_2024.json").read_text())
    fig, ax = plt.subplots(figsize=(7.5, 3.2), layout="constrained")
    for y, target in enumerate(["k", "bb"]):
        value = report["targets"][target]["C_minus_B"]
        estimate = value["per_bf"] * 1000
        lo, hi = np.array(value["ci95"]) * 1000
        ax.errorbar(estimate, y, xerr=[[estimate - lo], [hi - estimate]],
                    fmt="o", capsize=5, color=COLORS["C"])
    ax.axvline(0, color="#999999", linewidth=1, linestyle="--")
    ax.set(yticks=[0, 1], yticklabels=["Strikeouts", "Walks"], ylim=(1.5, -0.5),
           title="2024 validation: recent changes versus current pitches\n95% pitcher-bootstrap intervals",
           xlabel="Change in log loss per 1,000 BF (C − B)\nNegative favors recent-change features")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(ARTIFACTS / "changes_validation_2024.png", dpi=180)
    fig.savefig(ARTIFACTS / "changes_validation_2024.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for ax, (target, title) in zip(axes, [("k", "Strikeouts"), ("bb", "Walks")]):
        result = report["targets"][target]
        for y, key in enumerate(["B_minus_A", "C_minus_B", "C_minus_A"]):
            value = result[key]
            estimate = value["per_bf"] * 1000
            lo, hi = np.array(value["ci95"]) * 1000
            ax.errorbar(estimate, y, xerr=[[estimate - lo], [hi - estimate]],
                        fmt="o", capsize=4, color=COLORS[key[0]])
        ax.axvline(0, color="#999999", linewidth=1, linestyle="--")
        ax.set(yticks=range(3), yticklabels=["B − A", "C − B", "C − A"], title=title,
               xlabel="Change in log loss per 1,000 BF\nNegative favors the added features")
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("2024 validation: model comparisons with 95% pitcher-bootstrap intervals")
    fig.savefig(ARTIFACTS / "comparison_validation_2024.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for ax, (target, title) in zip(axes, [("k", "Strikeouts"), ("bb", "Walks")]):
        for model, values in report["targets"][target]["models"].items():
            bins = pd.DataFrame(values["calibration"])
            ax.plot(bins.pred * 100, bins.obs * 100, "o-", label=model, color=COLORS[model])
        limits = ax.get_xlim(), ax.get_ylim()
        lo, hi = min(limits[0][0], limits[1][0]), max(limits[0][1], limits[1][1])
        ax.plot([lo, hi], [lo, hi], "--", color="#999999", linewidth=1)
        ax.set(title=title, xlabel="Predicted rate (%)", ylabel="Observed rate (%)")
        ax.legend(title="Model")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("2024 validation: calibration by forecast decile")
    fig.savefig(ARTIFACTS / "calibration_validation_2024.png", dpi=180)
    plt.close(fig)

    preds = pd.read_parquet(ARTIFACTS / "predictions_validation_2024.parquet")
    context = pd.read_parquet(ROWS, columns=["pitcher", "forecast_date", "start_share"],
                              filters=[("season", "==", 2024)])
    preds = preds.merge(context, on=["pitcher", "forecast_date"], validate="one_to_one")
    preds["role"] = np.where(preds.start_share >= 0.5, "starter", "reliever")
    preds["repertoire"] = np.where(preds.repertoire_fallback, "fallback to A", "within usage ranges")
    preds["future_bf"] = pd.cut(preds.y_bf, [0, 50, 100, np.inf], labels=["1–50", "51–100", "101+"])
    segments, errors, pitcher_losses = [], [], []
    for target in ("k", "bb"):
        for model in COLORS:
            preds[f"loss_{model}"] = row_loss(preds[f"p_{target}_{model}"],
                                              preds[f"y_{target}"], preds.y_bf)
        by_pitcher = preds.groupby(["pitcher", "name"])[["y_bf", "loss_A", "loss_B", "loss_C"]].sum()
        by_pitcher["target"] = target
        by_pitcher["extra_loss_B_vs_A"] = by_pitcher.loss_B - by_pitcher.loss_A
        by_pitcher["extra_loss_C_vs_B"] = by_pitcher.loss_C - by_pitcher.loss_B
        pitcher_losses.append(by_pitcher.reset_index())
        for group in ("role", "future_bf", "repertoire"):
            for label, subset in preds.groupby(group, observed=True):
                record = {"target": target, "group": group, "segment": str(label),
                          "forecasts": len(subset), "bf": int(subset.y_bf.sum())}
                record.update({m: float(subset[f"loss_{m}"].sum() / subset.y_bf.sum())
                               for m in COLORS})
                record["C_minus_B"] = record["C"] - record["B"]
                segments.append(record)
        e = preds[["pitcher", "name", "forecast_date", "y_bf",
                   "repertoire_fallback", "repertoire_reasons"]].copy()
        e["target"] = target
        e["observed_count"] = preds[f"y_{target}"]
        e["pred_B"], e["pred_C"] = preds[f"p_{target}_B"], preds[f"p_{target}_C"]
        e["observed_rate"] = e.observed_count / e.y_bf
        e["error_C_pp"] = (e.pred_C - e.observed_rate) * 100
        e["extra_loss_C_vs_B"] = preds.loss_C - preds.loss_B
        # Require enough follow-up to avoid ranking tiny denominators as large misses.
        e = e[e.y_bf >= 50].sort_values("extra_loss_C_vs_B", ascending=False)
        errors.append(e.head(20))
    pd.DataFrame(segments).to_csv(ARTIFACTS / "segments_validation_2024.csv", index=False)
    pd.concat(errors, ignore_index=True).to_csv(ARTIFACTS / "errors_validation_2024.csv", index=False)
    pd.concat(pitcher_losses, ignore_index=True).to_csv(
        ARTIFACTS / "pitcher_losses_validation_2024.csv", index=False)
    print("Wrote validation comparison, calibration, segments, pitcher losses, and largest misses.")


if __name__ == "__main__":
    validation_report()
