"""
visualizations.py — Professional Visualization Suite
Project: Wildfire Risk Classification from Satellite and Weather Data
Dataset: Algerian Forest Fire Dataset

Generates publication-quality plots for model evaluation and analysis.
All plots saved to: outputs/plots/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ── Global Style Configuration ─────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

# Professional color palettes
MODEL_COLORS = {"SVM": "#6366F1", "Random Forest": "#10B981", "XGBoost": "#F59E0B"}
RISK_COLORS = {"Low": "#22C55E", "Medium": "#F59E0B", "High": "#EF4444"}
CMAP_CONFUSION = "YlOrRd"


def _save_fig(fig, output_dir: str, filename: str):
    """Save figure to disk and close."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. ACCURACY COMPARISON BAR CHART
# ═══════════════════════════════════════════════════════════════════════════
def plot_accuracy_comparison(metrics_dict: dict, output_dir: str):
    """
    Bar chart comparing accuracy across SVM, Random Forest, and XGBoost.

    Parameters
    ----------
    metrics_dict : dict
        {model_name: {"accuracy": float, ...}, ...}
    output_dir : str
        Directory to save the plot.
    """
    models = list(metrics_dict.keys())
    accuracies = [metrics_dict[m]["accuracy"] * 100 for m in models]
    colors = [MODEL_COLORS.get(m, "#8B5CF6") for m in models]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(models, accuracies, color=colors, width=0.55,
                  edgecolor="white", linewidth=1.5, zorder=3)

    # Value annotations on top of each bar
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{acc:.2f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=12, color="#1F2937")

    ax.set_title("Model Accuracy Comparison", pad=15)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, max(accuracies) + 10)
    ax.grid(axis="y", alpha=0.3)
    sns.despine(left=True)
    fig.tight_layout()
    _save_fig(fig, output_dir, "accuracy_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# 2. COMPREHENSIVE METRICS COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
def plot_metrics_comparison(metrics_dict: dict, output_dir: str):
    """
    Grouped bar chart comparing accuracy, precision, recall, F1 for all models.
    """
    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    models = list(metrics_dict.keys())

    x = np.arange(len(labels))
    width = 0.22
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, model in enumerate(models):
        vals = [metrics_dict[model].get(m, 0) * 100 for m in metric_names]
        bars = ax.bar(x + i * width, vals, width, label=model,
                      color=MODEL_COLORS.get(model, "#8B5CF6"),
                      edgecolor="white", linewidth=1.2, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_title("Model Performance — All Metrics", pad=15)
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.legend(frameon=True, framealpha=0.9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    sns.despine(left=True)
    fig.tight_layout()
    _save_fig(fig, output_dir, "metrics_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# 3. FEATURE IMPORTANCE (Random Forest / XGBoost)
# ═══════════════════════════════════════════════════════════════════════════
def plot_feature_importance(importances, feature_names, output_dir: str,
                            model_name: str = "Random Forest"):
    """
    Horizontal bar chart of feature importances.
    """
    idx = np.argsort(importances)
    sorted_imp = np.array(importances)[idx]
    sorted_names = np.array(feature_names)[idx]

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.viridis(np.linspace(0.25, 0.9, len(sorted_imp)))
    ax.barh(sorted_names, sorted_imp, color=colors, edgecolor="white", height=0.6)

    for i, v in enumerate(sorted_imp):
        ax.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=9, color="#374151")

    ax.set_title(f"Feature Importance — {model_name}", pad=15)
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.3)
    sns.despine(left=True)
    fig.tight_layout()
    safe = model_name.lower().replace(" ", "_")
    _save_fig(fig, output_dir, f"feature_importance_{safe}.png")


# ═══════════════════════════════════════════════════════════════════════════
# 4. WILDFIRE RISK DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════
def plot_risk_distribution(risk_series: pd.Series, output_dir: str):
    """
    Pie + bar chart showing distribution of Low / Medium / High risk levels.
    """
    counts = risk_series.value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
    colors = [RISK_COLORS[r] for r in counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Pie chart
    wedges, texts, autotexts = axes[0].pie(
        counts, labels=counts.index, colors=colors, autopct="%1.1f%%",
        startangle=140, pctdistance=0.8, explode=(0.03, 0.03, 0.06),
        textprops={"fontsize": 11, "fontweight": "bold"},
    )
    axes[0].set_title("Wildfire Risk Distribution", pad=15)

    # Bar chart
    axes[1].bar(counts.index, counts.values, color=colors,
                edgecolor="white", width=0.5, zorder=3)
    for i, (label, val) in enumerate(zip(counts.index, counts.values)):
        axes[1].text(i, val + 1, str(int(val)), ha="center",
                     fontweight="bold", fontsize=12)
    axes[1].set_title("Risk Level Counts", pad=15)
    axes[1].set_ylabel("Number of Samples")
    axes[1].grid(axis="y", alpha=0.3)
    sns.despine(ax=axes[1], left=True)

    fig.tight_layout()
    _save_fig(fig, output_dir, "risk_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════
# 5. CONFUSION MATRIX HEATMAPS
# ═══════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred, model_name: str, output_dir: str,
                          class_names=None):
    """
    Annotated confusion-matrix heatmap for a single model.
    """
    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=CMAP_CONFUSION,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=1.5, linecolor="white", cbar_kws={"shrink": 0.8},
                ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name}", pad=15)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    fig.tight_layout()
    safe = model_name.lower().replace(" ", "_")
    _save_fig(fig, output_dir, f"confusion_matrix_{safe}.png")


def plot_all_confusion_matrices(results: dict, output_dir: str, class_names=None):
    """
    Side-by-side confusion matrices for all models.

    Parameters
    ----------
    results : dict
        {model_name: {"y_true": array, "y_pred": array}, ...}
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(res["y_true"], res["y_pred"])
        labels = class_names or [f"C{i}" for i in range(cm.shape[0])]
        sns.heatmap(cm, annot=True, fmt="d", cmap=CMAP_CONFUSION,
                    xticklabels=labels, yticklabels=labels,
                    linewidths=1.5, linecolor="white", ax=ax)
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices — All Models", fontsize=15,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_dir, "confusion_matrices_all.png")


# ═══════════════════════════════════════════════════════════════════════════
# 6. PREDICTION PROBABILITY VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════
def plot_prediction_probabilities(probabilities, model_name: str,
                                  output_dir: str, class_names=None):
    """
    Histogram of prediction probabilities for each class.
    """
    n_classes = probabilities.shape[1]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 4))
    if n_classes == 1:
        axes = [axes]
    palette = sns.color_palette("husl", n_classes)

    for i, (ax, cname) in enumerate(zip(axes, class_names)):
        ax.hist(probabilities[:, i], bins=25, color=palette[i],
                edgecolor="white", alpha=0.85)
        ax.set_title(f"{cname}", fontweight="bold")
        ax.set_xlabel("Probability")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Prediction Probabilities — {model_name}",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    safe = model_name.lower().replace(" ", "_")
    _save_fig(fig, output_dir, f"prediction_probs_{safe}.png")


# ═══════════════════════════════════════════════════════════════════════════
# 7. CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════
def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str):
    """
    Full-feature correlation matrix heatmap.
    """
    corr = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Feature Correlation Matrix", pad=15)
    fig.tight_layout()
    _save_fig(fig, output_dir, "correlation_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
# 8. CLASS DISTRIBUTION (Binary Target)
# ═══════════════════════════════════════════════════════════════════════════
def plot_class_distribution(y: pd.Series, output_dir: str):
    """
    Bar chart of the binary fire / not-fire class distribution.
    """
    counts = y.value_counts().sort_index()
    labels = ["Low Risk (0)", "High Risk (1)"]
    colors = ["#22C55E", "#EF4444"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels[:len(counts)], counts.values, color=colors[:len(counts)],
                  edgecolor="white", width=0.5, zorder=3)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha="center", fontweight="bold", fontsize=13)
    ax.set_title("Target Class Distribution", pad=15)
    ax.set_ylabel("Number of Samples")
    ax.grid(axis="y", alpha=0.3)
    sns.despine(left=True)
    fig.tight_layout()
    _save_fig(fig, output_dir, "class_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════
# 9. RADAR / SPIDER CHART — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
def plot_model_radar(metrics_dict: dict, output_dir: str):
    """
    Radar chart comparing models across multiple metrics.
    """
    categories = ["Accuracy", "Precision", "Recall", "F1-Score"]
    metric_keys = ["accuracy", "precision", "recall", "f1_score"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model, metrics in metrics_dict.items():
        values = [metrics.get(k, 0) * 100 for k in metric_keys]
        values += values[:1]
        color = MODEL_COLORS.get(model, "#8B5CF6")
        ax.plot(angles, values, "o-", linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_title("Model Comparison — Radar Chart", y=1.08,
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=True)
    fig.tight_layout()
    _save_fig(fig, output_dir, "model_radar_chart.png")


# ═══════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION — Generate All Visualizations
# ═══════════════════════════════════════════════════════════════════════════
def generate_all_visualizations(
    df: pd.DataFrame,
    metrics_dict: dict,
    results: dict,
    feature_importances: dict,
    feature_names: list,
    output_dir: str = "outputs/plots",
):
    """
    Generate and save the complete visualization suite.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset (used for correlation, class distribution, risk).
    metrics_dict : dict
        {model_name: {metric: value, ...}, ...}
    results : dict
        {model_name: {"y_true": array, "y_pred": array,
                       "probabilities": array or None}, ...}
    feature_importances : dict
        {model_name: importance_array, ...}
    feature_names : list
        List of feature column names.
    output_dir : str
        Directory to save all plot files.
    """
    print("\n[VIZ] Generating visualizations...\n")

    # 1. Accuracy comparison
    plot_accuracy_comparison(metrics_dict, output_dir)

    # 2. All-metrics comparison
    plot_metrics_comparison(metrics_dict, output_dir)

    # 3. Feature importance for each model that provides it
    for model_name, imp in feature_importances.items():
        plot_feature_importance(imp, feature_names, output_dir, model_name)

    # 4. Risk distribution (derived from FWI)
    from src.utils import assign_risk_levels
    risk = assign_risk_levels(df)
    plot_risk_distribution(risk, output_dir)

    # 5. Individual confusion matrices
    class_names = ["Low Risk", "High Risk"]
    for name, res in results.items():
        plot_confusion_matrix(res["y_true"], res["y_pred"], name,
                              output_dir, class_names)

    # 6. Combined confusion matrices
    plot_all_confusion_matrices(results, output_dir, class_names)

    # 7. Prediction probabilities
    for name, res in results.items():
        probs = res.get("probabilities")
        if probs is not None:
            plot_prediction_probabilities(probs, name, output_dir, class_names)

    # 8. Correlation heatmap
    plot_correlation_heatmap(df, output_dir)

    # 9. Class distribution
    plot_class_distribution(df["Classes"], output_dir)

    # 10. Radar chart
    plot_model_radar(metrics_dict, output_dir)

    print(f"\n[OK] All visualizations saved to: {output_dir}/")
