import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def generate_all_visualizations(results, y_test, predictions, plot_dir):

    os.makedirs(plot_dir, exist_ok=True)

    # =========================
    # MODEL COMPARISON BAR PLOT
    # =========================
    models = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in models]

    plt.figure()
    plt.bar(models, accuracies)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(plot_dir, "model_comparison.png"))
    plt.close()

    # =========================
    # CONFUSION MATRIX (last model)
    # =========================
    last_model = list(predictions.keys())[0]

    cm = confusion_matrix(y_test, predictions[last_model])

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"Confusion Matrix - {last_model}")
    plt.savefig(os.path.join(plot_dir, f"confusion_{last_model}.png"))
    plt.close()
