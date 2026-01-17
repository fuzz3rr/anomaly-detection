import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd


def plot_confusion_matrices(results, save_path):
    n_models = len(results)
    _, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 12))

    if n_models == 1:
        axes = axes.reshape(-1, 1)

    colors = ["Blues", "Greens", "Oranges"]

    for idx, result in enumerate(results):
        # Get binary labels from result or use default
        binary_labels = result.get("binary_labels", ["BENIGN", "ATTACK"])

        cm_bin = confusion_matrix(result["y_test_bin"], result["y_pred_bin"])
        sns.heatmap(
            cm_bin,
            annot=True,
            fmt="d",
            cmap=colors[idx % len(colors)],
            ax=axes[0, idx],
            xticklabels=binary_labels,
            yticklabels=binary_labels,
        )
        axes[0, idx].set_title(
            f"{result['model_name']} - Binary Classification",
            fontsize=14,
            fontweight="bold",
        )
        axes[0, idx].set_ylabel("True class")
        axes[0, idx].set_xlabel("Predicted class")

        cm_multi = confusion_matrix(result["y_test_multi"], result["y_pred_multi"])
        sns.heatmap(
            cm_multi,
            annot=True,
            fmt="d",
            cmap=colors[idx % len(colors)],
            ax=axes[1, idx],
            xticklabels=result["class_names"],
            yticklabels=result["class_names"],
        )
        axes[1, idx].set_title(
            f"{result['model_name']} - Multiclass Classification",
            fontsize=14,
            fontweight="bold",
        )
        axes[1, idx].set_ylabel("True class")
        axes[1, idx].set_xlabel("Predicted class")
        axes[1, idx].tick_params(axis="x", rotation=45)
        axes[1, idx].tick_params(axis="y", rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path.split('/')[-1]}")


def plot_feature_importance(models_data, feature_names, save_path):
    n_models = len(models_data)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 6))

    if n_models == 1:
        axes = [axes]

    for idx, model_data in enumerate(models_data):
        if model_data["importances"] is None:
            axes[idx].text(
                0.5,
                0.5,
                "Feature importance\nnot available",
                ha="center",
                va="center",
                fontsize=14,
            )
            axes[idx].set_title(f"{model_data['name']}", fontsize=14, fontweight="bold")
            continue

        importance_df = (
            pd.DataFrame(
                {"feature": feature_names, "importance": model_data["importances"]}
            )
            .sort_values("importance", ascending=False)
            .head(15)
        )

        axes[idx].barh(range(len(importance_df)), importance_df["importance"])
        axes[idx].set_yticks(range(len(importance_df)))
        axes[idx].set_yticklabels(importance_df["feature"])
        axes[idx].set_xlabel("Feature Importance")
        axes[idx].set_title(
            f"Top 15 Features - {model_data['name']}", fontsize=14, fontweight="bold"
        )
        axes[idx].invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path.split('/')[-1]}")


def plot_class_distribution(y, save_path):
    plt.figure(figsize=(10, 6))

    class_counts = y.value_counts()
    colors = plt.cm.Set3(range(len(class_counts)))

    bars = plt.bar(range(len(class_counts)), class_counts.values, color=colors)
    plt.xticks(range(len(class_counts)), class_counts.index, rotation=45)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.title("Class Distribution", fontsize=14, fontweight="bold")

    for bar, count in zip(bars, class_counts.values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count}\n({count / len(y) * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path.split('/')[-1]}")


def plot_multiclass_only_confusion_matrices(results, y_test, class_names, save_path):
    """Plot confusion matrices for multiclass-only datasets."""
    n_models = len(results)
    _, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))

    if n_models == 1:
        axes = [axes]

    colors = ["Blues", "Greens", "Oranges"]

    for idx, result in enumerate(results):
        cm = confusion_matrix(y_test, result["predictions"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=colors[idx % len(colors)],
            ax=axes[idx],
            xticklabels=class_names,
            yticklabels=class_names,
        )
        axes[idx].set_title(
            f"{result['model_name']} - Multiclass Classification",
            fontsize=14,
            fontweight="bold",
        )
        axes[idx].set_ylabel("True class")
        axes[idx].set_xlabel("Predicted class")
        axes[idx].tick_params(axis="x", rotation=45)
        axes[idx].tick_params(axis="y", rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.split('/')[-1]}")


def plot_binary_only_confusion_matrices(results, y_test, class_names, save_path):
    """Plot confusion matrices for binary-only datasets."""
    n_models = len(results)
    _, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    colors = ["Blues", "Greens", "Oranges"]

    for idx, result in enumerate(results):
        cm = confusion_matrix(y_test, result["predictions"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=colors[idx % len(colors)],
            ax=axes[idx],
            xticklabels=class_names,
            yticklabels=class_names,
        )
        axes[idx].set_title(
            f"{result['model_name']} - Binary Classification",
            fontsize=14,
            fontweight="bold",
        )
        axes[idx].set_ylabel("True class")
        axes[idx].set_xlabel("Predicted class")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.split('/')[-1]}")


def plot_model_comparison(results_df, save_path):
    """Plot comparison of all model configurations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Prepare data
    configs = results_df['Model Config'].unique()
    models = results_df['Model'].unique()
    classification_types = results_df['Classification Type'].unique()

    # Colors for configurations
    colors = plt.cm.Set2(range(len(configs)))
    config_colors = {config: colors[i] for i, config in enumerate(configs)}

    # Plot for each classification type
    for ax_idx, class_type in enumerate(classification_types):
        ax = axes[ax_idx] if len(classification_types) > 1 else axes[0]

        type_data = results_df[results_df['Classification Type'] == class_type]

        x = range(len(models))
        width = 0.8 / len(configs)

        for i, config in enumerate(configs):
            config_data = type_data[type_data['Model Config'] == config]
            accuracies = []
            f1_scores = []

            for model in models:
                model_data = config_data[config_data['Model'] == model]
                if len(model_data) > 0:
                    accuracies.append(model_data['Accuracy'].values[0])
                    f1_scores.append(model_data['F1-score'].values[0])
                else:
                    accuracies.append(0)
                    f1_scores.append(0)

            offset = (i - len(configs) / 2 + 0.5) * width
            bars = ax.bar([xi + offset for xi in x], accuracies, width,
                          label=config, color=config_colors[config], alpha=0.8)

            # Add F1 score as text on bars
            for bar, f1 in zip(bars, f1_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'F1:{f1:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)

        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{class_type} Classification - Model Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.15)
        ax.legend(title='Configuration', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)

    # Hide unused subplot if only one classification type
    if len(classification_types) == 1:
        axes[1].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path.split('/')[-1]}")