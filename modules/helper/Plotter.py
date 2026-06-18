import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_metrics(data, min_value=None):
    """
    Parameters
    ----------
    data:
        list of dicts / dict of lists / list of lists

    min_value:
        dict like {"train_loss": 0.8, "val_accuracy": 0.5}
        Filters returned df only (plots always use full data)
    """

    # ---------------------------
    # Normalize input
    # ---------------------------
    if isinstance(data, dict):
        df = pd.DataFrame(data)

    elif isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict):
            df = pd.DataFrame(data)
        else:
            cols = [
                "epoch",
                "train_loss", "val_loss",
                "train_accuracy", "train_precision", "train_recall", "train_f1",
                "val_accuracy", "val_precision", "val_recall", "val_f1"
            ]
            df = pd.DataFrame(data, columns=cols)

    else:
        raise ValueError("Unsupported data format")

    df = df.sort_values("epoch").reset_index(drop=True)

    sns.set_style("darkgrid")

    # ---------------------------
    # Plot
    # ---------------------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    def plot_pair(ax, y1, y2, title):
        sns.lineplot(
            data=df,
            x="epoch",
            y=y1,
            label=y1,
            marker="o",
            ax=ax
        )

        sns.lineplot(
            data=df,
            x="epoch",
            y=y2,
            label=y2,
            marker="o",
            ax=ax
        )

        ax.set_title(title)

    plot_pair(axes[0], "train_loss", "val_loss", "Loss")
    plot_pair(axes[1], "train_accuracy", "val_accuracy", "Accuracy")
    plot_pair(axes[2], "train_precision", "val_precision", "Precision")
    plot_pair(axes[3], "train_recall", "val_recall", "Recall")
    plot_pair(axes[4], "train_f1", "val_f1", "F1 Score")

    axes[5].axis("off")

    plt.tight_layout()
    plt.show()

    # ---------------------------
    # Optional filtering
    # ---------------------------
    if min_value:
        filtered_df = df.copy()

        for metric, value in min_value.items():
            filtered_df = filtered_df[filtered_df[metric] >= value]

        return filtered_df

    return df



def plot_testing_history(test_scores):
    """
    Plot testing metrics across epochs.

    Parameters
    ----------
    test_scores:
        List of dicts returned by Tester.test_all_checkpoints()
    """

    # ---------------------------
    # Extract metrics
    # ---------------------------
    history = []

    for result in test_scores:
        history.append({
            "epoch": result["epoch"],
            "test_loss": result["test_loss"],
            "test_accuracy": result["test_accuracy"],
            "test_precision": result["test_precision"],
            "test_recall": result["test_recall"],
            "test_f1": result["test_f1"]
        })

    df = pd.DataFrame(history)

    # Sort by actual checkpoint epoch
    df = df.sort_values("epoch").reset_index(drop=True)

    sns.set_style("darkgrid")

    # ---------------------------
    # Plot
    # ---------------------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    metrics = [
        "test_loss",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1"
    ]

    titles = [
        "Test Loss",
        "Test Accuracy",
        "Test Precision",
        "Test Recall",
        "Test F1 Score"
    ]

    for ax, metric, title in zip(axes, metrics, titles):

        sns.lineplot(
            data=df,
            x="epoch",
            y=metric,
            marker="o",
            ax=ax
        )

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)

    axes[-1].axis("off")

    plt.tight_layout()
    plt.show()

    return df