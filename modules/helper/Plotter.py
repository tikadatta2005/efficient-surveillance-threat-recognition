import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_training_metrics(data):
    """
    Accepts:
    - list of dicts (preferred)
    - dict of lists
    - list of lists (ordered)

    Converts to pandas DataFrame and plots:
    - Loss (train vs val)
    - Accuracy
    - Precision
    - Recall
    - F1 score
    """

    # ---------------------------
    # Normalize input to DataFrame
    # ---------------------------
    if isinstance(data, dict):
        df = pd.DataFrame(data)

    elif isinstance(data, list):
        # list of dicts
        if len(data) > 0 and isinstance(data[0], dict):
            df = pd.DataFrame(data)
        else:
            # assume list of lists with fixed order
            cols = [
                "epoch",
                "train_loss", "val_loss",
                "train_accuracy", "train_precision", "train_recall", "train_f1",
                "val_accuracy", "val_precision", "val_recall", "val_f1"
            ]
            df = pd.DataFrame(data, columns=cols)

    else:
        raise ValueError("Unsupported data format")

    df = df.sort_values("epoch")

    sns.set_style("darkgrid")

    # ---------------------------
    # Loss Plot
    # ---------------------------
    plt.figure()
    sns.lineplot(data=df, x="epoch", y="train_loss", label="train_loss")
    sns.lineplot(data=df, x="epoch", y="val_loss", label="val_loss")
    plt.title("Loss Curve")
    plt.show()

    # ---------------------------
    # Accuracy Plot
    # ---------------------------
    plt.figure()
    sns.lineplot(data=df, x="epoch", y="train_accuracy", label="train_accuracy")
    sns.lineplot(data=df, x="epoch", y="val_accuracy", label="val_accuracy")
    plt.title("Accuracy Curve")
    plt.show()

    # ---------------------------
    # Precision Plot
    # ---------------------------
    plt.figure()
    sns.lineplot(data=df, x="epoch", y="train_precision", label="train_precision")
    sns.lineplot(data=df, x="epoch", y="val_precision", label="val_precision")
    plt.title("Precision Curve")
    plt.show()

    # ---------------------------
    # Recall Plot
    # ---------------------------
    plt.figure()
    sns.lineplot(data=df, x="epoch", y="train_recall", label="train_recall")
    sns.lineplot(data=df, x="epoch", y="val_recall", label="val_recall")
    plt.title("Recall Curve")
    plt.show()

    # ---------------------------
    # F1 Plot
    # ---------------------------
    plt.figure()
    sns.lineplot(data=df, x="epoch", y="train_f1", label="train_f1")
    sns.lineplot(data=df, x="epoch", y="val_f1", label="val_f1")
    plt.title("F1 Score Curve")
    plt.show()

    return df