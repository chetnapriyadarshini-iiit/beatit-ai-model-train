import os
import json
import pandas as pd

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def load_transform_output(transform_dir: str):
    """
    Loads Batch Transform output CSVs.

    Expected format per row:
        prediction, true_label
    """
    files = [
        os.path.join(transform_dir, f)
        for f in os.listdir(transform_dir)
        if f.endswith(".out") or f.endswith(".csv")
    ]

    if not files:
        raise FileNotFoundError(
            f"No transform output CSV files found in {transform_dir}"
        )

    print(f"[evaluate] Found transform files: {files}")

    dfs = [pd.read_csv(f, header=None) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    if df.shape[1] != 2:
        raise ValueError(
            f"Expected 2 columns (prediction, label), but got {df.shape[1]}"
        )

    y_pred_proba = df.iloc[:, 0]
    y_true = df.iloc[:, 1]

    return y_true, y_pred_proba

def evaluate(transform_dir: str, output_dir: str):
    print(f"[evaluate] transform_dir={transform_dir}")
    print(f"[evaluate] output_dir={output_dir}")

    y_true, y_pred_proba = load_transform_output(transform_dir)

    print("Computing metrics...")
    y_pred_label = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_pred_proba)
    acc = accuracy_score(y_true, y_pred_label)
    f1 = f1_score(y_true, y_pred_label)

    report = {
        "binary_classification_metrics": {
            "auc": {"value": float(auc)},
            "accuracy": {"value": float(acc)},
            "f1": {"value": float(f1)},
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "evaluation.json")

    with open(output_path, "w") as f:
        json.dump(report, f)

    print(f"Wrote evaluation report to {output_path}")
    print(json.dumps(report, indent=2))
    

def main():
    transform_dir = "/opt/ml/processing/transform"
    output_dir = "/opt/ml/processing/evaluation"

    evaluate(transform_dir, output_dir)


if __name__ == "__main__":
    main()
