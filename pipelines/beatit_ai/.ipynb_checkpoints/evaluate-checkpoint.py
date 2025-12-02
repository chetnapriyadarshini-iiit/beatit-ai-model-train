import os
import json
import tarfile

import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import xgboost as xgb


def load_model(model_dir: str):
    """
    Loads the XGBoost model from the model.tar.gz artifact.
    """
    model_tar_path = None

    # Find model.tar.gz
    for fname in os.listdir(model_dir):
        if fname.endswith(".tar.gz"):
            model_tar_path = os.path.join(model_dir, fname)
            break

    if model_tar_path is None:
        raise FileNotFoundError(f"No .tar.gz model artifact found in {model_dir}")

    # Extract
    with tarfile.open(model_tar_path) as tar:
        tar.extractall(path=model_dir)

    # In SageMaker XGBoost container, the model is usually saved as 'xgboost-model'
    model_file = os.path.join(model_dir, "xgboost-model")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"xgboost-model not found in {model_dir} after extraction")

    booster = xgb.Booster()
    booster.load_model(model_file)
    return booster


def load_test_data(test_dir: str):
    """
    Loads the test.csv written by preprocess.py.

    Assumes:
    - headerless CSV
    - first column = label 'is_churn'
    - remaining columns = features
    """
    test_path = os.path.join(test_dir, "test.csv")
    if not os.path.exists(test_path):
        # Fallback: list directory contents to help debug
        raise FileNotFoundError(
            f"test.csv not found in {test_dir}. Files: {os.listdir(test_dir)}"
        )

    df = pd.read_csv(test_path, header=None)
    y = df.iloc[:, 0]        # label
    X = df.iloc[:, 1:]       # features
    return X, y


def evaluate(model_dir: str, test_dir: str, output_dir: str):
    """
    Runs evaluation and writes metrics to evaluation.json in the format
    expected by the SageMaker Pipeline.
    """
    print("Loading model...")
    booster = load_model(model_dir)

    print("Loading test data...")
    X_test, y_test = load_test_data(test_dir)

    print("Running predictions...")
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = booster.predict(dtest)        # probabilities for class 1
    y_pred_label = (y_pred_proba >= 0.5).astype(int)

    print("Computing metrics...")
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred_label)
    f1 = f1_score(y_test, y_pred_label)

    # Build evaluation report
    report = {
        "binary_classification_metrics": {
            "auc": {
                "value": auc,
            },
            "accuracy": {
                "value": acc,
            },
            "f1": {
                "value": f1,
            },
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "evaluation.json")
    with open(output_path, "w") as f:
        json.dump(report, f)

    print(f"Wrote evaluation report to {output_path}")
    print(json.dumps(report, indent=2))


def main():
    model_dir = "/opt/ml/processing/model"
    test_dir = "/opt/ml/processing/test"
    output_dir = "/opt/ml/processing/evaluation"

    evaluate(model_dir, test_dir, output_dir)


if __name__ == "__main__":
    main()
