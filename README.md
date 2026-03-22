# Beatit-AI — Model Training Pipeline (AWS SageMaker)

> **Part of the [Beatit-AI Churn Prediction System](https://github.com/chetnapriyadarshini-iiit)** — a production MLOps system for predicting music streaming subscriber churn on AWS.

This repository implements the **SageMaker Training Pipeline** for the Beatit churn model — orchestrating preprocessing, model training, evaluation, and automatic registration in the SageMaker Model Registry via AWS CodePipeline and CodeBuild.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Pipeline Steps](#pipeline-steps)
- [Technologies Used](#technologies-used)
- [Setup and Running](#setup-and-running)
- [Related Repositories](#related-repositories)
- [Contact](#contact)

---

## Overview

This repository contains the model build component of the Beatit MLOps system. It defines a SageMaker Pipeline with preprocessing, training, and evaluation steps. Upon successful evaluation against defined performance thresholds, the trained model is automatically registered in the SageMaker Model Registry as a versioned `ModelPackage`, ready for downstream deployment.

The pipeline is triggered automatically via **AWS CodePipeline** on code commits, or can be run manually from SageMaker Studio using the included notebook.

---

## Pipeline Architecture

```
Gold ML Features (S3 / Redshift)
           │
           ▼
┌──────────────────────────┐
│   Preprocessing Step     │  preprocess.py
│   · Feature engineering  │  Runs as SageMaker
│   · Train/val/test split │  Processing Job
│   · Data validation      │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Training Step          │  pipeline.py
│   · XGBoost / sklearn    │  Runs as SageMaker
│   · Hyperparameter config│  Training Job
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Evaluation Step        │  evaluate.py
│   · AUC, Precision,      │  Runs as SageMaker
│     Recall, F1           │  Processing Job
│   · Model card generation│
└────────────┬─────────────┘
             │  Thresholds met?
             ▼
┌──────────────────────────┐
│   Model Registration     │
│   · SageMaker Model      │
│     Registry             │
│   · Status: Approved     │
│   · Triggers deploy      │
│     pipeline             │
└──────────────────────────┘
```

---

## Dataset

The pipeline is built around the **KKBox Churn Prediction** dataset (Kaggle), adapted for the Beatit music streaming use case. It includes:

- `members.csv` — user demographics and registration metadata
- `transactions.csv` — subscription and payment activity
- `user_logs.csv` — daily listening behaviour logs
- `train.csv` — labelled churn / no-churn target variable

Raw data is uploaded to S3 and the `gold_ml_features` table (produced by the [data pipeline repo](https://github.com/chetnapriyadarshini-iiit/beatit-ai-glue-redshift-tables)) is used as the primary model input.

---

## Repository Structure

```
beatit-ai-model-train/
├── pipelines/
│   ├── pipeline.py            # Core SageMaker Pipeline definition (get_pipeline)
│   ├── preprocess.py          # Feature engineering and data splitting
│   ├── evaluate.py            # Model evaluation and metric generation
│   ├── run_pipeline.py        # Pipeline execution entrypoint
│   ├── get_pipeline_definition.py  # Exports pipeline JSON for CodeBuild
│   └── _utils.py              # Helper utilities
├── tests/
│   └── test_pipelines.py      # Unit tests for pipeline components
├── img/                       # Architecture and run screenshots
├── sagemaker-pipelines-project.ipynb  # Studio notebook for interactive runs
├── codebuild-buildspec.yml    # CodeBuild instructions for CI/CD trigger
├── setup.py / setup.cfg       # Package configuration
├── tox.ini                    # Test runner configuration
└── .coveragerc                # Test coverage configuration
```

---

## Pipeline Steps

| Step | Script | Description |
|---|---|---|
| **Preprocessing** | `preprocess.py` | Loads gold features from S3, applies feature engineering, outputs train/val/test splits |
| **Training** | `pipeline.py` | Trains a classification model with configured hyperparameters |
| **Evaluation** | `evaluate.py` | Computes AUC, Precision, Recall, F1 on the test set; generates model card |
| **Registration** | (pipeline.py) | Registers model in SageMaker Model Registry; sets approval status based on metric thresholds |

---

## Technologies Used

| Service / Tool | Purpose |
|---|---|
| **AWS SageMaker Pipelines** | ML workflow orchestration |
| **AWS CodePipeline / CodeBuild** | CI/CD trigger on code commit |
| **SageMaker Model Registry** | Versioned model storage and approval workflow |
| **Amazon S3** | Data and artefact storage |
| **Python / scikit-learn / XGBoost** | Model training and preprocessing |
| **pytest / tox** | Unit testing and test runner |

---

## Setup and Running

```bash
git clone https://github.com/chetnapriyadarshini-iiit/beatit-ai-model-train.git
cd beatit-ai-model-train
pip install -e .

# Run tests
tox

# Execute pipeline from CLI
python pipelines/run_pipeline.py
```

Or open `sagemaker-pipelines-project.ipynb` in SageMaker Studio for interactive execution.

---

## Related Repositories

| Repository | Role |
|---|---|
| [beatit-ai-glue-redshift-tables](https://github.com/chetnapriyadarshini-iiit/beatit-ai-glue-redshift-tables) | Produces the `gold_ml_features` consumed by this pipeline |
| [beatit_ai_common_utilites](https://github.com/chetnapriyadarshini-iiit/beatit_ai_common_utilites) | Shared utilities imported by pipeline scripts |
| [beatit-ai-model-deploy](https://github.com/chetnapriyadarshini-iiit/beatit-ai-model-deploy) | Deploys the model registered by this pipeline |
| [beatit-ai-model-monitor](https://github.com/chetnapriyadarshini-iiit/beatit-ai-model-monitor) | Monitors the deployed endpoint for drift |

---

## Contact

Created by [@chetnapriyadarshini](https://github.com/chetnapriyadarshini) — feel free to reach out with questions or suggestions.
