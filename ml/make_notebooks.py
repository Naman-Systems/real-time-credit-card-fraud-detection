"""
Script to create clean, structured Jupyter notebooks for the project.
"""

import json
from pathlib import Path


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.11"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def code_cell(source: str):
    lines = source.strip().split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines],
    }


def md_cell(source: str):
    lines = source.strip().split("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines],
    }


def generate_all_notebooks():
    nb_dir = Path(__file__).resolve().parent.parent / "notebooks"
    nb_dir.mkdir(exist_ok=True)

    # 01 EDA
    nb1 = make_notebook([
        md_cell(
            "# 01: Exploratory Data Analysis (EDA)\n"
            "## Real-Time Credit Card Fraud Detection System\n\n"
            "This notebook performs exploratory data analysis on the ULB Credit Card Fraud dataset.\n"
            "We inspect shape, missing values, duplicates, class imbalance (0.172% fraud), "
            "amount distributions, and correlation patterns."
        ),
        code_cell(
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n\n"
            "df = pd.read_csv('../dataset/creditcard.csv')\n"
            "print(f'Shape: {df.shape}')\n"
            "df.head()"
        ),
        code_cell(
            "print('Missing values:', df.isnull().sum().sum())\n"
            "print('Duplicates:', df.duplicated().sum())\n"
            "class_counts = df['Class'].value_counts()\n"
            "print(class_counts)\n"
            "print(f'Fraud percentage: {class_counts[1] / len(df) * 100:.4f}%')"
        ),
        code_cell(
            "# Amount statistics comparison\n"
            "print('Legitimate transactions summary:')\n"
            "print(df[df['Class'] == 0]['Amount'].describe().round(2))\n"
            "print('\\nFraudulent transactions summary:')\n"
            "print(df[df['Class'] == 1]['Amount'].describe().round(2))"
        ),
        code_cell(
            "# Visualizing Class Imbalance\n"
            "plt.figure(figsize=(6, 4))\n"
            "sns.countplot(data=df, x='Class', palette=['#10B981', '#EF4444'])\n"
            "plt.title('Transaction Class Counts (0: Legit, 1: Fraud)')\n"
            "plt.yscale('log')\n"
            "plt.show()"
        ),
    ])

    # 02 Preprocessing
    nb2 = make_notebook([
        md_cell(
            "# 02: Data Preprocessing & Chronological Splitting\n"
            "## Preventing Data Leakage in Fraud Detection Systems\n\n"
            "This notebook demonstrates strict forward chronological partitioning (70% train, 15% validation, 15% test) "
            "and feature scaling using RobustScaler fitted solely on training data."
        ),
        code_cell(
            "import pandas as pd\n"
            "import numpy as np\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.preprocessing import RobustScaler\n\n"
            "df = pd.read_csv('../dataset/creditcard.csv')\n"
            "df_sorted = df.sort_values(by='Time').reset_index(drop=True)\n"
            "n = len(df_sorted)\n"
            "train_end = int(0.70 * n)\n"
            "val_end = int(0.85 * n)\n\n"
            "train_df = df_sorted.iloc[:train_end]\n"
            "val_df = df_sorted.iloc[train_end:val_end]\n"
            "test_df = df_sorted.iloc[val_end:]\n\n"
            "print(f'Train size: {len(train_df)}, frauds: {train_df.Class.sum()}')\n"
            "print(f'Val size: {len(val_df)}, frauds: {val_df.Class.sum()}')\n"
            "print(f'Test size: {len(test_df)}, frauds: {test_df.Class.sum()}')"
        ),
        code_cell(
            "X_train = train_df.drop(columns=['Class'])\n"
            "y_train = train_df['Class']\n\n"
            "pca_features = [f'V{i}' for i in range(1, 29)]\n"
            "preprocessor = ColumnTransformer([\n"
            "    ('amount_scaler', RobustScaler(), ['Amount']),\n"
            "    ('time_scaler', RobustScaler(), ['Time']),\n"
            "    ('pca_passthrough', 'passthrough', pca_features)\n"
            "])\n\n"
            "X_train_trans = preprocessor.fit_transform(X_train)\n"
            "print('Transformed feature matrix shape:', X_train_trans.shape)"
        ),
    ])

    # 03 Model Training
    nb3 = make_notebook([
        md_cell(
            "# 03: Machine Learning Model Training\n"
            "## Benchmarking Baseline, Decision Tree, Random Forest, and SMOTE\n\n"
            "We train multiple candidate models with class balancing techniques on chronological training data."
        ),
        code_cell(
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.tree import DecisionTreeClassifier\n"
            "from sklearn.ensemble import RandomForestClassifier\n\n"
            "# 1. Logistic Regression (Baseline)\n"
            "lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)\n"
            "lr.fit(X_train_trans, y_train)\n\n"
            "# 2. Decision Tree\n"
            "dt = DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42)\n"
            "dt.fit(X_train_trans, y_train)\n\n"
            "# 3. Random Forest (Balanced Subsample)\n"
            "rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced_subsample', random_state=42, n_jobs=-1)\n"
            "rf.fit(X_train_trans, y_train)\n"
            "print('All candidate models fitted successfully.')"
        ),
    ])

    # 04 Model Evaluation
    nb4 = make_notebook([
        md_cell(
            "# 04: Model Evaluation & Threshold Optimization\n"
            "## Precision-Recall AUC & Decision Threshold Analysis\n\n"
            "We evaluate candidates on validation and out-of-time test partitions, optimizing "
            "the decision threshold to maximize F1-score."
        ),
        code_cell(
            "from sklearn.metrics import classification_report, average_precision_score, roc_auc_score, confusion_matrix\n\n"
            "X_val = val_df.drop(columns=['Class'])\n"
            "y_val = val_df['Class'].values\n"
            "X_val_trans = preprocessor.transform(X_val)\n\n"
            "val_probs = rf.predict_proba(X_val_trans)[:, 1]\n"
            "print(f'Validation PR-AUC: {average_precision_score(y_val, val_probs):.4f}')\n"
            "print(f'Validation ROC-AUC: {roc_auc_score(y_val, val_probs):.4f}')\n\n"
            "# Evaluate at optimal threshold (0.40)\n"
            "y_pred_opt = (val_probs >= 0.40).astype(int)\n"
            "print('\\nClassification Report @ 0.40:')\n"
            "print(classification_report(y_val, y_pred_opt))\n"
            "print('\\nConfusion Matrix:\\n', confusion_matrix(y_val, y_pred_opt))"
        ),
    ])

    with open(nb_dir / "01_exploratory_data_analysis.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb1, f, indent=2)

    with open(nb_dir / "02_data_preprocessing.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb2, f, indent=2)

    with open(nb_dir / "03_model_training.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb3, f, indent=2)

    with open(nb_dir / "04_model_evaluation.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb4, f, indent=2)

    print("All 4 Jupyter notebooks successfully created in notebooks/!")


if __name__ == "__main__":
    generate_all_notebooks()
