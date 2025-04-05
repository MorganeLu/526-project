import os
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix

def clean_feature_names(df):
    cleaned_columns = df.columns.str.replace(r'[\"\'{}\[\]:,]', '_', regex=True)
    df.columns = cleaned_columns
    return df

def logistic_data_load():
    print("=================================Start loading data============================")
    df = pd.read_csv("21-22.csv")
    df = shuffle(df, random_state=42)

    # X = df.drop(columns=["FIRST_DECISION"])
    X = df.drop(columns=["FIRST_DECISION", df.columns[0]])
    X = pd.get_dummies(X, columns=['state'], drop_first=True)
    X = clean_feature_names(X)
    y = df["FIRST_DECISION"]


    X_remaining, X_test, y_remaining, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # subset_size = 20000
    # subsets = []
    # for i in range(3):
    #     start = i * subset_size
    #     end = start + subset_size
    #     X_subset = X_remaining.iloc[start:end].copy()
    #     y_subset = y_remaining.iloc[start:end].copy()
    #     subsets.append((X_subset, y_subset))

    # return subsets, X_test.reset_index(drop=True), y_test.reset_index(drop=True)


    # label ratio keep the same
    subset_size = 20000
    num_subsets = 3
    subsets = []

    sss = StratifiedShuffleSplit(n_splits=num_subsets, train_size=subset_size, random_state=42)
    used_indices = set()
    for i, (subset_indices, _) in enumerate(sss.split(X_remaining, y_remaining)):
        subset_indices = [idx for idx in subset_indices if idx not in used_indices][:subset_size]
        used_indices.update(subset_indices)

        X_subset = X_remaining.iloc[subset_indices].copy()
        y_subset = y_remaining.iloc[subset_indices].copy()
        subsets.append((X_subset, y_subset))

    return subsets, X_test.reset_index(drop=True), y_test.reset_index(drop=True)


def data_load():
    print("=================================Start loading data=============================")
    df = pd.read_csv("21-22.csv")
    df = shuffle(df, random_state=42)

    # X = df.drop(columns=["FIRST_DECISION"])
    X = df.drop(columns=["FIRST_DECISION", df.columns[0]])
    X = clean_feature_names(X)
    y = df["FIRST_DECISION"]

    X_remaining, X_test, y_remaining, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # # test: 2023
    # X_remaining = X
    # y_remaining = y

    # df_test = pd.read_csv("23.csv")
    # X_test = df_test.drop(columns=["FIRST_DECISION", df_test.columns[0]])
    # X_test = clean_feature_names(X_test)
    # y_test = df_test["FIRST_DECISION"]

    # label ratio keep the same
    subset_size = 20000
    num_subsets = 3
    subsets = []

    sss = StratifiedShuffleSplit(n_splits=num_subsets, train_size=subset_size, random_state=42)
    used_indices = set()
    for i, (subset_indices, _) in enumerate(sss.split(X_remaining, y_remaining)):
        subset_indices = [idx for idx in subset_indices if idx not in used_indices][:subset_size]
        used_indices.update(subset_indices)

        X_subset = X_remaining.iloc[subset_indices].copy()
        y_subset = y_remaining.iloc[subset_indices].copy()
        subsets.append((X_subset, y_subset))

    return subsets, X_test.reset_index(drop=True), y_test.reset_index(drop=True)


def print_param_lr(model, feature_names=None):
    coef = model.coef_[0]
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(coef))]

    print("模型系数：")
    for name, weight in zip(feature_names, coef):
        print(f"{name}: {weight:.4f}")

def plot_confusion_matrix(y_true, y_pred, model_name="Model", output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Confusion matrix saved to: {output_path}")
