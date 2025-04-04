import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

def clean_feature_names(df):
    cleaned_columns = df.columns.str.replace(r'[\"\'{}\[\]:,]', '_', regex=True)
    df.columns = cleaned_columns
    return df


def data_load():
    print("=================================Start loading data============================")
    df = pd.read_csv("21-22_TEST.csv")
    df = shuffle(df, random_state=42)

    # X = df.drop(columns=["FIRST_DECISION"])
    X = df.drop(columns=["FIRST_DECISION", df.columns[0]])
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


    # 使用 StratifiedShuffleSplit 创建子集，保证类别分布一致
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
