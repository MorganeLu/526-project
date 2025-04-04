import shap
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
)
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE


# LGB
def train_lightgbm_models(subsets, X_test):
    test_preds = []  # 存储每个模型的预测概率
    models = []      # 存储模型本身

    for i, (X_train, y_train) in enumerate(subsets):
        print(f"\n =========================================Start training LightGBM model {i+1}=======================================")

        # 初始化 LightGBM 模型
        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            num_leaves=31,
            objective="binary"
        )

        # 执行 5 折交叉验证评估
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f" model {i+1} - AUC-ROC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # 用完整子集数据重新训练
        model.fit(X_train, y_train)
        models.append(model)

        # 在测试集上预测（正类概率）
        proba = model.predict_proba(X_test)[:, 1]
        test_preds.append(proba)

    return models, test_preds


# Logistic Regression
def train_logistic_models(subsets, X_test):
    test_preds = []
    models = []
    scalers = []

    for i, (X_train, y_train) in enumerate(subsets):
        print(f"\n 训练 L2 Logistic Regression 模型 {i+1}...")

        # 标准化数据
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 创建模型（L2 正则）
        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='liblinear',  # 支持 L1/L2 且稳定
            max_iter=1000
        )

        # 交叉验证评估
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        print(f" 模型 {i+1} 的 5 折 AUC-ROC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # 拟合并预测
        model.fit(X_train_scaled, y_train)
        proba = model.predict_proba(X_test_scaled)[:, 1]

        models.append(model)
        scalers.append(scaler)
        test_preds.append(proba)

    return models, scalers, test_preds

def train_logistic_l2_models(subsets, X_test):
    test_preds = []
    models = []
    scalers = []

    for i, (X_train, y_train) in enumerate(subsets):
        print(f"\n🔄 训练 L2 Logistic Regression 模型 {i+1}（无 SMOTE）")

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 构建模型
        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='liblinear',
            max_iter=1000
        )

        # 交叉验证
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        print(f"✅ L2 模型 {i+1} AUC-ROC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # 模型训练与预测
        model.fit(X_train_scaled, y_train)
        proba = model.predict_proba(X_test_scaled)[:, 1]

        models.append(model)
        scalers.append(scaler)
        test_preds.append(proba)

    return models, scalers, test_preds

def train_logistic_l1_models(subsets, X_test):
    test_preds = []
    models = []
    scalers = []

    for i, (X_train, y_train) in enumerate(subsets):
        print(f"\n训练 L1 Logistic Regression 模型 {i+1}（无 SMOTE）")

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 构建模型
        model = LogisticRegression(
            penalty='l1',
            C=1.0,
            solver='liblinear',
            max_iter=1000
        )

        # 交叉验证
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        print(f"L1 模型 {i+1} AUC-ROC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # 模型训练与预测
        model.fit(X_train_scaled, y_train)
        proba = model.predict_proba(X_test_scaled)[:, 1]

        models.append(model)
        scalers.append(scaler)
        test_preds.append(proba)

    return models, scalers, test_preds


def train_logistic_models_with_smote(subsets, X_test):
    test_preds = []
    models = []
    scalers = []

    for i, (X_train, y_train) in enumerate(subsets):
        print(f"\n训练 L2 Logistic Regression 模型 {i+1}（使用 SMOTE）")

        # 使用 SMOTE 过采样
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        # 标准化
        scaler = StandardScaler()
        X_resampled_scaled = scaler.fit_transform(X_resampled)
        X_test_scaled = scaler.transform(X_test)

        # 创建并训练模型
        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='liblinear',
            max_iter=1000
        )

        # 交叉验证评估
        cv_scores = cross_val_score(model, X_resampled_scaled, y_resampled, cv=5, scoring='roc_auc')
        print(f"模型 {i+1} 的 5 折 AUC-ROC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # 模型训练 + 测试集预测
        model.fit(X_resampled_scaled, y_resampled)
        proba = model.predict_proba(X_test_scaled)[:, 1]

        models.append(model)
        scalers.append(scaler)
        test_preds.append(proba)

    return models, scalers, test_preds


# Random Forest
def train_random_forest_models(subsets, X_test):
    test_preds = []
    models = []

    for i, (X_train, y_train) in enumerate(subsets):
        print(f"\n训练 Random Forest 模型 {i+1}(default params)")

        # 构建模型（默认参数）
        model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )

        # 交叉验证（5折 AUC-ROC）
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"模型 {i+1} AUC-ROC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # 训练模型 + 在测试集上预测概率
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        models.append(model)
        test_preds.append(proba)

    return models, test_preds


# general
def majority_vote_from_probs(prob_list, threshold=0.5):
    # 把概率转为 0/1
    binary_preds = [pred > threshold for pred in prob_list]

    # 求每个样本在3个模型中的平均预测（0～1）
    avg_preds = np.mean(binary_preds, axis=0)

    # 多数投票（0.5 以上为 1）
    final_preds = np.round(avg_preds).astype(int)

    return final_preds


def shap_summary_ensemble(models, X_test, output_dir="output"):
    print("==========================================Calculating SHAP value for each model===============================")

    shap_values_list = []
    for i, model in enumerate(models):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test, check_additivity=False)

        # 如果是 binary 分类，取 class 1 的 shap 值
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_values_list.append(shap_values)


    # 对多个 shap_values 取平均
    shap_values_avg = np.mean(np.array(shap_values_list), axis=0)

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    shap.summary_plot(
        shap_values_avg,
        X_test,
        plot_type="dot",
        max_display=20,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lgb_ensemble_shap_summary.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    shap.summary_plot(
        shap_values_avg,
        X_test,
        plot_type="bar",
        max_display=20,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lgb_ensemble_shap_feature_importance.png"))
    plt.close()

    print(f" SHAP summary saved to {output_dir}/")