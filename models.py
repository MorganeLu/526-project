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
from sklearn.model_selection import GridSearchCV


# LGB
def train_lightgbm_models(subsets, X_test):
    test_preds = []
    models = []

    for i, (X_train, y_train) in enumerate(subsets):
        print(f"\n =======================================Start training LightGBM model {i+1}=======================================")

        # init
        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            num_leaves=31,
            objective="binary",
            verbose=-1
        )

        # 5-fold CV
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f" model {i+1} - AUC-ROC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        model.fit(X_train, y_train)
        models.append(model)

        proba = model.predict_proba(X_test)[:, 1]
        test_preds.append(proba)

    return models, test_preds


# Logistic Regression
def train_logistic_models(subsets, X_test):
    test_preds = []
    models = []
    scalers = []

    for i, (X_train, y_train) in enumerate(subsets):
        print(f"\n Training L2 Logistic Regression Model {i+1}...")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='liblinear',
            max_iter=1000
        )

        # CV
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        print(f" Model {i+1} 5-fold AUC-ROC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

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
        print(f"\nTraining L2 Logistic Regression Model {i+1}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='liblinear',
            max_iter=1000
        )

        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        print(f"L2 Model {i+1} AUC-ROC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        model.fit(X_train_scaled, y_train)
        proba = model.predict_proba(X_test_scaled)[:, 1]

        models.append(model)
        scalers.append(scaler)
        test_preds.append(proba)

    return models, scalers, test_preds

def train_logistic_l2_models_tune(subsets, X_test):
    test_preds = []
    models = []
    scalers = []

    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

    for i, (X_train, y_train) in enumerate(subsets):
        print(f"\nTraining L2 Logistic Regression Model {i+1} with tune")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(
            penalty='l2',
            solver='liblinear', 
            max_iter=1000
        )

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        print(f"Best params: {grid_search.best_params_}")
        print(f"Best AUC-ROC: {grid_search.best_score_:.4f}")

        best_model = grid_search.best_estimator_
        proba = best_model.predict_proba(X_test_scaled)[:, 1]

        models.append(best_model)
        scalers.append(scaler)
        test_preds.append(proba)

    return models, scalers, test_preds

def train_logistic_l1_models(subsets, X_test):
    test_preds = []
    models = []
    scalers = []

    for i, (X_train, y_train) in enumerate(subsets):
        print(f"\nTraining L1 Logistic Regression Model {i+1}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(
            penalty='l1',
            C=1.0,
            solver='liblinear',
            max_iter=1000
        )

        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        print(f"L1 Model {i+1} AUC-ROC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        model.fit(X_train_scaled, y_train)
        proba = model.predict_proba(X_test_scaled)[:, 1]

        models.append(model)
        scalers.append(scaler)
        test_preds.append(proba)

    return models, scalers, test_preds

def train_logistic_l1_models_tune(subsets, X_test):
    test_preds = []
    models = []
    scalers = []

    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

    for i, (X_train, y_train) in enumerate(subsets):
        print(f"\nTraining L1 Logistic Regression Model{i+1} with tune")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            max_iter=1000
        )

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        print(f"best lamda: {grid_search.best_params_}")
        print(f"best AUC-ROC: {grid_search.best_score_:.4f}")

        best_model = grid_search.best_estimator_
        proba = best_model.predict_proba(X_test_scaled)[:, 1]

        models.append(best_model)
        scalers.append(scaler)
        test_preds.append(proba)

    return models, scalers, test_preds

def train_logistic_models_with_smote(subsets, X_test):
    test_preds = []
    models = []
    scalers = []

    for i, (X_train, y_train) in enumerate(subsets):
        print(f"\nTraining L2 Logistic Regression Model {i+1} with SMOTE")

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        scaler = StandardScaler()
        X_resampled_scaled = scaler.fit_transform(X_resampled)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='liblinear',
            max_iter=1000
        )

        cv_scores = cross_val_score(model, X_resampled_scaled, y_resampled, cv=5, scoring='roc_auc')
        print(f"Model {i+1} AUC-ROC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

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
        print(f"\nTrain Random Forest Model {i+1}(default params)")

        model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"Model {i+1} AUC-ROC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        models.append(model)
        test_preds.append(proba)

    return models, test_preds


# general
def majority_vote_from_probs(prob_list, threshold=0.5):
    # p to 0/1
    binary_preds = [pred > threshold for pred in prob_list]

    avg_preds = np.mean(binary_preds, axis=0)

    # majority vote
    final_preds = np.round(avg_preds).astype(int)

    return final_preds


def shap_summary_ensemble(models, X_test, output_dir="output"):
    print("==========================================Calculating SHAP value for each model===============================")

    shap_values_list = []
    for i, model in enumerate(models):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test, check_additivity=False)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_values_list.append(shap_values)

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

def shap_summary_ensemble_rf(rf_models, X_test, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    print("🔍 计算 SHAP 值中（Random Forest 集成模型）...")

    # 每个模型计算 shap 值
    shap_values_all = []
    for i, model in enumerate(rf_models):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)[1]  # 取正类
        shap_values_all.append(shap_values)

    # 取平均 SHAP 值（或者你可以选择 max/median）
    mean_shap_values = np.mean(shap_values_all, axis=0)

    # summary plot（点图）
    plt.figure()
    shap.summary_plot(mean_shap_values, X_test, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rf_shap_summary.png")
    plt.close()

    # bar plot（条形图）
    plt.figure()
    shap.summary_plot(mean_shap_values, X_test, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rf_shap_feature_importance.png")
    plt.close()

    print(f"✅ SHAP 图已保存到 {output_dir}/")
