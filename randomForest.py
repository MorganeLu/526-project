import utils
import models
import shap
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


subsets, X_test, y_test = utils.data_load()
for i, (X_train, y_train) in enumerate(subsets):
    print(f"Subset {i+1}: {X_train.shape}, {y_train.shape}")

rf_models, rf_test_preds = models.train_random_forest_models(subsets, X_test)


final_preds_rf = models.majority_vote_from_probs(rf_test_preds)
avg_proba_rf = np.mean(rf_test_preds, axis=0)


print("Random Forest Model Evaluation")
print(classification_report(y_test, final_preds_rf))
print("AUC-ROC:", roc_auc_score(y_test, avg_proba_rf))

utils.plot_confusion_matrix(y_test, final_preds_rf, model_name="randomForest")
# models.shap_summary_ensemble_rf(rf_models, X_test)


# hyperparam tuning
print("\n\n================================Start Tuning=================================")
def tune_random_forest(subsets, X_test, y_test, param_grid):
    results = []

    for i, params in enumerate(param_grid):
        print(f"\nStart tuning for round {i+1}: {params}")

        test_preds = []
        models_list = []

        for j, (X_train, y_train) in enumerate(subsets):
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            test_preds.append(proba)
            models_list.append(model)

        final_preds = models.majority_vote_from_probs(test_preds)
        avg_proba = np.mean(test_preds, axis=0)

        auc = roc_auc_score(y_test, avg_proba)
        print(f"AUC-ROC: {auc:.4f}")
        print(classification_report(y_test, final_preds))

        results.append({
            "params": params,
            "auc": auc
        })

    print("\nAll results:")
    sorted_results = sorted(results, key=lambda x: x["auc"], reverse=True)
    for item in sorted_results:
        print(f"AUC={item['auc']:.4f} -> {item['params']}")

    return sorted_results

param_grid = [
    {"n_estimators": 100},
    {"n_estimators": 200, "max_depth": 10},
    {"n_estimators": 300, "max_depth": 15, "min_samples_split": 5},
    {"n_estimators": 150, "max_depth": None, "max_features": "sqrt"},
]

best_rf_results = tune_random_forest(subsets, X_test, y_test, param_grid)
