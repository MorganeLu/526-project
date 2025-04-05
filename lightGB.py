import utils
import models
import shap
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


subsets, X_test, y_test = utils.data_load()


for i, (X_train, y_train) in enumerate(subsets):
    print(f"Subset {i+1}: {X_train.shape}, {y_train.shape}")

# start training
lgb_models, lgb_test_preds = models.train_lightgbm_models(subsets, X_test)

models.shap_summary_ensemble(lgb_models, X_test)

final_preds = models.majority_vote_from_probs(lgb_test_preds)
avg_proba = np.mean(lgb_test_preds, axis=0)

print("\nLightGB final evalution")
print(classification_report(y_test, final_preds))
print("AUC-ROC:", roc_auc_score(y_test, avg_proba))

utils.plot_confusion_matrix(y_test, final_preds, model_name="LightGB")


# param tuning
# param_grid = [
#     {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 7},
#     {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 10},
#     {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 12},
#     {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 7, "class_weight": "balanced"},
# ]

# def tune_lightgbm(subsets, X_test, y_test, param_grid):
#     results = []

#     for i, params in enumerate(param_grid):
#         print(f"\nTry hyper params in round {i+1}: {params}")

#         test_preds = []
#         models_list = []

#         for j, (X_train, y_train) in enumerate(subsets):
#             model = lgb.LGBMClassifier(**params, random_state=42)
#             model.fit(X_train, y_train)
#             proba = model.predict_proba(X_test)[:, 1]
#             test_preds.append(proba)
#             models_list.append(model)

#         final_preds = models.majority_vote_from_probs(test_preds)
#         avg_proba = np.mean(test_preds, axis=0)

#         auc = roc_auc_score(y_test, avg_proba)
#         print(f"AUC-ROC: {auc:.4f}")
#         print(classification_report(y_test, final_preds))

#         results.append({
#             "params": params,
#             "auc": auc
#         })

#     print("\nAll results：")
#     sorted_results = sorted(results, key=lambda x: x["auc"], reverse=True)
#     for item in sorted_results:
#         print(f"AUC={item['auc']:.4f} -> {item['params']}")

#     return sorted_results

# best_results = tune_lightgbm(subsets, X_test, y_test, param_grid)