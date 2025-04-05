import utils
import models
import shap
import numpy as np
import lightgbm as lgb
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score


subsets, X_test, y_test = utils.logistic_data_load()
for i, (X_train, y_train) in enumerate(subsets):
    print(f"Subset {i+1}: {X_train.shape}, {y_train.shape}")


# lr_models, lr_scalers, lr_test_preds = models.train_logistic_models(subsets, X_test)

# final_preds = models.majority_vote_from_probs(lr_test_preds)
# avg_proba = np.mean(lr_test_preds, axis=0)

# print(classification_report(y_test, final_preds))
# print("Logistic Regression AUC-ROC:", roc_auc_score(y_test, avg_proba))
# feature_names = subsets[0][0].columns
# utils.print_param_lr(lr_models[0], feature_names)


# with L1
l1_models, l1_scalers, l1_test_preds = models.train_logistic_l1_models_tune(subsets, X_test)
final_preds_l1 = models.majority_vote_from_probs(l1_test_preds)
avg_proba_l1 = np.mean(l1_test_preds, axis=0)
print(" L1 Reg Evaluation:")
print(classification_report(y_test, final_preds_l1))
print("Logistic Regression with L1 AUC-ROC:", roc_auc_score(y_test, avg_proba_l1))


# # with L2
# l2_models, l2_scalers, l2_test_preds = models.train_logistic_l2_models_tune(subsets, X_test)
# final_preds_l2 = models.majority_vote_from_probs(l2_test_preds)
# avg_proba_l2 = np.mean(l2_test_preds, axis=0)
# print(" L2 Reg Evaluation:")
# print(classification_report(y_test, final_preds_l2))
# print("Logistic Regression with L2 AUC-ROC:", roc_auc_score(y_test, avg_proba_l2))

feature_names = subsets[0][0].columns
utils.print_param_lr(l1_models[0], feature_names)


# # SMOTE with L2
# lr_models, lr_scalers, lr_test_preds = models.train_logistic_models_with_smote(subsets, X_test)
# final_preds = models.majority_vote_from_probs(lr_test_preds)
# avg_proba = np.mean(lr_test_preds, axis=0)

# print(classification_report(y_test, final_preds))
# print("Logistic Regression with SMOTE AUC-ROC:", roc_auc_score(y_test, avg_proba))




