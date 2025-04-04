import utils
import models
import shap
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score


subsets, X_test, y_test = utils.data_load()
for i, (X_train, y_train) in enumerate(subsets):
    print(f"Subset {i+1}: {X_train.shape}, {y_train.shape}")

rf_models, rf_test_preds = models.train_random_forest_models(subsets, X_test)


final_preds_rf = models.majority_vote_from_probs(rf_test_preds)
avg_proba_rf = np.mean(rf_test_preds, axis=0)


print("Random Forest Model Evaluation")
print(classification_report(y_test, final_preds_rf))
print("AUC-ROC:", roc_auc_score(y_test, avg_proba_rf))
