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

# start training
lgb_models, lgb_test_preds = models.train_lightgbm_models(subsets, X_test)

models.shap_summary_ensemble(lgb_models, X_test)

final_preds = models.majority_vote_from_probs(lgb_test_preds)
avg_proba = np.mean(lgb_test_preds, axis=0)

print("\nLightGB final evalution")
print(classification_report(y_test, final_preds))
print("AUC-ROC:", roc_auc_score(y_test, avg_proba))
