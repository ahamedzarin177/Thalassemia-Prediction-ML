import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


def compute_specificity(cm):
    specs = []
    for c in range(len(cm)):
        tn = cm.sum() - (cm[c].sum() + cm[:, c].sum() - cm[c][c])
        fp = cm[:, c].sum() - cm[c][c]
        specs.append(tn / (tn + fp + 1e-9))
    return np.mean(specs) * 100


def caon_objective(p, X_tr, y_tr, X_val, y_val):
    """
    PSO minimizes the objective, so we return negative accuracy.
    p = [n_estimators, max_depth, learning_rate, subsample, colsample_bytree]
    """
    try:
        model = XGBClassifier(
            n_estimators=int(p[0]),
            max_depth=int(p[1]),
            learning_rate=float(p[2]),
            subsample=float(p[3]),
            colsample_bytree=float(p[4]),
            objective="multi:softprob",
            num_class=len(np.unique(y_tr)),   # fold-specific
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42,
            missing=np.nan,
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        return -accuracy_score(y_val, preds)   # PSO minimizes
    except Exception:
        return 1.0


# CAON (PSO-based optimization) search space
LB = [50, 3, 0.01, 0.5, 0.5]
UB = [300, 12, 0.3, 1.0, 1.0]


def build_xgb_from_params(best_params, num_class: int, random_state: int):
    """
    best_params = [n_estimators, max_depth, learning_rate, subsample, colsample_bytree]
    """
    return XGBClassifier(
        n_estimators=int(best_params[0]),
        max_depth=int(best_params[1]),
        learning_rate=float(best_params[2]),
        subsample=float(best_params[3]),
        colsample_bytree=float(best_params[4]),
        objective="multi:softprob",
        num_class=int(num_class),
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=int(random_state),
        missing=np.nan,
    )

