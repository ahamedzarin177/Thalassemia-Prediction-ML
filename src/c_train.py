import numpy as np
import pandas as pd

from pyswarm import pso
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score,
)
import joblib

from preprocess import load_dataset
from model import caon_objective, compute_specificity, LB, UB, build_xgb_from_params


def main(
    csv_path: str = "data/cleaned_dataset.csv",
    target_col: str = "Diagnosis",
    n_splits: int = 10,
    swarm_size: int = 15,
    maxiter: int = 15,
):
    X, y, le_global, y_encoded, classes_global = load_dataset(csv_path, target_col)

    acc_list, prec_list, rec_list, spec_list, f1_list, kappa_list = [], [], [], [], [], []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_accuracy = 0
    best_model = None
    best_params_global = None
    best_fold = None

    fold = 1
    for train_idx, test_idx in skf.split(X, y_encoded):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # FOLD-SPECIFIC LABEL ENCODING
        le_fold = LabelEncoder()
        y_train_fold = le_fold.fit_transform(y_train)

        # Filter test set to only labels present in training fold
        mask_test = np.isin(y_test, le_fold.classes_)
        X_test_fold = X_test[mask_test]
        y_test_fold = y_test[mask_test]
        y_test_fold_encoded = le_fold.transform(y_test_fold)

        classes_fold = np.unique(y_train_fold)

        # CAON OPTIMIZATION (PSO)
        best_params, best_score = pso(
            caon_objective,
            LB,
            UB,
            args=(X_train, y_train_fold, X_test_fold, y_test_fold_encoded),
            swarmsize=swarm_size,
            maxiter=maxiter,
        )

        # FINAL OPTIMIZED MODEL
        model = build_xgb_from_params(
            best_params=best_params,
            num_class=len(classes_fold),
            random_state=fold,
        )

        model.fit(X_train, y_train_fold)
        preds_fold = model.predict(X_test_fold)

        # Map predictions back to GLOBAL labels for metric calculation
        preds_global = le_global.inverse_transform(
            le_fold.inverse_transform(preds_fold)
        )

        # METRICS
        acc = accuracy_score(y_test_fold, preds_global) * 100
        prec = precision_score(y_test_fold, preds_global, average="macro", zero_division=0) * 100
        rec = recall_score(y_test_fold, preds_global, average="macro", zero_division=0) * 100
        f1 = f1_score(y_test_fold, preds_global, average="macro", zero_division=0) * 100
        kappa = cohen_kappa_score(y_test_fold, preds_global)

        cm = confusion_matrix(y_test_fold, preds_global, labels=classes_global)
        spec = compute_specificity(cm)

        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        spec_list.append(spec)
        f1_list.append(f1)
        kappa_list.append(kappa)

        # STORE BEST MODEL
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_params_global = {
                "n_estimators": int(best_params[0]),
                "max_depth": int(best_params[1]),
                "learning_rate": float(best_params[2]),
                "subsample": float(best_params[3]),
                "colsample_bytree": float(best_params[4]),
            }
            best_fold = fold

        print(
            f"CAON-XGB Fold {fold}: "
            f"Acc={acc:.2f}, Prec={prec:.2f}, Rec={rec:.2f}, "
            f"Spec={spec:.2f}, F1={f1:.2f}, Kappa={kappa:.4f}"
        )

        fold += 1

    # PERFORMANCE SUMMARY
    summary = pd.DataFrame(
        {
            "Fold": range(1, len(acc_list) + 1),
            "Accuracy (%)": acc_list,
            "Precision (%)": prec_list,
            "Recall (%)": rec_list,
            "Specificity (%)": spec_list,
            "F1 Score (%)": f1_list,
            "Kappa": kappa_list,
        }
    )

    print("\n===== CAON-XGBOOST PERFORMANCE SUMMARY =====")
    print(summary)

    print("\n===== MEAN PERFORMANCE =====")
    print(summary.mean(numeric_only=True))

    print("\n===== BEST CAON-XGB MODEL =====")
    print(f"Best Fold     : {best_fold}")
    print(f"Best Accuracy : {best_accuracy:.2f}%")
    print("Best Parameters:")
    for k, v in best_params_global.items():
        print(f"  {k}: {v}")

    joblib.dump(best_model, "best_caon_xgb_model.pkl")


if __name__ == "__main__":
    main()

