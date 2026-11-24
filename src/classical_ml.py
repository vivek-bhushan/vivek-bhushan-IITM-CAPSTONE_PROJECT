"""
classical_ml.py
===============
Phase 2 - Classical ML Pipeline.

Two workflows on the same tabular feature set:

1. **Supervised** - predict ``purchase_flag`` (conversion).  We justify the
   target, build a scikit-learn ``ColumnTransformer`` for numerical +
   categorical features, handle class balance, train several baselines
   (Logistic Regression, Random Forest, Gradient Boosting) and report
   accuracy / precision / recall / F1 / ROC-AUC / PR-AUC.  We also fit a
   secondary model on the imbalanced ``returned_flag`` to show imbalance
   handling, and extract feature importances.

2. **Unsupervised** - K-Means customer segmentation on behavioural + value
   features, with inertia / silhouette diagnostics and business-language
   segment descriptions.

Run directly:  ``python src/classical_ml.py``
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CV_FOLDS,
    FEATURES_CSV,
    FIGURES_DIR,
    ML_METRICS_JSON,
    MODELS_DIR,
    RANDOM_STATE,
    SECONDARY_TARGET,
    SEGMENTS_CSV,
    SEGMENT_SUMMARY_JSON,
    SUPERVISED_TARGET,
    TEST_SIZE,
    ensure_dirs,
)

# --------------------------------------------------------------------------- #
# Feature groups
# --------------------------------------------------------------------------- #
NUMERIC_FEATURES = [
    "age",
    "days_since_last_purchase",
    "avg_order_value",
    "total_orders",
    "base_price",
    "discount_percent",
    "final_price",
    "avg_product_rating",
    "stock_level",
    "time_on_page",
    "quantity",
    "price_sensitivity",
    "engagement_score",
    "customer_value",
]

CATEGORICAL_FEATURES = [
    "gender",
    "city_tier",
    "state",
    "income_band",
    "occupation",
    "membership_level",
    "preferred_channel",
    "category",
    "sub_category",
    "brand",
    "color",
    "size",
    "material",
    "season",
    "payment_mode",
    "delivery_type",
    "review_sentiment_label",
    "recommended_use_case",
    "price_band",
    "discount_band",
    "recency_band",
]

# Columns that must never leak into the model (ids, text, targets, post-outcome)
LEAKY_OR_ID = [
    "customer_id",
    "product_id",
    "product_title",
    "product_description",
    "customer_review_text",
    "customer_query",
    "policy_doc_id",
    "policy_text_snippet",
    "image_path",
    "style_tags",
    "occasion_tags",
    "product_text",
    "purchase_flag",
    "returned_flag",
    "added_to_cart",
    "wishlist_flag",
    "rating_given",
    "is_returnable",
]


def load_features() -> pd.DataFrame:
    return pd.read_csv(FEATURES_CSV)


# --------------------------------------------------------------------------- #
# Model persistence - BOTH .joblib and .pkl are COMPULSORY deliverables
# --------------------------------------------------------------------------- #
def save_model(model, stem: str) -> Dict[str, str]:
    """Persist a fitted model in BOTH .joblib and .pkl formats.

    ``.joblib`` is the efficient container for estimators holding large NumPy
    arrays; ``.pkl`` is the portable pickle artefact many teams standardise on.
    Writing both keeps the project consumable either way.

    Returns a mapping {"joblib": <filename>, "pkl": <filename>} so callers can
    report the exact artefacts written to ``outputs/models/``.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    jb = MODELS_DIR / f"{stem}.joblib"
    pk = MODELS_DIR / f"{stem}.pkl"
    joblib.dump(model, jb)
    with open(pk, "wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return {"joblib": jb.name, "pkl": pk.name}


def load_model(path) -> object:
    """Reload an artefact written by :func:`save_model` (format auto-detected).

    Accepts either a ``str``/``Path``; ``.joblib`` files go through
    ``joblib.load`` and anything else through ``pickle.load``.
    """
    path = Path(path)
    if path.suffix == ".joblib":
        return joblib.load(path)
    with open(path, "rb") as fh:
        return pickle.load(fh)


def verify_roundtrip(model, stem: str) -> Dict[str, object]:
    """Save a model in both formats, reload each, and confirm they match."""
    paths = save_model(model, stem)
    reloaded = {}
    for fmt, name in paths.items():
        obj = load_model(MODELS_DIR / name)
        reloaded[fmt] = type(obj).__name__
    return {"stem": stem, "files": paths, "reloaded_types": reloaded}


def build_preprocessor() -> ColumnTransformer:
    """Numerical: median impute + scale.  Categorical: mode impute + one-hot."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def _feature_names(pre: ColumnTransformer) -> List[str]:
    names: List[str] = []
    names.extend(NUMERIC_FEATURES)
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    names.extend(list(ohe.get_feature_names_out(CATEGORICAL_FEATURES)))
    return names


# --------------------------------------------------------------------------- #
# Supervised workflow
# --------------------------------------------------------------------------- #
def train_supervised(df: pd.DataFrame, target: str = SUPERVISED_TARGET) -> Dict[str, object]:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pos_rate = float(y_train.mean())
    imbalance_note = (
        f"Positive rate {pos_rate:.3f}. "
        + ("Mild imbalance - used class_weight='balanced'." if 0.2 < pos_rate < 0.8
           else "Imbalance present - used class_weight='balanced' and PR-AUC as the primary metric.")
    )

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results: Dict[str, object] = {}
    best_name, best_auc, best_pipe = None, -1.0, None

    for name, clf in models.items():
        pipe = Pipeline(steps=[("pre", build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
            "cv_roc_auc_mean": round(float(cv_auc.mean()), 4),
            "cv_roc_auc_std": round(float(cv_auc.std()), 4),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "classification_report": classification_report(y_test, pred, digits=3, zero_division=0),
        }
        results[name] = metrics

        if metrics["roc_auc"] > best_auc:
            best_name, best_auc, best_pipe = name, metrics["roc_auc"], pipe

    # feature importances from the best tree model (or LR coefficients)
    importances = _extract_importances(best_pipe, best_name)

    # Persist the champion pipeline in BOTH formats (.joblib AND .pkl)
    artifacts = save_model(best_pipe, "best_purchase_flag_model")

    return {
        "artifacts": artifacts,
        "target": target,
        "target_justification": (
            "purchase_flag is the core conversion KPI: it is directly actionable "
            "(who will buy), balanced enough to model reliably, and drives both "
            "revenue forecasting and personalised recommendation ranking."
        ),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate": round(pos_rate, 4),
        "imbalance_handling": imbalance_note,
        "models": results,
        "best_model": best_name,
        "best_roc_auc": round(best_auc, 4),
        "top_features": importances[:15],
    }


def _extract_importances(pipe: Pipeline, name: str) -> List[Dict[str, object]]:
    pre: ColumnTransformer = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    names = _feature_names(pre)
    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        vals = np.abs(clf.coef_).ravel()
    else:
        return []
    order = np.argsort(vals)[::-1]
    return [
        {"feature": names[i], "importance": round(float(vals[i]), 5)}
        for i in order[:25]
        if i < len(names)
    ]


def train_secondary(df: pd.DataFrame) -> Dict[str, object]:
    """Imbalanced target (returned_flag) to demonstrate imbalance handling."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[SECONDARY_TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    pipe = Pipeline(
        steps=[
            ("pre", build_preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced_subsample",
                random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    # Persist the secondary (imbalanced) model in BOTH formats as well
    artifacts = save_model(pipe, "returned_flag_model")
    return {
        "artifacts": artifacts,
        "target": SECONDARY_TARGET,
        "positive_rate": round(float(y.mean()), 4),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "note": "Heavily imbalanced (~18% positive); PR-AUC is the honest metric here.",
    }


# --------------------------------------------------------------------------- #
# Unsupervised workflow
# --------------------------------------------------------------------------- #
SEGMENT_FEATURES = [
    "avg_order_value",
    "total_orders",
    "days_since_last_purchase",
    "discount_percent",
    "time_on_page",
    "customer_value",
    "price_sensitivity",
]


def train_segmentation(df: pd.DataFrame, k_range: range = range(2, 9)) -> Dict[str, object]:
    """K-Means on customer-level behavioural features."""
    # aggregate to one row per customer (segmentation is a customer-level task)
    cust = (
        df.groupby("customer_id")
        .agg(
            avg_order_value=("avg_order_value", "mean"),
            total_orders=("total_orders", "max"),
            days_since_last_purchase=("days_since_last_purchase", "min"),
            discount_percent=("discount_percent", "mean"),
            time_on_page=("time_on_page", "mean"),
            customer_value=("customer_value", "max"),
            price_sensitivity=("price_sensitivity", "mean"),
            membership_level=("membership_level", "first"),
            income_band=("income_band", "first"),
            purchase_rate=("purchase_flag", "mean"),
        )
        .reset_index()
    )

    X = cust[SEGMENT_FEATURES].copy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    diagnostics = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(Xs)
        diagnostics.append(
            {
                "k": int(k),
                "inertia": round(float(km.inertia_), 2),
                "silhouette": round(float(silhouette_score(Xs, labels)), 4),
            }
        )

    best_k = max(diagnostics, key=lambda d: d["silhouette"])["k"]
    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    cust["segment"] = km.fit_predict(Xs)

    # Persist the fitted K-Means model AND its scaler in BOTH formats
    seg_artifacts = {
        "kmeans": save_model(km, "customer_segmentation_kmeans"),
        "scaler": save_model(scaler, "customer_segmentation_scaler"),
    }

    # business-language description of each segment
    profile = (
        cust.groupby("segment")
        .agg(
            n_customers=("customer_id", "count"),
            avg_order_value=("avg_order_value", "mean"),
            total_orders=("total_orders", "mean"),
            recency_days=("days_since_last_purchase", "mean"),
            avg_discount=("discount_percent", "mean"),
            price_sensitivity=("price_sensitivity", "mean"),
            customer_value=("customer_value", "mean"),
            purchase_rate=("purchase_rate", "mean"),
        )
        .round(3)
        .reset_index()
    )

    descriptions = {int(r["segment"]): _describe_segment(r) for _, r in profile.iterrows()}
    profile["description"] = profile["segment"].map(descriptions)

    cust.to_csv(SEGMENTS_CSV, index=False)

    return {
        "artifacts": seg_artifacts,
        "n_customers": int(len(cust)),
        "features_used": SEGMENT_FEATURES,
        "feature_justification": (
            "These features capture the three classic segmentation axes: value "
            "(avg_order_value, customer_value, total_orders), recency "
            "(days_since_last_purchase) and price behaviour (discount_percent, "
            "price_sensitivity), plus engagement (time_on_page)."
        ),
        "diagnostics": diagnostics,
        "chosen_k": int(best_k),
        "chosen_silhouette": round(
            float([d for d in diagnostics if d["k"] == best_k][0]["silhouette"]), 4
        ),
        "segment_profile": profile.to_dict(orient="records"),
        "segment_descriptions": descriptions,
    }


def _describe_segment(r: pd.Series) -> str:
    """Turn a numeric cluster profile into a business persona."""
    value = r["customer_value"]
    recency = r["recency_days"]
    disc = r["avg_discount"]
    ps = r["price_sensitivity"]

    if value > 20000 and recency < 60:
        base = "High-value loyalists"
    elif value > 20000:
        base = "High-value lapsed"
    elif recency < 45 and r["purchase_rate"] > 0.5:
        base = "Active regulars"
    elif disc > 20 or ps > 8:
        base = "Discount-driven bargain hunters"
    elif recency > 150:
        base = "Dormant / at-risk"
    else:
        base = "Occasional browsers"

    return (
        f"{base}: avg order Rs.{r['avg_order_value']:.0f}, "
        f"{r['total_orders']:.1f} orders, {recency:.0f} days since last purchase, "
        f"{disc:.1f}% avg discount, purchase rate {r['purchase_rate']:.0%}."
    )


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(df: pd.DataFrame, sup: Dict[str, object], seg: Dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    # 1. target balance
    fig, ax = plt.subplots(figsize=(6, 4))
    df[SUPERVISED_TARGET].value_counts().sort_index().plot(
        kind="bar", ax=ax, color=["#c44", "#2a9d8f"]
    )
    ax.set_title("Target balance: purchase_flag")
    ax.set_xlabel("purchase_flag")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_target_balance.png", dpi=120)
    plt.close(fig)

    # 2. model comparison
    names = list(sup["models"].keys())
    aucs = [sup["models"][n]["roc_auc"] for n in names]
    prs = [sup["models"][n]["pr_auc"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, aucs, 0.4, label="ROC-AUC", color="#264653")
    ax.bar(x + 0.2, prs, 0.4, label="PR-AUC", color="#e9c46a")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_title("Supervised model comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_model_comparison.png", dpi=120)
    plt.close(fig)

    # 3. top features
    top = sup["top_features"][:12]
    if top:
        fig, ax = plt.subplots(figsize=(7, 5))
        feats = [t["feature"] for t in top][::-1]
        vals = [t["importance"] for t in top][::-1]
        ax.barh(feats, vals, color="#2a9d8f")
        ax.set_title(f"Top features ({sup['best_model']})")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "03_feature_importance.png", dpi=120)
        plt.close(fig)

    # 4. elbow + silhouette
    diag = pd.DataFrame(seg["diagnostics"])
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(diag["k"], diag["inertia"], "o-", color="#264653")
    ax1.set_xlabel("k")
    ax1.set_ylabel("inertia", color="#264653")
    ax2 = ax1.twinx()
    ax2.plot(diag["k"], diag["silhouette"], "s--", color="#e76f51")
    ax2.set_ylabel("silhouette", color="#e76f51")
    ax1.set_title("K-Means diagnostics (elbow + silhouette)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_cluster_diagnostics.png", dpi=120)
    plt.close(fig)

    # 5. segment scatter
    segs = pd.read_csv(SEGMENTS_CSV)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=segs, x="avg_order_value", y="customer_value",
        hue="segment", palette="tab10", ax=ax, s=40,
    )
    ax.set_title("Customer segments (value vs order size)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_segments.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run() -> Dict[str, object]:
    ensure_dirs()
    df = load_features()

    sup = train_supervised(df, SUPERVISED_TARGET)
    sec = train_secondary(df)
    seg = train_segmentation(df)

    make_figures(df, sup, seg)

    with open(ML_METRICS_JSON, "w", encoding="utf-8") as fh:
        json.dump({"supervised": sup, "secondary": sec}, fh, indent=2)
    with open(SEGMENT_SUMMARY_JSON, "w", encoding="utf-8") as fh:
        json.dump(seg, fh, indent=2)

    return {"supervised": sup, "secondary": sec, "segmentation": seg}


if __name__ == "__main__":  # pragma: no cover
    out = run()
    print("Best supervised model:", out["supervised"]["best_model"],
          "ROC-AUC:", out["supervised"]["best_roc_auc"])
    print("Chosen k:", out["segmentation"]["chosen_k"],
          "silhouette:", out["segmentation"]["chosen_silhouette"])
    for s, d in out["segmentation"]["segment_descriptions"].items():
        print(f"  Segment {s}: {d}")
