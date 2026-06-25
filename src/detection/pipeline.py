"""
High-level COD training pipeline: generates the labelled
KPI dataset, trains the Threshold / Logistic Regression /
Random Forest detectors, and reports results.
"""
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
                              accuracy_score, f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config import SimConfig
from src.detection.kpi_logger import KPILogger
from src.detection.classifiers import (
    ThresholdClassifier, COD_FEATURE_NAMES)


def _generate_cod_dataset(config, n_episodes,
                          n_normal_steps, n_outage_steps,
                          verbose=True):
    """Internal: run KPILogger over many episodes
    to build a labelled DataFrame."""
    logger = KPILogger(config)
    for ep in range(n_episodes):
        logger.run_episode(
            n_normal_steps=n_normal_steps,
            n_outage_steps=n_outage_steps,
            episode_id=ep)
        if verbose and (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"  [COD dataset] episode "
                 f"{ep + 1}/{n_episodes}")
    return logger.get_dataframe()


def train_cod_model(config=None, n_episodes=60,
                    n_normal_steps=10, n_outage_steps=10,
                    test_size=0.25, save_dir="models",
                    verbose=True):
    """
    Full COD training pipeline: generate dataset, train
    Threshold / Logistic Regression / Random Forest
    classifiers, evaluate all three, and save the trained
    ML models to disk.

    Parameters
    ----------
    config : SimConfig or None
    n_episodes : int
        Number of simulated episodes for dataset generation.
    n_normal_steps, n_outage_steps : int
        Timesteps of normal/outage operation per episode.
    test_size : float
        Fraction of data held out for testing.
    save_dir : str
        Directory to save trained .pkl model files.
    verbose : bool

    Returns
    -------
    results : dict
        {
          'dataset': pd.DataFrame,
          'models': {'threshold': ..., 'logreg': ...,
                     'rf': ...},
          'reports': {'threshold': str, 'logreg': str,
                      'rf': str},
          'scores': {'threshold': {...}, 'logreg': {...},
                     'rf': {...}}
        }
    """
    cfg = config if config is not None else SimConfig()
    os.makedirs(save_dir, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("COD TRAINING PIPELINE")
        print("=" * 60)
        print(f"Generating dataset: {n_episodes} episodes "
             f"({n_normal_steps} normal + "
             f"{n_outage_steps} outage steps each)")

    df = _generate_cod_dataset(
        cfg, n_episodes, n_normal_steps, n_outage_steps,
        verbose=verbose)

    X = df[COD_FEATURE_NAMES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y,
        random_state=cfg.RANDOM_SEED)

    models, reports, scores = {}, {}, {}

    # --- 1. Threshold (rule-based) classifier ---
    if verbose:
        print("\n--- Threshold Classifier ---")
    thresh = ThresholdClassifier()
    y_pred_thresh = thresh.predict(X_test)
    reports["threshold"] = classification_report(
        y_test, y_pred_thresh, zero_division=0)
    scores["threshold"] = {
        "accuracy": accuracy_score(y_test, y_pred_thresh),
        "f1_macro": f1_score(y_test, y_pred_thresh,
                             average="macro",
                             zero_division=0)}
    models["threshold"] = thresh
    if verbose:
        print(reports["threshold"])
        print(f"Accuracy: {scores['threshold']['accuracy']:.3f}"
             f" | F1(macro): "
             f"{scores['threshold']['f1_macro']:.3f}")

    # --- 2. Logistic Regression ---
    if verbose:
        print("\n--- Logistic Regression ---")
    logreg = LogisticRegression(max_iter=1000,
                                multi_class="auto")
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)
    reports["logreg"] = classification_report(
        y_test, y_pred_lr, zero_division=0)
    scores["logreg"] = {
        "accuracy": accuracy_score(y_test, y_pred_lr),
        "f1_macro": f1_score(y_test, y_pred_lr,
                             average="macro",
                             zero_division=0)}
    models["logreg"] = logreg
    if verbose:
        print(reports["logreg"])
        print(f"Accuracy: {scores['logreg']['accuracy']:.3f}"
             f" | F1(macro): "
             f"{scores['logreg']['f1_macro']:.3f}")

    # --- 3. Random Forest ---
    if verbose:
        print("\n--- Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=10,
        random_state=cfg.RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    reports["rf"] = classification_report(
        y_test, y_pred_rf, zero_division=0)
    scores["rf"] = {
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "f1_macro": f1_score(y_test, y_pred_rf,
                             average="macro",
                             zero_division=0)}
    models["rf"] = rf
    if verbose:
        print(reports["rf"])
        print(f"Accuracy: {scores['rf']['accuracy']:.3f}"
             f" | F1(macro): {scores['rf']['f1_macro']:.3f}")

    # Save ML models (threshold model has no parameters
    # to persist)
    joblib.dump(logreg, os.path.join(
        save_dir, "cod_logreg.pkl"))
    joblib.dump(rf, os.path.join(save_dir, "cod_rf.pkl"))
    df.to_csv(os.path.join(
        "results", "cod_dataset.csv"), index=False)

    if verbose:
        print(f"\nSaved models to '{save_dir}/' and "
             f"dataset to 'results/cod_dataset.csv'")

    return {"dataset": df, "models": models,
            "reports": reports, "scores": scores}
