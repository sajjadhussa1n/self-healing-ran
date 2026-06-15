# src/detection/cod_classifier.py
"""
Cell Outage Detection Classifiers
===================================
Three-class COD: Normal (0) / Outage (1) / Degraded (2).

Classifiers:
  ThresholdCOD : Rule-based, no training required
  MLCOD        : Random Forest or Logistic Regression
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report,
                              confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

from src.config import CFG

# ── Feature set ───────────────────────────────────────────────────────
# All features are E2-observable PM counters.
# Excluded: own-cell SINR/throughput (collapse to 0 on outage),
#           outage_ues flag (circular), tx_power (directly reveals
#           failure — not always available in real networks).
COD_FEATURES = [
    'ue_count', 'prb_load', 'ue_ratio',
    'delta_ue_count', 'delta_prb_load',
    'n1_ue_count', 'n1_delta_ue',
    'n1_mean_sinr', 'n1_prb_load',
    'n2_ue_count', 'n2_delta_ue',
    'n2_mean_sinr', 'n2_prb_load',
    'n3_ue_count', 'n3_delta_ue',
    'n3_mean_sinr', 'n3_prb_load',
]

CLASS_NAMES = ['Normal', 'Outage', 'Degraded']


class ThresholdCOD:
    """
    Rule-based three-class COD detector.

    Label 1 (Outage):
      Current ue_count near zero AND
      previous ue_count was meaningful AND
      at least one neighbour gained UEs

    Label 2 (Degraded):
      Own UE count increased AND
      a neighbour simultaneously lost UEs

    Label 0 (Normal): all other cases.

    Mirrors 3GPP TR 36.902 sleeping cell detection.
    """

    def __init__(self,
                 ue_drop_ratio      : float = 0.80,
                 min_ue_threshold   : int   = 2,
                 empty_ue_threshold : int   = 1,
                 nb_gain_threshold  : int   = 2):
        self.ue_drop_ratio      = ue_drop_ratio
        self.min_ue_threshold   = min_ue_threshold
        self.empty_ue_threshold = empty_ue_threshold
        self.nb_gain_threshold  = nb_gain_threshold

    def predict(self, df: pd.DataFrame) -> pd.Series:
        preds = np.zeros(len(df), dtype=int)
        for i, row in df.iterrows():
            idx     = df.index.get_loc(i)
            prev_ue = (row['ue_count'] -
                       row['delta_ue_count'])

            # Transition signal
            if prev_ue >= self.min_ue_threshold:
                ue_drop    = (-row['delta_ue_count'] /
                               max(prev_ue, 1))
                large_drop = ue_drop >= self.ue_drop_ratio
            else:
                large_drop = False

            # State signal (sustained outage)
            state_empty = (row['ue_count'] <=
                           self.empty_ue_threshold)

            # Neighbour confirmation
            nb_gained = (
                row['n1_delta_ue'] >=
                self.nb_gain_threshold or
                row['n2_delta_ue'] >=
                self.nb_gain_threshold or
                row['n3_delta_ue'] >=
                self.nb_gain_threshold)
            nb_elevated = (
                row['n1_ue_count'] >=
                self.min_ue_threshold * 1.3 or
                row['n2_ue_count'] >=
                self.min_ue_threshold * 1.3 or
                row['n3_ue_count'] >=
                self.min_ue_threshold * 1.3)

            outage_signal = (
                (large_drop and nb_gained) or
                (state_empty and
                 (nb_elevated or nb_gained)))

            if outage_signal:
                preds[idx] = 1
                continue

            # Degradation detection
            nb_lost = (
                row['n1_delta_ue'] <=
                -self.nb_gain_threshold or
                row['n2_delta_ue'] <=
                -self.nb_gain_threshold or
                row['n3_delta_ue'] <=
                -self.nb_gain_threshold)
            own_gained = row['delta_ue_count'] >= 2

            if nb_lost and own_gained:
                preds[idx] = 2
                continue

            preds[idx] = 0

        return pd.Series(preds, index=df.index)

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        preds  = self.predict(df)
        labels = df['label']
        print("\n📊 Threshold COD — Evaluation:")
        print(classification_report(
            labels, preds,
            target_names=CLASS_NAMES, digits=3))
        cm = confusion_matrix(
            labels, preds, labels=[0, 1, 2])
        print("   Confusion matrix (rows=true, cols=pred):")
        print(f"   {'':12s} "
              f"{'Normal':>8s} "
              f"{'Outage':>8s} "
              f"{'Degraded':>10s}")
        for r, name in enumerate(CLASS_NAMES):
            print(f"   {name:12s} "
                  f"{cm[r,0]:8d} "
                  f"{cm[r,1]:8d} "
                  f"{cm[r,2]:10d}")
        return preds


class MLCOD:
    """
    ML-based three-class COD detector.
    Supports Random Forest and Logistic Regression.
    """

    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.scaler     = StandardScaler()
        self.is_fitted  = False

        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators = 100,
                max_depth    = 10,
                class_weight = 'balanced',
                random_state = CFG.RANDOM_SEED,
                n_jobs       = -1)
        else:
            self.model = LogisticRegression(
                class_weight = 'balanced',
                max_iter     = 1000,
                multi_class  = 'multinomial',
                random_state = CFG.RANDOM_SEED)

    def prepare_features(self, df: pd.DataFrame):
        available = [f for f in COD_FEATURES
                     if f in df.columns]
        return df[available].fillna(0).values, available

    def fit(self, df_train: pd.DataFrame) -> None:
        X, feats = self.prepare_features(df_train)
        y        = df_train['label'].values
        if self.model_type == 'logistic_regression':
            X = self.scaler.fit_transform(X)
        else:
            self.scaler.fit(X)
        self.model.fit(X, y)
        self.features_ = feats
        self.is_fitted  = True
        n0, n1, n2 = ((y==0).sum(),
                       (y==1).sum(),
                       (y==2).sum())
        print(f"✅ {self.model_type} trained: "
              f"Normal={n0:,} | "
              f"Outage={n1:,} | "
              f"Degraded={n2:,}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = self.prepare_features(df)
        if self.model_type == 'logistic_regression':
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = self.prepare_features(df)
        if self.model_type == 'logistic_regression':
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)

    def evaluate(self, df_test: pd.DataFrame,
                  label: str = 'Test') -> tuple:
        preds = self.predict(df_test)
        proba = self.predict_proba(df_test)
        y     = df_test['label'].values
        print(f"\n📊 {self.model_type} COD — {label}:")
        print(classification_report(
            y, preds,
            target_names=CLASS_NAMES, digits=3))
        cm = confusion_matrix(y, preds, labels=[0, 1, 2])
        print("   Confusion matrix (rows=true, cols=pred):")
        print(f"   {'':12s} "
              f"{'Normal':>8s} "
              f"{'Outage':>8s} "
              f"{'Degraded':>10s}")
        for r, name in enumerate(CLASS_NAMES):
            print(f"   {name:12s} "
                  f"{cm[r,0]:8d} "
                  f"{cm[r,1]:8d} "
                  f"{cm[r,2]:10d}")

        if (self.model_type == 'random_forest' and
                hasattr(self.model,
                        'feature_importances_')):
            imps = self.model.feature_importances_
            fi   = sorted(zip(self.features_, imps),
                          key=lambda x: x[1],
                          reverse=True)
            print(f"\n   Top 10 features:")
            for fname, fimp in fi[:10]:
                bar = '█' * int(fimp * 60)
                print(f"   {fname:30s} "
                      f"{fimp:.4f} {bar}")

        return preds, proba
