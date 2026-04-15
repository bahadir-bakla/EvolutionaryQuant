"""
ML Engine — LightGBM Trainer
==============================
Walk-forward cross-validation for Gold M1 scalper signal classifier.

Pipeline:
  1. Load M1 data (2019-2022 train, 2023-2024 OOS)
  2. Build features (ml_engine.features)
  3. Label with Triple Barrier (ml_engine.labels)
  4. Walk-forward: train on [t0, t0+train_months], test on [t0+train_months, t0+train_months+test_months]
  5. Aggregate OOS predictions -> final model trained on all data
  6. Save model to ml_engine/models/lgbm_gold.pkl

Binary classification: predict +1 vs rest
Signal: predict_proba >= threshold -> LONG entry
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

from ml_engine.features import build_features
from ml_engine.labels import triple_barrier_labels, barrier_label_stats

warnings.filterwarnings('ignore', category=UserWarning)


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

MODELS_DIR = Path(__file__).parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)

# Triple-barrier params
# Gold ATR@M1 ≈ 0.5 pts. SL=3 = 6×ATR, TP=6 = 12×ATR, 2:1 RR
# Entry spread is baked into labels (open of next bar ≈ close + 0.4)
TB_TP_PTS  = 6.0
TB_SL_PTS  = 3.0
TB_MAXBARS = 120   # 2h forward at M1
SPREAD_PTS = 0.4

# Walk-forward params (months)
TRAIN_MONTHS = 12
TEST_MONTHS  = 3

# LightGBM params
LGBM_PARAMS = {
    'objective':      'binary',
    'metric':         'auc',
    'num_leaves':     64,
    'learning_rate':  0.03,
    'n_estimators':   500,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq':   5,
    'min_child_samples': 50,
    'class_weight':   'balanced',
    'random_state':   42,
    'n_jobs':         -1,
    'verbose':        -1,
}

# Probability threshold to trigger signal
SIGNAL_THRESHOLD = 0.65


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def _prepare_dataset(
    m1: pd.DataFrame,
    tp_pts: float = TB_TP_PTS,
    sl_pts: float = TB_SL_PTS,
    max_bars: int = TB_MAXBARS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix X and binary label y (1 = win, 0 = loss/timeout)."""
    print("  Building features...", end=' ', flush=True)
    X = build_features(m1)
    print(f"shape={X.shape}")

    print("  Computing triple-barrier labels...", end=' ', flush=True)
    raw_labels = triple_barrier_labels(
        m1, tp_pts=tp_pts, sl_pts=sl_pts, max_bars=max_bars, spread_pts=SPREAD_PTS
    )
    stats = barrier_label_stats(raw_labels)
    print(
        f"total={stats['total']:,} | win={stats['win_rate']:.1%} "
        f"loss={stats['loss_rate']:.1%} timeout={stats['timeout_rate']:.1%}"
    )

    # Binary: 1 = TP hit (+1), 0 = everything else
    y = (raw_labels == 1).astype(int)
    y.name = 'label'

    # Align and drop NaNs
    mask = raw_labels.notna() & X.notna().all(axis=1)
    return X[mask], y[mask]


def _walk_forward_folds(
    index: pd.DatetimeIndex,
    train_months: int = TRAIN_MONTHS,
    test_months: int = TEST_MONTHS,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate (train_start, train_end, test_start, test_end) tuples."""
    start = index.min()
    end   = index.max()
    folds = []
    fold_start = start

    while True:
        train_end = fold_start + pd.DateOffset(months=train_months)
        test_end  = train_end  + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        folds.append((fold_start, train_end, train_end, test_end))
        fold_start = fold_start + pd.DateOffset(months=test_months)

    return folds


# -----------------------------------------------------------------------------
# WALK-FORWARD TRAINER
# -----------------------------------------------------------------------------

def walk_forward_train(
    m1_train: pd.DataFrame,
    m1_oos: pd.DataFrame | None = None,
    save_model: bool = True,
    model_name: str = 'lgbm_gold',
) -> dict:
    """
    Train LightGBM with walk-forward CV on training data,
    then evaluate on optional OOS (2023-2024).

    Returns dict with metrics and trained model.
    """
    print("\n" + "=" * 60)
    print("  ML Engine — Walk-Forward Training")
    print("=" * 60)

    # -- Prepare full training dataset ---------------------------------
    print("\n[1] Preparing training dataset...")
    X_all, y_all = _prepare_dataset(m1_train)

    # -- Walk-forward CV ------------------------------------------------
    print("\n[2] Walk-forward cross-validation...")
    folds = _walk_forward_folds(X_all.index)
    print(f"  Folds: {len(folds)}")

    oos_preds_all = pd.Series(dtype=float)
    oos_true_all  = pd.Series(dtype=int)

    for i, (tr_s, tr_e, ts_s, ts_e) in enumerate(folds):
        X_tr = X_all[(X_all.index >= tr_s) & (X_all.index < tr_e)]
        y_tr = y_all[(y_all.index >= tr_s) & (y_all.index < tr_e)]
        X_ts = X_all[(X_all.index >= ts_s) & (X_all.index < ts_e)]
        y_ts = y_all[(y_all.index >= ts_s) & (y_all.index < ts_e)]

        if len(X_tr) < 1000 or len(X_ts) < 100:
            continue

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_ts, y_ts)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        proba = model.predict_proba(X_ts)[:, 1]
        oos_preds_all = pd.concat([oos_preds_all, pd.Series(proba, index=X_ts.index)])
        oos_true_all  = pd.concat([oos_true_all,  pd.Series(y_ts.values, index=X_ts.index)])

        pred_binary = (proba >= SIGNAL_THRESHOLD).astype(int)
        prec = precision_score(y_ts, pred_binary, zero_division=0)
        rec  = recall_score(y_ts, pred_binary, zero_division=0)
        try:
            auc = roc_auc_score(y_ts, proba)
        except ValueError:
            auc = 0.0
        print(
            f"  Fold {i+1:02d} [{tr_s.date()} -> {ts_e.date()}] "
            f"AUC={auc:.3f}  Prec={prec:.3f}  Recall={rec:.3f}  "
            f"Signals={pred_binary.sum()}"
        )

    # -- Walk-forward CV summary ----------------------------------------
    if len(oos_preds_all) > 0:
        pred_b = (oos_preds_all >= SIGNAL_THRESHOLD).astype(int)
        try:
            cv_auc = roc_auc_score(oos_true_all, oos_preds_all)
        except ValueError:
            cv_auc = 0.0
        cv_prec  = precision_score(oos_true_all, pred_b, zero_division=0)
        cv_recall = recall_score(oos_true_all, pred_b, zero_division=0)
        print(f"\n  CV Summary: AUC={cv_auc:.3f}  Prec={cv_prec:.3f}  "
              f"Recall={cv_recall:.3f}  Total signals={pred_b.sum():,}")
    else:
        cv_auc = cv_prec = cv_recall = 0.0

    # -- Final model on all training data ------------------------------
    print("\n[3] Training final model on all training data...")
    final_model = lgb.LGBMClassifier(**LGBM_PARAMS)
    final_model.fit(X_all, y_all, callbacks=[lgb.log_evaluation(-1)])
    print(f"  Features: {len(X_all.columns)}")
    print(f"  Training bars: {len(X_all):,}")

    # -- OOS evaluation -------------------------------------------------
    oos_metrics = {}
    if m1_oos is not None:
        print("\n[4] OOS evaluation (2023-2024)...")
        X_oos, y_oos = _prepare_dataset(m1_oos)
        oos_proba  = final_model.predict_proba(X_oos)[:, 1]
        oos_binary = (oos_proba >= SIGNAL_THRESHOLD).astype(int)
        try:
            oos_auc = roc_auc_score(y_oos, oos_proba)
        except ValueError:
            oos_auc = 0.0
        oos_prec  = precision_score(y_oos, oos_binary, zero_division=0)
        oos_recall = recall_score(y_oos, oos_binary, zero_division=0)
        print(f"  OOS AUC={oos_auc:.3f}  Prec={oos_prec:.3f}  "
              f"Recall={oos_recall:.3f}  Signals={oos_binary.sum():,}")
        oos_metrics = {
            'oos_auc': oos_auc, 'oos_prec': oos_prec,
            'oos_recall': oos_recall, 'oos_signals': int(oos_binary.sum()),
        }

    # -- Feature importance top 20 --------------------------------------
    fi = pd.Series(
        final_model.feature_importances_,
        index=X_all.columns
    ).sort_values(ascending=False)
    print("\n  Top 20 Feature Importances:")
    for feat, score in fi.head(20).items():
        print(f"    {feat:<35} {score:>6.0f}")

    # -- Save model -----------------------------------------------------
    if save_model:
        model_path = MODELS_DIR / f'{model_name}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({'model': final_model, 'features': list(X_all.columns),
                         'threshold': SIGNAL_THRESHOLD}, f)
        print(f"\n  Model saved -> {model_path}")

    return {
        'model':       final_model,
        'features':    list(X_all.columns),
        'threshold':   SIGNAL_THRESHOLD,
        'cv_auc':      cv_auc,
        'cv_prec':     cv_prec,
        'cv_recall':   cv_recall,
        'feature_importance': fi,
        **oos_metrics,
    }


def load_model(model_name: str = 'lgbm_gold') -> dict:
    """Load a saved model bundle."""
    model_path = MODELS_DIR / f'{model_name}.pkl'
    with open(model_path, 'rb') as f:
        return pickle.load(f)
