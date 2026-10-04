"""
Reproducible LSTM baseline (Section 4, Tier 1 / 92 features), with all
random seeds fixed (Python, NumPy, TensorFlow) and TensorFlow op-level
determinism enabled, replacing the paper's earlier unfixed-seed result.
The 92 engineered features are fed to the LSTM as an artificial
length-92 sequence of scalars (not a true multi-timestep sequence).

Usage:
    python lstm_fixed_seed.py energy_features_master_labeled.csv
"""
import os
os.environ['PYTHONHASHSEED'] = '42'
import random
random.seed(42)
import numpy as np
np.random.seed(42)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (recall_score, precision_score, f1_score, roc_auc_score,
                              average_precision_score, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("determinism note:", e)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import sys
features = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else 'energy_features_master_labeled.csv')
X = features.drop(['household_id', 'energy_poor', 'vulnerability_score'], axis=1)
y = features['energy_poor']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

model = keras.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
    layers.Dropout(0.2),
    layers.LSTM(32),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Recall(name='recall'),
             keras.metrics.Precision(name='precision'), keras.metrics.AUC(name='auc')]
)

early_stop = EarlyStopping(monitor='val_recall', patience=10, mode='max', restore_best_weights=True, verbose=0)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=0)

history = model.fit(
    X_train_lstm, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=2,
    shuffle=True,
)

y_pred_proba = model.predict(X_test_lstm, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)
n_epochs_trained = len(history.history['loss'])

print("\n=== FIXED-SEED LSTM RESULTS (Tier 1, 92 features, artificial-sequence input) ===")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print(f"Confusion matrix (tn,fp,fn,tp): {cm[0,0]},{cm[0,1]},{cm[1,0]},{cm[1,1]}")
print(f"Epochs trained before stopping: {n_epochs_trained}")

pd.DataFrame([{
    'recall': recall, 'precision': precision, 'f1': f1, 'roc_auc': roc_auc, 'pr_auc': pr_auc,
    'tn': cm[0,0], 'fp': cm[0,1], 'fn': cm[1,0], 'tp': cm[1,1], 'epochs_trained': n_epochs_trained,
    'seed': 42
}]).to_csv('lstm_fixed_seed_results.csv', index=False)
