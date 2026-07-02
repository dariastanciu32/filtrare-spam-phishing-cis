# =============================================================
# TEST_TOTAL.PY — VERSIUNEA FINALA
# =============================================================
# Scopul acestui fisier:
#   Evaluarea finala a tuturor celor 5 modele antrenate
#   pe test set-ul SIGILAT (8.610 emailuri nevazute).
#
#   Modele evaluate:
#   1. Linear-SGD  (calibrat)
#   2. SVM-SGD     (calibrat)
#   3. Gradient Boosting
#   4. BiLSTM
#   5. CNN1D
#
#   Metrici calculate:
#   - Accuracy, Log Loss
#   - Precision, Recall, F1-Score
#   - AUC-ROC
#   - Confusion Matrix
#
#   Grafice generate:
#   - Tabel comparativ toate metricile
#   - Matrici de confuzie (toate 5 modele)
#   - Comparativ Accuracy pe test
#   - Comparativ Loss pe test
#   - Comparativ F1-Score pe test
#   - Curbe ROC suprapuse
#   - Comparativ complet ML vs DL pe test
#
# REGULI CRITICE:
#   1. Test set-ul e deschis O SINGURA DATA
#   2. Niciun fit pe test — doar transform
#   3. Nu modificam nimic dupa ce vedem rezultatele
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import scipy.sparse as sp
import joblib
import os
import time  # pentru masurarea timpului de inferenta
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import (
    accuracy_score,
    log_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
from sklearn.utils import resample  # pentru bootstrap intervale de incredere
from sklearn.base import BaseEstimator, ClassifierMixin  # pentru clasa CalibratorPrefit
from sklearn.isotonic import IsotonicRegression  # pentru calibrare isotonica

# ---------------------------------------------------------------
# STILUL GLOBAL AL GRAFICELOR
# Identic cu train_ml_total.py si train_dl_total.py
# ---------------------------------------------------------------
CULORI = {
    'Linear-SGD':        '#0F766E',
    'SVM-SGD':           '#4338CA',
    'Gradient Boosting': '#C2410C',
    'BiLSTM':            '#0891B2',
    'CNN1D':             '#7C3AED',
}

BG       = '#F8FAFC'
AX_BG    = '#FFFFFF'
TEXT     = '#1E293B'
GRID_COL = '#E2E8F0'
COL_VAL  = '#DC2626'

plt.rcParams.update({
    'figure.facecolor':   BG,
    'figure.dpi':         120,
    'axes.facecolor':     AX_BG,
    'axes.grid':          True,
    'grid.color':         GRID_COL,
    'grid.alpha':         0.55,
    'grid.linestyle':     '-',
    'grid.linewidth':     0.6,
    'font.family':        'sans-serif',
    'font.size':          11,
    'axes.titlesize':     14,
    'axes.titleweight':   'semibold',
    'axes.labelsize':     11,
    'axes.labelcolor':    TEXT,
    'axes.titlecolor':    TEXT,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.edgecolor':     '#CBD5E1',
    'xtick.color':        TEXT,
    'ytick.color':        TEXT,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.framealpha':  0.96,
    'legend.edgecolor':   '#CBD5E1',
    'legend.facecolor':   '#FFFFFF',
    'legend.fontsize':    9,
    'savefig.facecolor':  BG,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.3,
    'savefig.dpi':        300,
    'axes.unicode_minus': False,
})

NUMERIC_COLS = [
    'nr_url_total', 'nr_url_suspect', 'nr_url_scurtat', 'are_url',
    'lung_email', 'nr_cuvinte', 'nr_exclamare',
    'nr_simboluri_ban', 'pct_majuscule', 'nr_trigger_spam'
]

MAX_LEN = 200  # Identic cu train_dl_total.py

os.makedirs('grafice', exist_ok=True)
os.makedirs('results', exist_ok=True)  # pentru salvarea rezultatelor CSV

# ---------------------------------------------------------------
# CLASA CALIBRATORPREFIT
# Necesara pentru incarcarea modelelor ML calibrate
# (Linear-SGD si SVM-SGD folosesc calibrare izotonica)
# ---------------------------------------------------------------
class CalibratorPrefit(BaseEstimator, ClassifierMixin):
    """
    Calibrare isotonică pe un clasificator deja antrenat.
    Înlocuiește CalibratedClassifierCV(cv='prefit'), care nu mai e acceptat
    în unele versiuni sklearn.
    """

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def _proba_clasa_pozitiva(self, X):
        if hasattr(self.base_estimator, 'predict_proba'):
            p = self.base_estimator.predict_proba(X)
            return np.asarray(p[:, 1], dtype=np.float64)
        z = np.clip(self.base_estimator.decision_function(X), -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.array([0, 1])
        p1 = self._proba_clasa_pozitiva(X)
        self.calibrator_ = IsotonicRegression(out_of_bounds='clip')
        try:
            self.calibrator_.fit(
                p1, y.astype(np.float64), 
                sample_weight=sample_weight
            )
        except TypeError:
            self.calibrator_.fit(p1, y.astype(np.float64))
        return self

    def predict_proba(self, X):
        p1 = self._proba_clasa_pozitiva(X)
        p_cal = np.clip(self.calibrator_.predict(p1), 0.0, 1.0)
        return np.column_stack([1.0 - p_cal, p_cal])

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

print("=" * 65)
print("TEST_TOTAL.PY — Evaluare finala pe test set sigilat")
print("=" * 65)

# ---------------------------------------------------------------
# PASUL 1: Incarcam datele de test
#
# Doua formate:
#   ML: X_test_sparse.npz (matrice TF-IDF + features numerice)
#   DL: test_data.csv (text procesat + features numerice)
#
# y_test.npy: etichetele reale (0=ham, 1=spam)
# ---------------------------------------------------------------
print("\n" + "="*60)
print("PASUL 1: Incarcare date test")
print("="*60)

# Date pentru ML
X_test_sparse = sp.load_npz('data/test/X_test_sparse.npz')
# Matricea TF-IDF + features numerice pentru modele ML
# Shape: (8610, 15010) — 8610 emailuri, 15010 features

y_test = np.load('data/test/y_test.npy')
# Vector cu etichetele reale: 0=ham, 1=spam
# Shape: (8610,)

# Date pentru DL
df_test = pd.read_csv('data/test/test_data.csv')
df_test['text_procesat'] = df_test['text_procesat'].fillna('')
# Curatam valorile lipsa din textul procesat

X_text_test = df_test['text_procesat'].values
# Array cu textul procesat al fiecarui email (string)

X_num_test = df_test[NUMERIC_COLS].values.astype(np.float32)
# Matricea celor 10 features numerice pentru DL
# Shape: (8610, 10)

print(f"Test set: {len(y_test):,} emailuri")
print(f"  Spam (1): {y_test.sum():,} ({y_test.mean()*100:.1f}%)")
print(f"  Ham  (0): {(y_test==0).sum():,} ({(1-y_test.mean())*100:.1f}%)")
print(f"  Features ML:  {X_test_sparse.shape[1]:,}")
print(f"  Features DL:  text ({MAX_LEN} tokens) + {len(NUMERIC_COLS)} numerice")

# ---------------------------------------------------------------
# PASUL 2: Incarcam modelele si artefactele
#
# ML: modele calibrate pentru Linear-SGD si SVM
#     model nativ pentru GB (produce probabilitati direct)
# DL: modele Keras in format .keras
# Scalere: unul pentru ML (15010 features), unul pentru DL (10 features)
# Tokenizer: pentru transformarea textului in secvente DL
# ---------------------------------------------------------------
print("\n" + "="*60)
print("PASUL 2: Incarcare modele si artefacte")
print("="*60)

# Scalere
feature_scaler    = joblib.load('models/feature_scaler.pkl')
# MaxAbsScaler antrenat pe 15010 features (TF-IDF + numerice)
# Folosit pentru normalizarea datelor de test ML

dl_numeric_scaler = joblib.load('models/dl_numeric_scaler.pkl')
# MaxAbsScaler antrenat pe 10 features numerice
# Folosit pentru normalizarea datelor de test DL

# Tokenizer DL
tokenizer = joblib.load('models/dl_tokenizer.pkl')
# Keras Tokenizer antrenat pe textul de train
# Folosit pentru transformarea textului test in secvente de indici

# Modele ML
linear_sgd = joblib.load('models/ml_linear_sgd_calibrated.pkl')
# Linear-SGD cu calibrare izotonica — produce probabilitati calibrate
# Folosim versiunea calibrata pentru log loss corect

svm_model = joblib.load('models/ml_svm_calibrated.pkl')
# SVM-SGD cu calibrare izotonica — identic cu Linear-SGD

gb_model = joblib.load('models/ml_gb.pkl')
# Gradient Boosting — produce probabilitati direct, fara calibrare separata

print("Modele ML incarcate: Linear-SGD, SVM-SGD, Gradient Boosting")

# Modele DL
bilstm_model = tf.keras.models.load_model('models/dl_bilstm.keras')
# BiLSTM Bidirectional — incarcam arhitectura + greutatile antrenate

cnn1d_model = tf.keras.models.load_model('models/dl_cnn1d.keras')
# CNN 1D — identic cu BiLSTM

print("Modele DL incarcate: BiLSTM, CNN1D")

# ---------------------------------------------------------------
# PASUL 3: Pregatirea datelor de test
#
# REGULA CRITICA: transform(), NU fit()
# fit() = inveti maximele/vocabularul din date
# transform() = aplici transformarile invatate din TRAIN
#
# Daca am face fit() pe test:
#   - ML: scaler ar invata alte maxime → scale diferite → comparatie incorecta
#   - DL: tokenizer ar invata cuvinte noi → indici diferiti → date inconsistente
# ---------------------------------------------------------------
print("\n" + "="*60)
print("PASUL 3: Pregatire date test")
print("="*60)

# Preprocesare pentru ML
X_test_sc = feature_scaler.transform(X_test_sparse)
# Aplicam aceleasi maxime invatate din train
# Nu fit() — nu invatam nimic nou din test
print("ML: matrice test normalizata cu feature_scaler.pkl")

# Preprocesare pentru DL — text
seq_test   = tokenizer.texts_to_sequences(X_text_test)
# Transformam textul in secvente de indici folosind vocabularul din train
# Cuvintele necunoscute (nu apar in train) → indexul <OOV>

X_seq_test = pad_sequences(
    seq_test,
    maxlen=MAX_LEN,       # 200 tokens per email
    padding='post',       # zerouri la sfarsit
    truncating='post'     # trunchiem la sfarsit daca e prea lung
)
# Shape: (8610, 200) — 8610 emailuri, 200 tokens fiecare
print(f"DL text: secvente tokenizate si padded → shape {X_seq_test.shape}")

# Preprocesare pentru DL — features numerice
X_num_test_sc = dl_numeric_scaler.transform(X_num_test).astype(np.float32)
# Aplicam aceleasi maxime invatate din train DL
print(f"DL numeric: {X_num_test_sc.shape[1]} features normalizate")

# ---------------------------------------------------------------
# PASUL 4: Predictiile tuturor modelelor
#
# Fiecare model produce:
#   proba_* = probabilitati spam ∈ (0, 1)
#   pred_*  = predictii binare (0=ham, 1=spam) cu prag 0.5
#
# Pragul 0.5 e standard pentru clasificare binara echilibrata.
# Dataset-ul nostru e ~50% spam / 50% ham → prag 0.5 e optim.
#
# MĂSURĂM TIMPUL DE INFERENȚĂ pentru fiecare model:
#   Important pentru aplicații real-time și alegerea modelului în producție.
#
# WARM-UP pentru modele DL:
#   TensorFlow face lazy initialization la prima predicție.
#   Facem un warm-up cu 32 emailuri pentru a elimina overhead-ul
#   de inițializare și a măsura doar timpul real de inferență.
# ---------------------------------------------------------------
print("\n" + "="*60)
print("PASUL 4: Predictii pe test set")
print("="*60)

# Warm-up pentru modele DL — elimina overhead-ul de initializare TensorFlow
print("Warm-up modele DL (eliminare overhead initializare)...", end=' ')
_ = bilstm_model.predict([X_seq_test[:32], X_num_test_sc[:32]], verbose=0)
_ = cnn1d_model.predict([X_seq_test[:32], X_num_test_sc[:32]], verbose=0)
print("OK")

# Dictionar pentru timpii de inferenta
timpi_inferenta = {}

# Linear-SGD — predict_proba direct (model calibrat)
start_time = time.perf_counter()
proba_linear = linear_sgd.predict_proba(X_test_sc)[:, 1]
timpi_inferenta['Linear-SGD'] = time.perf_counter() - start_time
# [:, 1] = luam coloana pentru clasa 1 (spam)
# Coloana 0 = probabilitate ham, coloana 1 = probabilitate spam
pred_linear  = (proba_linear >= 0.5).astype(int)
print(f"Linear-SGD: predictii calculate ({timpi_inferenta['Linear-SGD']:.3f}s)")

# SVM-SGD — identic cu Linear-SGD
start_time = time.perf_counter()
proba_svm = svm_model.predict_proba(X_test_sc)[:, 1]
timpi_inferenta['SVM-SGD'] = time.perf_counter() - start_time
pred_svm  = (proba_svm >= 0.5).astype(int)
print(f"SVM-SGD: predictii calculate ({timpi_inferenta['SVM-SGD']:.3f}s)")

# Gradient Boosting
start_time = time.perf_counter()
proba_gb = gb_model.predict_proba(X_test_sc)[:, 1]
timpi_inferenta['Gradient Boosting'] = time.perf_counter() - start_time
pred_gb  = (proba_gb >= 0.5).astype(int)
print(f"Gradient Boosting: predictii calculate ({timpi_inferenta['Gradient Boosting']:.3f}s)")

# BiLSTM — predict cu doua inputuri (text + numeric)
start_time = time.perf_counter()
proba_bilstm = bilstm_model.predict(
    [X_seq_test, X_num_test_sc],
    verbose=0   # nu afisam progress bar
).flatten()
timpi_inferenta['BiLSTM'] = time.perf_counter() - start_time
# .flatten() = transforma (8610, 1) in (8610,)
pred_bilstm = (proba_bilstm >= 0.5).astype(int)
print(f"BiLSTM: predictii calculate ({timpi_inferenta['BiLSTM']:.3f}s)")

# CNN1D — identic cu BiLSTM
start_time = time.perf_counter()
proba_cnn1d = cnn1d_model.predict(
    [X_seq_test, X_num_test_sc],
    verbose=0
).flatten()
timpi_inferenta['CNN1D'] = time.perf_counter() - start_time
pred_cnn1d = (proba_cnn1d >= 0.5).astype(int)
print(f"CNN1D: predictii calculate ({timpi_inferenta['CNN1D']:.3f}s)")

# Afisam sumar timpi de inferenta
print(f"\nTimpi de inferenta pentru {len(y_test):,} emailuri:")
for model_name, timp in timpi_inferenta.items():
    emails_per_sec = len(y_test) / timp if timp > 0 else 0
    print(f"  {model_name:<22} {timp:>6.3f}s  ({emails_per_sec:>7.0f} emailuri/s)")

# ---------------------------------------------------------------
# PASUL 5: Calculul tuturor metricilor
#
# Accuracy:  (TN + TP) / total
# Log Loss:  -mean[y*log(p) + (1-y)*log(1-p)]
# Precision: TP / (TP + FP) — cat de putine alarme false
# Recall:    TP / (TP + FN) — cat spam am detectat
# F1-Score:  2 * (Prec * Recall) / (Prec + Recall)
# AUC-ROC:   aria sub curba ROC — independent de pragul 0.5
# ---------------------------------------------------------------
print("\n" + "="*60)
print("PASUL 5: Calcul metrici finale")
print("="*60)

# Organizam toate modelele intr-o lista pentru iterare
modele_test = [
    ('Linear-SGD',        proba_linear, pred_linear),
    ('SVM-SGD',           proba_svm,    pred_svm),
    ('Gradient Boosting', proba_gb,     pred_gb),
    ('BiLSTM',            proba_bilstm, pred_bilstm),
    ('CNN1D',             proba_cnn1d,  pred_cnn1d),
]

rezultate = {}
# Dictionar care va contine toate metricile pentru fiecare model

for nume, proba, pred in modele_test:
    # Clipuim probabilitatile pentru log_loss numeric stabil
    # Identic cu ce am facut in train — previne log(0) = -inf
    proba_clip = np.clip(proba, 1e-7, 1 - 1e-7)

    # Calculam curba ROC
    # fpr = False Positive Rate (alarme false la fiecare prag)
    # tpr = True Positive Rate (spam detectat la fiecare prag)
    # thresholds = pragurile de decizie (de la 0 la 1)
    fpr, tpr, thresholds = roc_curve(y_test, proba)
    auc_score = auc(fpr, tpr)
    # auc() = aria sub curba ROC
    # 0.5 = model aleatoriu, 1.0 = model perfect

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    # [[TN, FP],
    #  [FN, TP]]

    rezultate[nume] = {
        'accuracy':  accuracy_score(y_test, pred),
        'loss':      log_loss(y_test, proba_clip, labels=[0, 1]),
        'precision': precision_score(y_test, pred, zero_division=0),
        # zero_division=0 → returneaza 0 daca nu exista predictii pozitive
        # (nu se va intampla pe datele noastre, dar e protectie)
        'recall':    recall_score(y_test, pred, zero_division=0),
        'f1':        f1_score(y_test, pred, zero_division=0),
        'auc':       auc_score,
        'cm':        cm,
        'fpr':       fpr,
        'tpr':       tpr,
        'proba':     proba,
        'pred':      pred,
        'culoare':   CULORI[nume],
        'timp_inferenta': timpi_inferenta[nume],  # adaugam timpul de inferenta
    }

    # Extragem TN, FP, FN, TP din matricea de confuzie
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{nume}:")
    print(f"  Accuracy:  {rezultate[nume]['accuracy']*100:.2f}%")
    print(f"  Loss:      {rezultate[nume]['loss']:.4f}")
    print(f"  Precision: {rezultate[nume]['precision']*100:.2f}%")
    print(f"  Recall:    {rezultate[nume]['recall']*100:.2f}%")
    print(f"  F1-Score:  {rezultate[nume]['f1']*100:.2f}%")
    print(f"  AUC-ROC:   {rezultate[nume]['auc']:.4f}")
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

names = list(rezultate.keys())
# Lista ordonata a numelor modelelor pentru grafice

# ---------------------------------------------------------------
# Salvam rezultatele in CSV pentru documentare si analiza
# ---------------------------------------------------------------
print("\nSalvare rezultate in CSV...")
df_rezultate = pd.DataFrame({
    'Model': names,
    'Accuracy': [rezultate[nm]['accuracy'] for nm in names],
    'Log_Loss': [rezultate[nm]['loss'] for nm in names],
    'Precision': [rezultate[nm]['precision'] for nm in names],
    'Recall': [rezultate[nm]['recall'] for nm in names],
    'F1_Score': [rezultate[nm]['f1'] for nm in names],
    'AUC_ROC': [rezultate[nm]['auc'] for nm in names],
    'TN': [rezultate[nm]['cm'].ravel()[0] for nm in names],  # True Negatives
    'FP': [rezultate[nm]['cm'].ravel()[1] for nm in names],  # False Positives
    'FN': [rezultate[nm]['cm'].ravel()[2] for nm in names],  # False Negatives (spam scapat - CRITIC!)
    'TP': [rezultate[nm]['cm'].ravel()[3] for nm in names],  # True Positives
    'Timp_Inferenta_sec': [rezultate[nm]['timp_inferenta'] for nm in names],
    'Timp_mediu_ms_per_email': [
        (rezultate[nm]['timp_inferenta'] / len(y_test)) * 1000
        if rezultate[nm]['timp_inferenta'] > 0 else 0
        for nm in names
    ],
    'Emailuri_per_sec': [len(y_test) / rezultate[nm]['timp_inferenta'] 
                         if rezultate[nm]['timp_inferenta'] > 0 else 0 
                         for nm in names],
})
df_rezultate.to_csv('results/test_final_results.csv', index=False, float_format='%.6f')
print("Salvat: results/test_final_results.csv")

# ---------------------------------------------------------------
# INTERVALE DE INCREDERE (Bootstrap) pentru TOATE modelele
# ---------------------------------------------------------------
print("\n" + "="*60)
print("CALCUL INTERVALE DE INCREDERE (Bootstrap)")
print("="*60)
print(f"Calculam intervale de incredere pentru toate cele {len(names)} modele")
print(f"Bootstrap cu 1000 iteratii per model...")

# Dictionar pentru a stoca toate intervalele de incredere
all_confidence_intervals = {
    'Model': [],
    'Accuracy': [],
    'Accuracy_CI_Lower': [],
    'Accuracy_CI_Upper': [],
    'F1_Score': [],
    'F1_CI_Lower': [],
    'F1_CI_Upper': [],
    'Precision': [],
    'Precision_CI_Lower': [],
    'Precision_CI_Upper': [],
    'Recall': [],
    'Recall_CI_Lower': [],
    'Recall_CI_Upper': [],
}

n_iterations = 1000
np.random.seed(42)  # pentru reproductibilitate

for model_name in names:
    print(f"  Bootstrap pentru {model_name}...", end=' ')
    
    model_pred = rezultate[model_name]['pred']
    
    # Bootstrap pentru toate metricile
    bootstrap_accuracies = []
    bootstrap_f1s = []
    bootstrap_precisions = []
    bootstrap_recalls = []
    
    for i in range(n_iterations):
        # Resample cu inlocuire
        indices = resample(range(len(y_test)), n_samples=len(y_test), random_state=i)
        
        # Calculam metricile pe sample-ul resampled
        acc = accuracy_score(y_test[indices], model_pred[indices])
        f1 = f1_score(y_test[indices], model_pred[indices], zero_division=0)
        prec = precision_score(y_test[indices], model_pred[indices], zero_division=0)
        rec = recall_score(y_test[indices], model_pred[indices], zero_division=0)
        
        bootstrap_accuracies.append(acc)
        bootstrap_f1s.append(f1)
        bootstrap_precisions.append(prec)
        bootstrap_recalls.append(rec)
    
    # Calculam intervalele de incredere 95% (percentile 2.5% si 97.5%)
    all_confidence_intervals['Model'].append(model_name)
    
    all_confidence_intervals['Accuracy'].append(rezultate[model_name]['accuracy'])
    all_confidence_intervals['Accuracy_CI_Lower'].append(np.percentile(bootstrap_accuracies, 2.5))
    all_confidence_intervals['Accuracy_CI_Upper'].append(np.percentile(bootstrap_accuracies, 97.5))
    
    all_confidence_intervals['F1_Score'].append(rezultate[model_name]['f1'])
    all_confidence_intervals['F1_CI_Lower'].append(np.percentile(bootstrap_f1s, 2.5))
    all_confidence_intervals['F1_CI_Upper'].append(np.percentile(bootstrap_f1s, 97.5))
    
    all_confidence_intervals['Precision'].append(rezultate[model_name]['precision'])
    all_confidence_intervals['Precision_CI_Lower'].append(np.percentile(bootstrap_precisions, 2.5))
    all_confidence_intervals['Precision_CI_Upper'].append(np.percentile(bootstrap_precisions, 97.5))
    
    all_confidence_intervals['Recall'].append(rezultate[model_name]['recall'])
    all_confidence_intervals['Recall_CI_Lower'].append(np.percentile(bootstrap_recalls, 2.5))
    all_confidence_intervals['Recall_CI_Upper'].append(np.percentile(bootstrap_recalls, 97.5))
    
    print("OK")

# Salvam toate intervalele de incredere intr-un singur CSV
df_all_ci = pd.DataFrame(all_confidence_intervals)
df_all_ci.to_csv('results/test_confidence_intervals_all_models.csv', 
                 index=False, float_format='%.6f')
print(f"\nSalvat: results/test_confidence_intervals_all_models.csv")

# Afisam intervalele de incredere pentru cel mai bun model
best_model_name = max(names, key=lambda nm: rezultate[nm]['accuracy'])
best_idx = names.index(best_model_name)

print(f"\nIntervale de incredere 95% pentru cel mai bun model ({best_model_name}):")
print(f"  Accuracy:  {all_confidence_intervals['Accuracy'][best_idx]*100:.2f}% "
      f"(95% CI: [{all_confidence_intervals['Accuracy_CI_Lower'][best_idx]*100:.2f}%, "
      f"{all_confidence_intervals['Accuracy_CI_Upper'][best_idx]*100:.2f}%])")
print(f"  F1-Score:  {all_confidence_intervals['F1_Score'][best_idx]*100:.2f}% "
      f"(95% CI: [{all_confidence_intervals['F1_CI_Lower'][best_idx]*100:.2f}%, "
      f"{all_confidence_intervals['F1_CI_Upper'][best_idx]*100:.2f}%])")
print(f"  Precision: {all_confidence_intervals['Precision'][best_idx]*100:.2f}% "
      f"(95% CI: [{all_confidence_intervals['Precision_CI_Lower'][best_idx]*100:.2f}%, "
      f"{all_confidence_intervals['Precision_CI_Upper'][best_idx]*100:.2f}%])")
print(f"  Recall:    {all_confidence_intervals['Recall'][best_idx]*100:.2f}% "
      f"(95% CI: [{all_confidence_intervals['Recall_CI_Lower'][best_idx]*100:.2f}%, "
      f"{all_confidence_intervals['Recall_CI_Upper'][best_idx]*100:.2f}%])")

# ---------------------------------------------------------------
# PASUL 6: GRAFICE FINALE
# ---------------------------------------------------------------
print("\n" + "="*60)
print("PASUL 6: Generare grafice finale")
print("="*60)

# ===== GRAFIC 1: Tabel comparativ toate metricile =====
# Vizualizeaza toate metricile intr-un format tabelar clar
# Celula cu cel mai bun scor per coloana evidentiata cu verde deschis
fig, ax = plt.subplots(figsize=(14, 4.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.axis('off')
# Dezactivam axele — vrem doar tabelul

# Pregatim datele pentru tabel
col_labels = ['Model', 'Accuracy', 'Log Loss', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
# Etichetele coloanelor

row_data = []
for nm in names:
    r = rezultate[nm]
    row_data.append([
        nm,
        f"{r['accuracy']*100:.2f}%",
        f"{r['loss']:.4f}",
        f"{r['precision']*100:.2f}%",
        f"{r['recall']*100:.2f}%",
        f"{r['f1']*100:.2f}%",
        f"{r['auc']:.4f}",
    ])

table = ax.table(
    cellText=row_data,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
    # bbox=[x, y, width, height] in coordonate relative la ax
)

table.auto_set_font_size(False)
table.set_fontsize(11)

# Stilizam header-ul
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#1E293B')
    # Fundal inchis pentru header
    table[0, j].set_text_props(color='white', fontweight='bold')
    # Text alb bold pentru header

# Determinam cel mai bun scor per coloana
# Accuracy, Precision, Recall, F1, AUC → vrem maximul
# Loss → vrem minimul
metrici_valori = {
    1: [rezultate[nm]['accuracy']  for nm in names],  # Accuracy
    2: [rezultate[nm]['loss']      for nm in names],  # Loss
    3: [rezultate[nm]['precision'] for nm in names],  # Precision
    4: [rezultate[nm]['recall']    for nm in names],  # Recall
    5: [rezultate[nm]['f1']        for nm in names],  # F1
    6: [rezultate[nm]['auc']       for nm in names],  # AUC
}

for col_idx, valori in metrici_valori.items():
    if col_idx == 2:  # Loss — vrem minimul
        best_row = int(np.argmin(valori)) + 1
    else:             # restul — vrem maximul
        best_row = int(np.argmax(valori)) + 1

    table[best_row, col_idx].set_facecolor('#DCFCE7')
    # Verde deschis pentru cel mai bun scor din coloana

# Stilizam randurile alternativ pentru lizibilitate
# Folosim un set separat pentru a sti ce celule au fost deja colorate cu verde
celule_verzi = set()
for col_idx, valori in metrici_valori.items():
    if col_idx == 2:
        best_row = int(np.argmin(valori)) + 1
    else:
        best_row = int(np.argmax(valori)) + 1
    celule_verzi.add((best_row, col_idx))

for i in range(1, len(names) + 1):
    for j in range(len(col_labels)):
        if (i, j) not in celule_verzi and j != 0:
            # Rand par → fundal usor colorat, doar pe celulele neevidientiate
            if i % 2 == 0:
                table[i, j].set_facecolor('#F8FAFC')

    # Coloram prima coloana (numele modelului) cu culoarea modelului
    table[i, 0].set_facecolor(CULORI[names[i-1]])
    table[i, 0].set_text_props(color='white', fontweight='bold')

ax.set_title('Rezultate finale — Test Set',
             fontsize=14, fontweight='semibold',
             color=TEXT, pad=20, y=1.05)
plt.tight_layout()
plt.savefig('grafice/test_tabel_metrici.png')
plt.close()
print("Salvat: grafice/test_tabel_metrici.png")

# ===== GRAFIC 2: Matrici de confuzie (toate 5 modele) =====
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
fig.patch.set_facecolor(BG)
fig.suptitle('Matrice de confuzie — Test Set (toate modelele)',
             fontsize=13, fontweight='semibold', color=TEXT, y=1.02)

lab = ['Ham (0)', 'Spam (1)']
etichete_cm = [['TN', 'FP'], ['FN', 'TP']]

for idx, nm in enumerate(names):
    ax = axes[idx]
    cm = rezultate[nm]['cm']
    ax.imshow(cm, cmap='Blues')

    ax.set_title(nm, fontsize=11, fontweight='semibold',
                 color=rezultate[nm]['culoare'], pad=10)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(lab, fontsize=8.5)
    ax.set_yticklabels(lab, fontsize=8.5)
    ax.set_xlabel('Predicție', fontsize=9)
    ax.set_ylabel('Adevărat', fontsize=9)

    cm_max   = cm.max() if cm.max() > 0 else 1
    cm_total = cm.sum()

    for i in range(2):
        for j in range(2):
            txt_color = 'white' if cm[i, j] > cm_max * 0.5 else '#1E293B'
            pct = (cm[i, j] / cm_total * 100) if cm_total > 0 else 0
            ax.text(j, i,
                    f"{cm[i, j]}\n{etichete_cm[i][j]}\n({pct:.1f}%)",
                    ha='center', va='center', color=txt_color,
                    fontsize=8.5, fontweight='bold', linespacing=1.3)

    tn, fp, fn, tp = cm.ravel()
    ax.text(0.5, -0.22,
            f"Acc: {rezultate[nm]['accuracy']*100:.2f}%  |  FN: {fn}",
            transform=ax.transAxes, ha='center', va='top',
            fontsize=8.5, color=TEXT)

plt.tight_layout()
plt.savefig('grafice/test_matrice_confuzie.png')
plt.close()
print("Salvat: grafice/test_matrice_confuzie.png")

# ===== GRAFIC 3: Accuracy comparativ pe test =====
fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(AX_BG)
ax.set_axisbelow(True)

accs   = [rezultate[nm]['accuracy'] for nm in names]
culori = [rezultate[nm]['culoare']  for nm in names]
x_pos  = np.arange(len(names))

bars = ax.bar(x_pos, accs, color=culori,
              alpha=0.88, edgecolor='white', linewidth=0.8,
              zorder=3, width=0.5)

# Evidentierea celui mai bun model cu bordura aurie
best_idx = int(np.argmax(accs))
bars[best_idx].set_edgecolor('#F59E0B')
bars[best_idx].set_linewidth(2.5)

for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2.,
            val + 0.0008,
            f'{val*100:.2f}%', ha='center', va='bottom',
            fontsize=11, fontweight='bold', color=TEXT)

ax.set_xlabel('Model', fontsize=12, fontweight='semibold')
ax.set_ylabel('Accuracy pe Test Set', fontsize=12, fontweight='semibold')
ax.set_title('Comparație Accuracy — Test Set',
             fontweight='semibold', pad=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(names, fontsize=10)
ymin = max(0, min(accs) - 0.02)
ax.set_ylim(ymin, 1.008)
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f'{v*100:.1f}%'))

# Linie verticala de separare ML / DL
ax.axvline(x=2.5, color='#94A3B8', linestyle='--',
           linewidth=1.5, alpha=0.7, zorder=1)
ax.text(1.0, min(accs) - 0.008, 'ML clasic',
        ha='center', fontsize=9, color='#64748B', fontstyle='italic')
ax.text(3.5, min(accs) - 0.008, 'Deep Learning',
        ha='center', fontsize=9, color='#64748B', fontstyle='italic')

plt.tight_layout()
plt.savefig('grafice/test_accuracy.png')
plt.close()
print("Salvat: grafice/test_accuracy.png")

# ===== GRAFIC 4: Loss comparativ pe test =====
fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(AX_BG)
ax.set_axisbelow(True)

losses = [rezultate[nm]['loss'] for nm in names]

bars = ax.bar(x_pos, losses, color=culori,
              alpha=0.88, edgecolor='white', linewidth=0.8,
              zorder=3, width=0.5)

# Evidentierea celui mai bun model (loss minim)
best_loss_idx = int(np.argmin(losses))
bars[best_loss_idx].set_edgecolor('#F59E0B')
bars[best_loss_idx].set_linewidth(2.5)

for bar, val in zip(bars, losses):
    ax.text(bar.get_x() + bar.get_width()/2.,
            val + max(losses)*0.015,
            f'{val:.4f}', ha='center', va='bottom',
            fontsize=11, fontweight='bold', color=TEXT)

ax.set_xlabel('Model', fontsize=12, fontweight='semibold')
ax.set_ylabel('Log Loss pe Test Set', fontsize=12, fontweight='semibold')
ax.set_title('Comparație Log Loss — Test Set',
             fontweight='semibold', pad=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylim(0, max(losses) * 1.28)

ax.axvline(x=2.5, color='#94A3B8', linestyle='--',
           linewidth=1.5, alpha=0.7, zorder=1)
ax.text(1.0, max(losses) * 0.05, 'ML clasic',
        ha='center', fontsize=9, color='#64748B', fontstyle='italic')
ax.text(3.5, max(losses) * 0.05, 'Deep Learning',
        ha='center', fontsize=9, color='#64748B', fontstyle='italic')

plt.tight_layout()
plt.savefig('grafice/test_loss.png')
plt.close()
print("Salvat: grafice/test_loss.png")

# ===== GRAFIC 5: F1-Score comparativ pe test =====
fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(AX_BG)
ax.set_axisbelow(True)

f1s = [rezultate[nm]['f1'] for nm in names]

bars = ax.bar(x_pos, f1s, color=culori,
              alpha=0.88, edgecolor='white', linewidth=0.8,
              zorder=3, width=0.5)

best_f1_idx = int(np.argmax(f1s))
bars[best_f1_idx].set_edgecolor('#F59E0B')
bars[best_f1_idx].set_linewidth(2.5)

for bar, val in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width()/2.,
            val + 0.0008,
            f'{val*100:.2f}%', ha='center', va='bottom',
            fontsize=11, fontweight='bold', color=TEXT)

ax.set_xlabel('Model', fontsize=12, fontweight='semibold')
ax.set_ylabel('F1-Score pe Test Set', fontsize=12, fontweight='semibold')
ax.set_title('Comparație F1-Score — Test Set',
             fontweight='semibold', pad=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(names, fontsize=10)
ymin_f1 = max(0, min(f1s) - 0.02)
ax.set_ylim(ymin_f1, 1.008)
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f'{v*100:.1f}%'))

ax.axvline(x=2.5, color='#94A3B8', linestyle='--',
           linewidth=1.5, alpha=0.7, zorder=1)
ax.text(1.0, min(f1s) - 0.008, 'ML clasic',
        ha='center', fontsize=9, color='#64748B', fontstyle='italic')
ax.text(3.5, min(f1s) - 0.008, 'Deep Learning',
        ha='center', fontsize=9, color='#64748B', fontstyle='italic')

plt.tight_layout()
plt.savefig('grafice/test_f1score.png')
plt.close()
print("Salvat: grafice/test_f1score.png")

# ===== GRAFIC 6: Curbe ROC suprapuse =====
#
# Curba ROC (Receiver Operating Characteristic):
#   Axa X = FPR (False Positive Rate) = FP / (FP + TN)
#           → procentul emailurilor ham clasificate gresit ca spam
#   Axa Y = TPR (True Positive Rate) = TP / (TP + FN)
#           → procentul emailurilor spam detectate corect
#
# Fiecare punct de pe curba corespunde unui prag de decizie diferit.
# AUC = aria sub curba → cu cat mai aproape de 1.0, cu atat mai bun.
# Linia diagonala = model aleatoriu (AUC = 0.5).
fig, ax = plt.subplots(figsize=(9, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(AX_BG)
ax.set_axisbelow(True)

for nm in names:
    fpr = rezultate[nm]['fpr']
    tpr = rezultate[nm]['tpr']
    auc_score = rezultate[nm]['auc']
    culoare = rezultate[nm]['culoare']

    ax.plot(fpr, tpr, color=culoare, linewidth=2.5,
            label=f'{nm}  (AUC = {auc_score:.4f})',
            zorder=5)

# Linia diagonala = model aleatoriu
ax.plot([0, 1], [0, 1], color='#94A3B8', linewidth=1.5,
        linestyle='--', label='Aleatoriu (AUC = 0.5000)', zorder=4)

ax.set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='semibold')
ax.set_ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='semibold')
ax.set_title('Curbe ROC — Test Set',
             fontweight='semibold', pad=14, fontsize=14)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.02])

ax.legend(fontsize=10, loc='lower right',
          bbox_to_anchor=(0.99, 0.01),
          frameon=True, framealpha=0.97,
          edgecolor='#CBD5E1')

# Adaugam o zona umbrita in coltul stanga-sus pentru zona ideala
ax.fill_between([0, 0, 0.1], [0, 1, 1], alpha=0.04, color='#16A34A')
ax.text(0.02, 0.97, 'Zona\nideală', fontsize=8, color='#16A34A',
        va='top', fontstyle='italic')

plt.tight_layout()
plt.savefig('grafice/test_roc_curves.png')
plt.close()
print("Salvat: grafice/test_roc_curves.png")

# ===== GRAFIC 7: Comparativ complet ML vs DL pe test =====
# Identic structural cu comparatie_ml_vs_dl.png din train_dl
# dar cu valorile REALE de pe test set, nu valorile de pe validation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(BG)
fig.suptitle('Comparație completă — ML clasic vs Deep Learning (Test Set)',
             fontsize=14, fontweight='semibold', color=TEXT, y=1.02)

all_accs   = [rezultate[nm]['accuracy'] for nm in names]
all_losses = [rezultate[nm]['loss']     for nm in names]
all_culori = [rezultate[nm]['culoare']  for nm in names]
x_pos_all  = np.arange(len(names))

# Subplot 1 — Accuracy
bars1 = ax1.bar(x_pos_all, all_accs, color=all_culori,
                alpha=0.88, edgecolor='white', linewidth=0.8,
                zorder=3, width=0.6)

best_acc_all = int(np.argmax(all_accs))
bars1[best_acc_all].set_edgecolor('#F59E0B')
bars1[best_acc_all].set_linewidth(2.5)

for bar, val in zip(bars1, all_accs):
    ax1.text(bar.get_x() + bar.get_width()/2.,
             val + 0.0005,
             f'{val*100:.2f}%', ha='center', va='bottom',
             fontsize=8.5, fontweight='bold', color=TEXT)

ax1.axvline(x=2.5, color='#94A3B8', linestyle='--',
            linewidth=1.5, alpha=0.8, zorder=1)

y_lbl_top = max(all_accs) + (max(all_accs) - min(all_accs)) * 0.10
ax1.text(1.0, y_lbl_top, 'ML clasic', ha='center', fontsize=10,
         color='#475569', fontstyle='italic', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.35', facecolor='#F1F5F9',
                   edgecolor='#94A3B8', alpha=0.95, linewidth=1.2))
ax1.text(3.5, y_lbl_top, 'Deep Learning', ha='center', fontsize=10,
         color='#475569', fontstyle='italic', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.35', facecolor='#F1F5F9',
                   edgecolor='#94A3B8', alpha=0.95, linewidth=1.2))

ax1.set_xticks(x_pos_all)
ax1.set_xticklabels(names, rotation=15, ha='right', fontsize=9.5)
ax1.set_ylabel('Accuracy', fontsize=11, fontweight='semibold')
ax1.set_title('Accuracy', fontweight='semibold', pad=10)
ymin_all = max(0, min(all_accs) - 0.02)
ymax_all  = max(all_accs) + (max(all_accs) - min(all_accs)) * 0.15
ax1.set_ylim(ymin_all, ymax_all)
ax1.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f'{v*100:.1f}%'))
ax1.set_facecolor(AX_BG)
ax1.set_axisbelow(True)

# Subplot 2 — Loss
bars2 = ax2.bar(x_pos_all, all_losses, color=all_culori,
                alpha=0.88, edgecolor='white', linewidth=0.8,
                zorder=3, width=0.6)

best_loss_all = int(np.argmin(all_losses))
bars2[best_loss_all].set_edgecolor('#F59E0B')
bars2[best_loss_all].set_linewidth(2.5)

for bar, val in zip(bars2, all_losses):
    ax2.text(bar.get_x() + bar.get_width()/2.,
             val + max(all_losses)*0.012,
             f'{val:.4f}', ha='center', va='bottom',
             fontsize=8.5, fontweight='bold', color=TEXT)

ax2.axvline(x=2.5, color='#94A3B8', linestyle='--',
            linewidth=1.5, alpha=0.8, zorder=1)

y_lbl2_top = max(all_losses) + (max(all_losses) - 0) * 0.10
ax2.text(1.0, y_lbl2_top, 'ML clasic', ha='center', fontsize=10,
         color='#475569', fontstyle='italic', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.35', facecolor='#F1F5F9',
                   edgecolor='#94A3B8', alpha=0.95, linewidth=1.2))
ax2.text(3.5, y_lbl2_top, 'Deep Learning', ha='center', fontsize=10,
         color='#475569', fontstyle='italic', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.35', facecolor='#F1F5F9',
                   edgecolor='#94A3B8', alpha=0.95, linewidth=1.2))

ax2.set_xticks(x_pos_all)
ax2.set_xticklabels(names, rotation=15, ha='right', fontsize=9.5)
ax2.set_ylabel('Log Loss', fontsize=11, fontweight='semibold')
ax2.set_title('Log Loss', fontweight='semibold', pad=10)
ymax2_all = max(all_losses) + (max(all_losses) - 0) * 0.15
ax2.set_ylim(0, ymax2_all)
ax2.set_facecolor(AX_BG)
ax2.set_axisbelow(True)

plt.tight_layout(rect=[0.01, 0.03, 0.99, 0.92])
plt.savefig('grafice/test_comparatie_ml_vs_dl.png')
plt.close()
print("Salvat: grafice/test_comparatie_ml_vs_dl.png")

# ---------------------------------------------------------------
# PASUL 7: SUMAR FINAL
# ---------------------------------------------------------------
print("\n" + "="*65)
print("SUMAR FINAL — EVALUARE PE TEST SET")
print("="*65)
print(f"{'Model':<22} {'Acc':>8} {'Loss':>8} "
      f"{'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8}")
print("-"*72)

for nm in names:
    r = rezultate[nm]
    print(f"{nm:<22} "
          f"{r['accuracy']*100:>7.2f}%  "
          f"{r['loss']:>7.4f}  "
          f"{r['precision']*100:>7.2f}%  "
          f"{r['recall']*100:>7.2f}%  "
          f"{r['f1']*100:>7.2f}%  "
          f"{r['auc']:>7.4f}")

# Determinam cel mai bun model per metrica
print("\nCel mai bun model per metrica:")
print(f"  Accuracy:  {names[np.argmax([rezultate[nm]['accuracy']  for nm in names])]}")
print(f"  Log Loss:  {names[np.argmin([rezultate[nm]['loss']      for nm in names])]}")
print(f"  Precision: {names[np.argmax([rezultate[nm]['precision'] for nm in names])]}")
print(f"  Recall:    {names[np.argmax([rezultate[nm]['recall']    for nm in names])]}")
print(f"  F1-Score:  {names[np.argmax([rezultate[nm]['f1']        for nm in names])]}")
print(f"  AUC-ROC:   {names[np.argmax([rezultate[nm]['auc']       for nm in names])]}")

# FN-urile — critic pentru infrastructura CIS
print("\nSpam nedetectat (FN) — critic pentru infrastructura CIS:")
for nm in names:
    tn, fp, fn, tp = rezultate[nm]['cm'].ravel()
    print(f"  {nm:<22} FN = {fn:4d} emailuri spam scapate")

# Recomandarea finala
print("\n" + "="*65)
print("RECOMANDARE PENTRU INFRASTRUCTURA CIS")
print("="*65)

# Gasim modelul cu cel mai mic FN
fn_values = {}
for nm in names:
    tn, fp, fn, tp = rezultate[nm]['cm'].ravel()
    fn_values[nm] = fn
best_fn_model = min(fn_values, key=fn_values.get)

# Gasim modelul cu cel mai rapid timp de inferenta
fastest_model = min(names, key=lambda nm: rezultate[nm]['timp_inferenta'])

print(f"\n  Performanta maxima (cel mai putin spam scapat):")
print(f"    → {best_fn_model}")
print(f"       FN = {fn_values[best_fn_model]} emailuri spam scapate")
print(f"       Accuracy = {rezultate[best_fn_model]['accuracy']*100:.2f}%")
print(f"       F1-Score = {rezultate[best_fn_model]['f1']*100:.2f}%")

# Verificam daca e model DL (necesita GPU pentru antrenare, nu neaparat pentru inferenta)
if best_fn_model in ['BiLSTM', 'CNN1D']:
    print(f"       Recomandat pentru scenarii cu resurse compute disponibile (antrenare DL)")
else:
    print(f"       Recomandat pentru scenarii cu resurse limitate (antrenare rapida ML)")
print(f"       Prioritate: reducerea mesajelor spam/phishing nedetectate (FN minim)")

print(f"\n  Optim pentru inferenta rapida:")
print(f"    → {fastest_model}")
print(f"       Timp inferenta = {rezultate[fastest_model]['timp_inferenta']:.3f}s pentru {len(y_test):,} emailuri")
print(f"       Timp mediu/email = {(rezultate[fastest_model]['timp_inferenta'] / len(y_test)) * 1000:.2f} ms")
print(f"       Throughput = {len(y_test) / rezultate[fastest_model]['timp_inferenta']:.0f} emailuri/s")
print(f"       Accuracy = {rezultate[fastest_model]['accuracy']*100:.2f}%")
print(f"       F1-Score = {rezultate[fastest_model]['f1']*100:.2f}%")
print(f"       Recomandat pentru sisteme cu volum mare de emailuri si latenta critica")

print("\nGrafice salvate in grafice/:")
print("  test_tabel_metrici.png")
print("  test_matrice_confuzie.png")
print("  test_accuracy.png")
print("  test_loss.png")
print("  test_f1score.png")
print("  test_roc_curves.png")
print("  test_comparatie_ml_vs_dl.png")

print("\nRezultate salvate in results/:")
print("  test_final_results.csv")
print("  test_confidence_intervals_all_models.csv")

print("\n=== EVALUARE FINALA COMPLETA! ===")
print("Proiectul ML/DL este complet.")