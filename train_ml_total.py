# =============================================================
# TRAIN_ML_TOTAL.PY — VERSIUNEA FINALA
# =============================================================
# Scopul acestui fisier:
#   Antrenare iterativa pentru 3 algoritmi ML:
#   1. Linear SGD  → SGDClassifier(loss='modified_huber')
#   2. SVM-SGD          → SGDClassifier(loss='hinge')
#   3. Gradient Boosting → GradientBoostingClassifier
#      (mai puternic decat Random Forest — arbori secventiali)
#
#   + Calibrare probabilistica pentru Linear-SGD si SVM (scade loss-ul)
#
# La fiecare epoca (Linear SGD/SVM) calculam:
#   - Train Loss + Validation Loss
#   - Train Accuracy + Validation Accuracy
#
# Gradient Boosting: antrenare completa (nu iterativa)
#   dar afiseaza progresul intern prin verbose=1
#
# NU atingem test set-ul — sigilat pentru test_total.py
# =============================================================

import numpy as np # Pentru matrice sparse, array-uri, operatii matematice
import pandas as pd # folosit pentru smooth() — medie mobila in grafice (netezire curbe)
import matplotlib.pyplot as plt # deseneaza graficele de convergenta
import matplotlib.ticker as mticker # formatare axa y in procente pentru graficele de acuratete
import scipy.sparse as sp # pentru incarcarea matricei sparse TF-IDF (format .npz)
import joblib # pentru salvarea si incarcarea modelelor ML (SGD, GB) si a scaler-ului
import time # pentru masurarea timpului de antrenare
import os # pentru crearea folderului 'models' unde salvam toate artefactele (modele, scaler)
import copy # pentru deepcopy() — salvam o copie a modelului la fiecare imbunatatire a validation loss (early stopping)
import warnings # pentru a suprima avertismentele legate de convergenta sau deprecieri in sklearn (nu ne intereseaza in acest context)
warnings.filterwarnings('ignore') # suprimam toate avertismentele pentru output curat (optional, dar face console-ul mai lizibil)

from sklearn.linear_model import SGDClassifier # pentru Linear-SGD si SVM cu antrenare iterativa (partial_fit)
from sklearn.ensemble import GradientBoostingClassifier # pentru Gradient Boosting (antrenare completa, dar cu verbose pentru progres)
from sklearn.isotonic import IsotonicRegression # pentru calibrare probabilistica (mapare monotonă a scorurilor necalibrate la probabilități calibrate)
from sklearn.base import BaseEstimator, ClassifierMixin # pentru a defini CalibratorPrefit ca un estimator compatibil sklearn (pentru a inlocui CalibratedClassifierCV(cv='prefit'))
from sklearn.model_selection import train_test_split # pentru a crea validation split intern 90/10 din train (monitorizare convergenta) si un split separat pentru calibrare (evitam optimism pe validation)
from sklearn.preprocessing import MaxAbsScaler # pentru normalizarea features (TF-IDF + numerice) la scara [-1, 1] — necesar pentru SVM si Gradient Boosting
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix # pentru evaluarea performantei (accuracy_score pentru acuratete, log_loss pentru pierdere, confusion_matrix pentru analiza erorilor)
from sklearn.utils.class_weight import compute_class_weight # pentru calcularea ponderilor de clasa invers proportional cu frecventa (ham vs spam) — ajuta la echilibrarea antrenarii pe clase dezechilibrate 


class CalibratorPrefit(BaseEstimator, ClassifierMixin): # definitie personalizata pentru calibrare isotonică pe un model deja antrenat (similar cu CalibratedClassifierCV(cv='prefit') dar compatibil cu toate versiunile sklearn). ne ajuta sa imbunatatim log loss-ul prin maparea monotonă a scorurilor necalibrate la probabilități calibrate, folosind IsotonicRegression. fit pe un set separat (X_cal_sc, y_cal) pentru a evita optimismul pe validation. predict_proba returneaza probabilitatile calibrate pentru fiecare email, iar predict returneaza clasele prezise (0 sau 1) bazate pe pragul de 0.5.
    # Prefit = modelul de baza este deja antrenat. doar ii calibreaza iesirea
    # BaseEstimator = face obiectul compatibil cu sklearn (fit, predict_proba, predict)
    # ClassifierMixin = acest obiect se comporta ca un clasificator => are logica de clasificare, de tip predict 
    """
    Calibrare isotonică pe un clasificator deja antrenat.
    Înlocuiește CalibratedClassifierCV(cv='prefit'), care nu mai e acceptat
    în unele versiuni sklearn.
    """

    def __init__(self, base_estimator): # __init__ este constructorul clasei 
        # self este obiectul curent care se creeaza 
        # base_estimator este modelul de bază (NB sau SVM) care a fost deja antrenat pe datele de train și produce scoruri necalibrate. CalibratorPrefit va folosi aceste scoruri pentru a învăța o funcție de calibrare isotonică care mapează scorurile necalibrate la probabilități calibrate. Salvăm base_estimator ca atribut al obiectului pentru a-l folosi ulterior în predict_proba.
        self.base_estimator = base_estimator # salvăm modelul de bază (NB sau SVM) ca atribut al obiectului pentru a-l folosi ulterior în predict_proba


    def _proba_clasa_pozitiva(self, X): # metoda privata care alculeaza pentru fiecare exemplu din X, un scor pentru clasa pozitiva, adica pentru spam
        if hasattr(self.base_estimator, 'predict_proba'): # verificam daca modelul de bază are metoda predict_proba (NB cu modified_huber o are, SVM cu hinge nu o are)
            p = self.base_estimator.predict_proba(X) # daca modelul are predict_proba, o folosim direct pentru a obtine probabilitatile pentru fiecare clasa (ham si spam). p va fi o matrice cu shape (n_samples, 2) unde p[i, 0] este probabilitatea ca emailul i sa fie ham si p[i, 1] este probabilitatea ca emailul i sa fie spam. Returnam doar coloana 1 (p[:, 1]) care reprezinta probabilitatea clasei pozitive (spam) pentru fiecare email.
            return np.asarray(p[:, 1], dtype=np.float64) # returnam doar coloana 1, care reprezinta probabilitatea pentru clasa 1 (spam)
            # np.asarray(..., dtype=np.float64) transforma rezultatul intr-un array numpy de tip float64, pentru a asigura compatibilitatea cu IsotonicRegression care asteapta valori float64
        z = np.clip(self.base_estimator.decision_function(X), -20.0, 20.0) # daca modelul nu are predict_proba, se executa randul asta care foloseste decision_function, clip este folosit ca sa prevenim overflow-ul cand aplicam exp 
        return 1.0 / (1.0 + np.exp(-z)) # sigmoida care transforma orice numar real in interval (0,1)

    def fit(self, X, y, sample_weight=None): # antrenam calibratorul. invatam doar functia care corecteaza probabilitatile brute 
        self.classes_ = np.array([0, 1]) # definim clasele (ham = 0 si spam = 1) pentru compatibilitate cu sklearn
        p1 = self._proba_clasa_pozitiva(X) # calculam scorurile pentru toate exemplele din setul de calibrare 
        self.calibrator_ = IsotonicRegression(out_of_bounds='clip') # creem modelul de calibrare => daca un email are scor mai mare decat altul inainte de calibrare, va ramane mai mare si dupa calibrare
        # out_of_bounds='clip' inseamna ca daca la predictie primim un scor care este in afara intervalului de scoruri vazute la fit, atunci va returna probabilitatea calibrata corespunzatoare celui mai apropiat scor vazut la fit (daca e sub minim, returneaza probabilitatea pentru minim; daca e peste maxim, returneaza probabilitatea pentru maxim). Acest lucru previne returnarea unor probabilitati necalibrate pentru scoruri extreme care nu au fost vazute la fit.
        try: # incercam sa antrenam calibratorul 
            self.calibrator_.fit(
                p1, y.astype(np.float64), 
                sample_weight=sample_weight
                # p1 = intrarea pentru calibrare (scorurile necalibrate pentru clasa pozitiva)
                # y.astype(np.float64) = etichetele reale (ham=0, spam=1) convertite in float64 pentru compatibilitate cu IsotonicRegression
            )
        except TypeError: # daca versiunea de sklearn nu suporta sample_weight in IsotonicRegression, incercam fara sample_weight
            self.calibrator_.fit(p1, y.astype(np.float64)) 
        return self # returnam obiectul calibrat pentru a permite chaining 

    def predict_proba(self, X): # calculeaza probabilitatile calibrate pentru fiecare email din X
        p1 = self._proba_clasa_pozitiva(X) # oobtinem scorul initial pentru clasa pozitiva (spam)
        p_cal = np.clip(self.calibrator_.predict(p1), 0.0, 1.0) # aplicam functia de calibrare invatata pentru a obtine probabilitatea calibrata pentru clasa pozitiva (spam). clip este folosit pentru a ne asigura ca rezultatul este in intervalul [0, 1], deoarece unele valori de scor necalibrate pot duce la probabilitati calibrate care depasesc acest interval.
        return np.column_stack([1.0 - p_cal, p_cal]) # returnam o matrice cu doua coloane ( prima coloana, probabilitatea pentru clasa 0, a doua pentru clasa 1 )

    def predict(self, X): # returneaza clasele finale prezise pentru fiecare exemplu din X 
        return np.argmax(self.predict_proba(X), axis=1) # predict_proba(X) returneaza o matrice cu probabilitatea de calibrare pentru fiecare clasa 
        # axis = 1 = fa asta pe fiecare rand
        # argmax = ia pozitia valorii celei mai mari 


# ---------------------------------------------------------------
# STILUL GLOBAL AL GRAFICELOR
# ---------------------------------------------------------------
CULORI = {
    'Linear-SGD':        '#0F766E', # verde petrol pentru Linear-SGD
    'SVM-SGD':           '#4338CA', # indigo pentru SVM-SGD
    'Gradient Boosting': '#C2410C', # portocaliu pentru Gradient Boosting
}

COL_VAL  = '#DC2626'
BG       = '#F8FAFC'
AX_BG    = '#FFFFFF'
TEXT     = '#1E293B'
GRID_COL = '#E2E8F0'

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

# ---------------------------------------------------------------
# PASUL 1: Incarcam datele salvate de preprocess_total.py
#
# X_train_sparse.npz = matrice 34.436 x 15.010
#   - 15.000 features TF-IDF (text procesat cu trigrame)
#   - 10 features numerice (4 URL + 6 structurale)
# y_train.npy = vector 34.436 valori (0=ham, 1=spam)
# ---------------------------------------------------------------
print("=" * 65)
print("TRAIN_ML_TOTAL.PY — Antrenare ML")
print("=" * 65)

print("\nIncarc datele de antrenare...")
X_train_full = sp.load_npz('data/train/X_train_sparse.npz') # matricea completa de train ( toate exemplele de antrenare si toate features-urile lor )
y_train_full = np.load('data/train/y_train.npy') # vectorul care spune pentru fiecare email din X_train_full daca este ham sau spam

print(f"  Emailuri: {X_train_full.shape[0]:,}") # afisam numarul de emailuri din matrice
print(f"  Features: {X_train_full.shape[1]:,} " # afisam numarul de features din matrice
      f"(15.000 TF-IDF + 10 numerice)")
print(f"  Spam:     {y_train_full.sum():,} " # afisam nr de emailuri spam (adunam toate valorile din vector, iar suma ne da exact numarul de 1(spam))
      f"({y_train_full.mean()*100:.1f}%)") # afisam procentul de spam, dar in forma zecimala
print(f"  Ham:      {(y_train_full==0).sum():,} " # afisam cate emailuri sunt ham (verificam fiecare element daca este egal cu 0)
      f"({(1-y_train_full.mean())*100:.1f}%)") # afisam procentul de ham 

# ---------------------------------------------------------------
# PASUL 2: Validation split intern 90/10
#
# 90% din train → antrenare efectiva
# 10% din train → validare per epoca (monitorizare convergenta)
#
# Test set-ul NU este atins — ramane sigilat pentru test_total.py
# ---------------------------------------------------------------
print("\nCreez validation split intern 90/10...")
X_tr, X_val, y_tr, y_val = train_test_split(
    # X_train_full = matricea completa de train, incarcata de fisier 
    # y_train_full = etichetele complete pentru acel train
    # X_tr = datele de intrare folosite efectiv la antrenare
    # y_tr = etichetele pentru aceste date de antrenare 
    # X_val = datele de validare
    # y_val - etichetele reale pentru datele de validare 
    X_train_full, y_train_full,
    test_size=0.1, # 10% pentru test 
    random_state=42,
    stratify=y_train_full
)
print(f"  Train efectiv: {X_tr.shape[0]:,} emailuri") # afisam nr de emailuri pentru train
print(f"  Validation:    {X_val.shape[0]:,} emailuri") # afisam nr de emailuri din validation

# ---------------------------------------------------------------
# PASUL 3: Normalizare MaxAbsScaler
#
# De ce? TF-IDF produce valori intre 0 si 1, dar features
# numerice au scale diferit (lung_email poate fi 5000+).
# MaxAbsScaler imparte fiecare coloana la maximul sau absolut
# → toate features ajung in intervalul [-1, 1].
# Necesar pentru SVM si Gradient Boosting.
#
# fit pe train = invata scalele din train
# transform pe val = aplica aceleasi scale (nu fit din nou!)
# ---------------------------------------------------------------
print("\nNormalizez features (MaxAbsScaler)...") # afisam mesaj in terminal ca sa stim ce executam
scaler = MaxAbsScaler() # creaza un obiect nou de tip MaxAbsScaler si il stocheaza in variabila scaler
X_tr_sc  = scaler.fit_transform(X_tr) # fit- scaler-ul invata maximele din toate datele de train; transform - aplica imediat normalizarea cu maximele tocmai invatate
# rezultatul X_tr_sc = matricea de train normalizata 
X_val_sc = scaler.transform(X_val) # normalizeaza validation set-ul cu aceleasi maxime invatate din train
# Asigura folderul pentru toate artefactele model
os.makedirs('models', exist_ok=True) # creem folderul daca nu exista 
joblib.dump(scaler, 'models/feature_scaler.pkl') # salveaza obiectul scaler pe disk
print("  Salvat: models/feature_scaler.pkl")

# Split separat pentru calibrare, ca sa evitam optimism pe validation
X_cal_sc, X_val_eval_sc, y_cal, y_val_eval = train_test_split(
    X_val_sc, y_val, # taiem validation set-ul normalizat
    test_size=0.5, # taiem validation in doua jumatati 
    random_state=43, # seed diferit fata de 42 - alt split aleatoriu
    stratify=y_val # proportie spam/ham identica in ambele jumatati
    # X_cal_sc.shape[0] = numaarul de randuri din matrice = numarul de emailuri 
    # am facut toate astea ca SVM si GB sa poata lucra corect - nicio feature nu domina prin marime 
)
print(f"  Calibrare:     {X_cal_sc.shape[0]:,} emailuri")
print(f"  Val raportare: {X_val_eval_sc.shape[0]:,} emailuri")

# ---------------------------------------------------------------
# PASUL 4: Ponderi de clasa
#
# compute_class_weight('balanced') calculeaza ponderi inversate
# proportional cu frecventa clasei:
#   w_spam = total / (2 × nr_spam)
#   w_ham  = total / (2 × nr_ham)
# Modelul va acorda mai multa atentie clasei mai rare.
# ---------------------------------------------------------------
classes = np.array([0, 1]) # creem un array cu clasele posibile. face calculul ponderilor
cw_array = compute_class_weight( # functie sklearn care calculeaza automat ponderile
    class_weight='balanced', # foloseste formula inversa frecventei
    # balanced calculeaza automat ponderile in functie de cat de des apare fiecare clasa, adica w_spam si w_ham 
    classes=classes, # spunem ce clase exista. fara ea, functia nu stie sa calculeze 
    y=y_tr # etichetele din train efectiv. functia numara cate emailuri are fiecare clasa si calculeaza ponderile
)
cw_dict = dict(zip(classes, cw_array)) # transformam rezultatul intr-un dictionar. il creem deoarece e mai usor sa accesam greutatea unei clase dupa eticheta ei
sample_weight_tr = np.array( # creem un vector cu ponderi pentru fiecare exemplu din train
    [cw_dict[int(y)] for y in y_tr], 
    # daca y este 0, atribuim greutatea cw_dict[0]
    # daca y este 1, atribuim greutatea cw_dict[1]
    dtype=np.float32 # convertim vectorul la tip float32. pentru economie de memorie si compatubilitate buna in antrenare
)
print(f"\nPonderi clase: ham={cw_dict[0]:.3f}, spam={cw_dict[1]:.3f}") # afisam ponderile calculate

# ---------------------------------------------------------------
# PASUL 5: Hiperparametrii
#
# ---------------------------------------------------------------
RUN_PRESET = 'best_efficiency'   # 'normal' | 'heavy' | 'best_efficiency'
# best_efficiency - ofera un compromis bun intre performanta, timp si risc de overfit
# !!! facem if pentru toate cele trei cazuri DOAR pentru flexibilitate. Noi folosim doar best_efficiency
if RUN_PRESET == 'heavy': 
    # Rulare foarte lunga (cand vrei curbe extrem de dense) 
    EPOCHS        = 5000 # => creste timpul de antrenare
    ETA0          = 0.0025 # learning rate initial putin mai mic => face antrenarea ceva mai fina
    PATIENCE_SGD  = 900 # modelul asteapta 900 de epoci fara imbunatatire inainte sa se opreasca
    MIN_EPOCI     = 1200 # modelul nu are voie sa se opreasca inainte de 1200 de epoci
    GB_ESTIMATORS = 1800 # Gradient Boosting poate folosi pana la 1800 de arbori
    GB_LR         = 0.03 # learning rate pentru GB este 0.03
    GB_DEPTH      = 6 # adancimea arborilor 
    GB_PATIENCE   = 250 # la monitorizarea GB, acceptam 250 de etape fara imbunatatire
elif RUN_PRESET == 'best_efficiency':
    # Varianta recomandata: performanta buna + timp controlat + risc redus de overfit
    EPOCHS        = 3200 # SGD poate rula pana la 3200 epoci
    ETA0          = 0.0028 # learning rate initial 0.0028 ( e mai mic decat in varianta normala, ceea ce ajuta la rafinare mai stabila)
    PATIENCE_SGD  = 380 # asteptam 380 de epoci fara progres inainte de early stop (valoare mai mare => modelul are timp sa se stabilizeze)
    MIN_EPOCI     = 500 # modelul trebuie sa parcurga minim 500 de epoci inainte sa poata fi oprit 
    GB_ESTIMATORS = 2500 # GB poate merge pana la 2500 de arbori
    GB_LR         = 0.035 # learning rate pentru GB 
    GB_DEPTH      = 4 # arborii au adancime 4 
    GB_PATIENCE   = 300 # la monitorizare GB acceptam 300 de etape fara progres
else: # pentru varianta normala (varianta cea mai scurta si mai simpla)
    EPOCHS        = 1500
    ETA0          = 0.003
    PATIENCE_SGD  = 150
    MIN_EPOCI     = 200
    GB_ESTIMATORS = 500
    GB_LR         = 0.05
    GB_DEPTH      = 5
    GB_PATIENCE   = 80

print(f"\nHiperparametri SGD:")
print(f"  Preset rulare:  {RUN_PRESET}")
print(f"  Epoci max:      {EPOCHS}")
print(f"  Learning Rate:  {ETA0} (invscaling, scade cu sqrt(t))")
print(f"  Patience:       {PATIENCE_SGD} epoci")
print(f"  Min epoci:      {MIN_EPOCI}")
print(f"  GB estimators:  {GB_ESTIMATORS}")
print(f"  GB lr/depth:    {GB_LR} / {GB_DEPTH}")
print(f"  GB patience:    {GB_PATIENCE} etape")

# ---------------------------------------------------------------
# PASUL 6: Definim modelele SGD
#
# loss='modified_huber' pentru Linear SGD:
#   - Varianta robusta a log_loss
#   - Suporta predict_proba() direct
#   - Mai robust la outlieri (emailuri atipice)
#   - Curbe de convergenta mai netede
#
# loss='hinge' pentru SVM:
#   - Echivalentul matematic al SVM liniar
#   - Minimizeaza max(0, 1 - y * scor)
#   - Nu produce probabilitati direct
#   - Aplicam sigmoida pe decision_function pentru log_loss
#
# alpha=1e-5:
#   - Regularizare L2 slaba — model mai expresiv
#   - Penalizeaza parametrii prea mari (previne overfit)
#
# average=50:
#   - Averaged SGD — pastreaza media ultimilor 50 pasi
#   - Reduce variatia si imbunatateste generalizarea
# ---------------------------------------------------------------
# separam definitia de antrenare deoarece putem modifica parametrii fara sa atingem logica de antrenare. E o practica de programare curata
linear_sgd_model = SGDClassifier( # SGDClassifier = o clasa din sklearn care implementeaza clasificare prin Stochastic Gradient Descent
    loss='modified_huber', # linear SGD. este stabil, merge bine pe date sparse, suporta predict_proba, mai robust la exemple atipice 
    alpha=1e-5, # controleaza regularizarea L2. adauga o penalizare la functia loss
    learning_rate='invscaling', # defineste cum se schimba learning rate-ul pe parcursul antrenarii
    # invscalling = learning rate scade invers proportional cu radacina patratica a numarului de pasi
    eta0=ETA0, # learning rate-ul initial. Valoarea de start din care invscaliing calculeaza toate valorile ulterioare
    power_t=0.5, # controleaza cat de repede scade learning rate-ul in forma invscalling
    random_state=42, # seteaza seed-ul generatorului de numere pseudo-aleatoare al modelului
    max_iter=1, # facem N(=1) treceri automat
    tol=None, # toleranta pentru oprirea automata interna a sklearn 
    average=50 # mentine o medie rulanta a ultimelor 50 de seturi de parametri in loc sa returneze ultimul set
)

svm_model = SGDClassifier(
    loss='hinge', # se concentreaza pe separarea claselor cu marja, foarte potrivita pentru clasificare liniara 
    alpha=1e-5,
    learning_rate='invscaling',
    eta0=ETA0,
    power_t=0.5,
    random_state=42,
    max_iter=1,
    tol=None,
    average=50
)

# hinge este mai apropiata de SVM-ul clasic si lucreaza cu scoruri brute si marja
# modified_huber este o varianta mai robusta si mai comoda cand vrem si probabilitati

# ---------------------------------------------------------------
# PASUL 7: Functii helper
# Este "motorul" pentru: antrenare, urmarirea metricilor, salvarea celui mai bun model, desenarea graficelor
# ---------------------------------------------------------------
def smooth(values, window=7): # values este lista de valori pe care vrem sa le netezim. folosita pentru netezirea graifcelor
    # window = 7 inseamna ca mmedia se calculeaza pe o fereastra de 7 valori
    """
    Medie mobila pentru netezirea curbelor zgomotoase.
    SGD e stochastic — valorile variaza natural intre epoci.
    Netezirea evidentiaza tendinta generala (trendul).
    window = numarul de valori incluse in medie
    """
    s = pd.Series(values, dtype=float) # transformam lista intr-un obiect Series din pandas
    return s.rolling(window=window, min_periods=1).mean().tolist() # le netezim cu medie mobila
    # .rolling(window=window, min_periods=1) = creeaza o "fereastra glisanta". La fiecare pozitie, fereastra acopera ultimele window aici
    # min_periods = 1 => cate valori minime trebuie sa existe in fereastra pentru a calcula media

# functia centrala pentru antrenarea Linear-SGD si SVM-SGD
def antreneaza_sgd(model, model_name, epochs, # aceasta functie antreneaza un model SGD epoca cu epoca, calculeaza metrici dupa fiecare epoca, aplica early stopping, salveaza cel mai bun model si intoarce istoricul
                   patience, min_epoci, fisier_pkl,
                   X_train_data, X_val_data,
                   sample_weight_train=None):
    """
    Antreneaza un SGDClassifier iterativ pe epoci.

    La fiecare epoca:
      1. partial_fit() pe datele de train (shuffled)
      2. Calculeaza Train Loss si Val Loss
      3. Calculeaza Train Accuracy si Val Accuracy
      4. Verifica daca e best model (early stopping)

    Returneaza: (model_best, history_dict, timp_antrenare)
    """
    print(f"\n{'='*60}")
    print(f"ANTRENEZ: {model_name}")
    print(f"  Loss: {model.loss} | Epoci max: {epochs} | LR: {ETA0}")
    print(f"{'='*60}")

    history = { # creem un dictionar gol in care vom salva evolutia metricilor
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  []
    }

    best_val_loss = np.inf # incepem cu o valoare infinita
    best_model    = None # la inceput nu avem inca niciun model optim salvat
    best_epoca    = 0 # nu avem inca epoca optima
    fara_imbun    = 0 # numara cate epoci trec fara imbunatatire 
    rng           = np.random.default_rng(42) # creeem generatorul aleator pentru shuffling
    start         = time.time() # memoram momentul de inceputt pentru a calcula timpul total de antrenare 

# incepem antrenarea propriu zisa:
    for epoca in range(epochs): # functia repeta procesul de antrenare de la epoca 0 pana la epoca epochs-1
        # Shufflam datele la fiecare epoca
        # Ordinea diferita previne ca modelul sa invete
        # tipare legate de ordinea emailurilor
        idx      = rng.permutation(X_train_data.shape[0]) # amesteca datele inainte de epoca curenta ( se creeaza o ordine aleatoare a tuturor exemplelor din train)
        X_epoch  = X_train_data[idx] # luam datele de train si le punem in ordinea noua, amestecata
        y_epoch  = y_tr[idx] # luam etichetele trainului si le punem exact in aceiasi ordine
        sw_epoca = (sample_weight_train[idx] # se reordoneaza si ponderile daca exista 
                    if sample_weight_train is not None else None)

        # partial_fit = o singura epoca de antrenare
        # classes=[0,1] spune modelului ce clase exista
        # aici are loc invatarea efectiva
        model.partial_fit( # il antrenam epoca cu epoca 
            # partial_fit, spre deosebire de fit, continua din punctul anterior - adauga ce a invatat din aceasta epoca la ce stia deja
            X_epoch, y_epoch,
            classes=classes,
            sample_weight=sw_epoca # ponderile reorodante 
        )

        # Calculam probabilitatile pentru log_loss
        if model.loss == 'modified_huber': # pentru Linear SGD cu modified huber
            # modified_huber suporta predict_proba direct
            # predict_proba poate produce exact 0 sau 1 -> clip pentru log_loss stabil
            tr_proba = np.clip( # calculam valorile pentru train 
                model.predict_proba(X_train_data), 1e-7, 1-1e-7
            )
            vl_proba = np.clip( # calculam valorile pentru validation 
                model.predict_proba(X_val_data), 1e-7, 1-1e-7
            )
            # avem valori limitate intre un nr foarte mic si 1 
        else: # pentru SVM-SGD cu hinge loss
            # hinge nu suporta predict_proba
            # Aplicam sigmoida: f(x) = 1 / (1 + e^(-x))
            # Transforma scoruri reale in probabilitati (0,1)
            tr_sc    = np.clip(
                model.decision_function(X_train_data), -20, 20)
            vl_sc    = np.clip(
                model.decision_function(X_val_data), -20, 20)
            tr_proba = np.clip(1/(1+np.exp(-tr_sc)), 1e-7, 1-1e-7)
            vl_proba = np.clip(1/(1+np.exp(-vl_sc)), 1e-7, 1-1e-7)
        # train metrics = cat de bine a memeorat datele pe care a antrenat
        # val metrics = cat de bine generalizeaza pe date nevazute
        # gap mare (train>>val) -> overfit -> modelul memoreaza
        # ambele mici -> underfit -> modelul nu a invatat 
        # ambele mari, aproap -> ideal

        # Log loss = cat de departe sunt probabilitatile de realitate
        tl = log_loss(y_tr,  tr_proba, labels=[0, 1])
        vl = log_loss(y_val, vl_proba, labels=[0, 1])

        # Accuracy = procentul de emailuri clasificate corect
        ta = accuracy_score(y_tr,  model.predict(X_train_data))
        va = accuracy_score(y_val, model.predict(X_val_data))

        history['train_loss'].append(tl) # adauga cate o valoare la sfarsitul listei. Construim punct cu punct cele 4 curbe de grafic
        history['val_loss'].append(vl)
        history['train_acc'].append(ta)
        history['val_acc'].append(va)

        # Early stopping:
        # Daca val loss scade cu minim 0.0001 → salvam modelul
        # Daca nu scade in PATIENCE epoci consecutive → oprim
        if vl < (best_val_loss - 1e-4):
            best_val_loss = vl
            best_model    = copy.deepcopy(model) # copierea profunda a modelului
            best_epoca    = epoca + 1
            fara_imbun    = 0
        else:
            fara_imbun += 1

        if (epoca + 1) % 100 == 0: # afisam din 100 in 100 de epoci 
            print(f"  Epoca {epoca+1:5d}/{epochs} | "
                  f"TrLoss={tl:.4f} VlLoss={vl:.4f} | "
                  f"TrAcc={ta*100:.2f}% VlAcc={va*100:.2f}%")

        if (epoca + 1) >= min_epoci and fara_imbun >= patience: # conditia de early stopping 
            # (epoca + 1) >= min_epoci => am depasit numarul minim de epoci. protectie impotriva opririi premature cand modelul oscileaza natural la inceput
            # fara_imbun >= patience => au trecut 380 epoci fara imbunatatire 
            print(f"\n  [Early stopping] Oprit la epoca {epoca+1} "
                  f"(best={best_epoca}, VlLoss={best_val_loss:.4f})")
            break # iesim din bucla imediat. nu mai rulam epocile ramase 

    timp = time.time() - start # calculam durata 

    if best_model is not None: # restaurarea celui mai bun model 
        model = best_model

    print(f"\n  Timp total:    {timp/60:.1f} min")
    print(f"  Best Val Loss: {best_val_loss:.4f} (epoca {best_epoca})")
    print(f"  Best Val Acc:  {max(history['val_acc'])*100:.2f}%")

    os.makedirs('models', exist_ok=True) # creem folderul models/ daca nu exista
    joblib.dump(model, fisier_pkl) # salvam modelul pe disk
    print(f"  Salvat:        {fisier_pkl}")

    return model, history, timp # functia returneaza: cel mai bun model antrenat, history (pentru grafice), timp ( pentru graficul de timp)


def deseneaza_subplot(ax, xv, raw_tr, raw_vl, culoare,
                      xl, ylabel, fmt_y=False, show_max=False,
                      annotate_below=False, legend_loc='upper left',
                      legend_anchor=(0.01, 0.99)):
    # ax - obiectul subplot matplotlib pe care desenam. Fiecare panou dintr-un grafic e un ax separat
    # xv = valorile axei X. Pentru modelele cu SGD lista de epoci, pentru GB lista de arbori
    # raw_tr = lista de valori brute Train
    # raw_vl = lista de valori brute Validation
    # culoare = culoarea specifica modelului
    # x1 = eticheta axei X
    # ylabel = eticheta axei Y ( Log Loss sau accuracy)
    # fmt_y=False - daca formatam axa Y ca procent. True pentru accuracy, false pentru loss
    # show_max = False => ce punct optim sa marcam. False = minimul (pentru loss), True = maximul (pentru accuracy)
    # annotate_below = false => unde sa punem eticheta punctului optim
    """
    Deseneaza un subplot de convergenta cu 4 curbe:
      - train raw (transparent): zgomotul natural SGD
      - val raw (transparent): zgomotul natural SGD
      - train smoothed (opac): tendinta generala clara
      - val smoothed (opac, punctat): tendinta validare
    Marcheaza automat punctul optim (min loss / max accuracy).
    """

    # calculam fereastra adaptiva si netezim ambele curbe (deja explicat la functia smooth)
    w     = max(5, min(20, len(xv) // 15)) # dimensiunea ferestrei de netezire
    sm_tr = smooth(raw_tr, window=w) # luam valorile brute de train si aplicam functia smooth cu fereastra calculata mai sus 
    sm_vl = smooth(raw_vl, window=w) # luam valorile brute de validation si aplicam functia smooth cu fereastra calculata mai sus

    # Curbe raw — zgomot vizibil
    ax.plot(xv, raw_tr, color=culoare,  alpha=0.12, linewidth=0.7) # desenam curba train bruta
    ax.plot(xv, raw_vl, color=COL_VAL, alpha=0.12, linewidth=0.7) # desenam curba validation bruta

    # Curbe smoothed — tendinta clara. desenam peste curbele brute doua curbe mai clare si mai groase: una pentru trian si una pentru validation
    ltr, = ax.plot(xv, sm_tr, color=culoare, linewidth=2.3,
                   label='Train (smoothed)')
    lvl, = ax.plot(xv, sm_vl, color=COL_VAL, linewidth=2.3,
                   linestyle='--', label='Validation (smoothed)')
    # fara , ltr ar fi o lista, nu obiect 
    # label='train(smoothed)' eticheta pentru legenda
    # linestyle ='--' validation e desenat punctat

    # Punct optim de pe curba de validation si pregatim eticheta care va fi afisata pe grafic
    # alegem valoarea maxima pentru accuracy si valoare minima pentru loss
    if show_max:
        opt_idx = int(np.argmax(raw_vl))
        opt_val = raw_vl[opt_idx]
        lbl     = f'{opt_val*100:.2f}%'
    else:
        opt_idx = int(np.argmin(raw_vl))
        opt_val = raw_vl[opt_idx]
        lbl     = f'{opt_val:.4f}'

    ax.scatter(xv[opt_idx], opt_val, color=COL_VAL, # folosim scatter pentru a desena un punct izolat, nu o linie
               s=90, zorder=7, edgecolors='white', linewidth=1.2) 
                # xv[opt_idx] = pozitia X a punctului optim (epoca sau numarul de arbori)
                # s=90 -> dimensiunea punctului (in pixeli patrati)
                # zordwe = 7 -> ordinea de afisare pe axa Z (adancimea)
                # edgecolors ='white' bordura alba in jurul punctului. creeaza un efect de inel care face punctul mai vizibil pe fundal colorat

    # ne ocupam de partea vizuala a graficului, de eticheta punctului optim
    if annotate_below and show_max: # daca punctul optim e maxi, atunci punem textul sub punct
        ax.annotate(
            lbl,
            xy=(xv[opt_idx], opt_val),
            xytext=(0, -16), textcoords='offset points',
            fontsize=9.5, color=COL_VAL, fontweight='bold',
            ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      edgecolor=COL_VAL, alpha=0.95, linewidth=0.9)
        )
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0 - (y1 - y0) * 0.08, y1)
    else: # daca nu vrem eticheta sub punct, atunci o punem putin in dreapta si sus fata de punct
        ax.annotate(
            lbl,
            xy=(xv[opt_idx], opt_val),
            xytext=(10, 8), textcoords='offset points',
            fontsize=8.5, color=COL_VAL, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor=COL_VAL, alpha=0.9, linewidth=0.8)
        )

# formatam axele si legendele
    ax.set_xlabel(xl, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if fmt_y:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f'{v*100:.1f}%'))
    ax.set_facecolor(AX_BG)
    ax.set_axisbelow(True)
    # Poziție fixă: 'best' + tight_layout() tăiau deseori legenda din PNG
    leg = ax.legend(
        handles=[ltr, lvl],
        fontsize=9,
        loc=legend_loc,
        bbox_to_anchor=legend_anchor,
        borderaxespad=0.4,
        frameon=True,
        fancybox=True,
        framealpha=0.97,
        edgecolor='#CBD5E1',
    )
    leg.set_zorder(2000)
    try:
        leg.set_in_layout(False)
    except Exception:
        pass


# ---------------------------------------------------------------
# PASUL 8: Antrenam LinearSGD_Model si SVM
# ---------------------------------------------------------------
linear_sgd_model, history_linear_sgd, timp_linear_sgd = antreneaza_sgd( # functia intoarce 3 rezultate si le salvam in 3 variabile
    linear_sgd_model, # modelul pe care il trimitem la antrenare
    'Linear-SGD (modified_huber)', # numele modelului, folosit la afisare in terminal
    EPOCHS, PATIENCE_SGD, MIN_EPOCI, # hiperparametrii stabiliti mai devreme 
    'models/ml_linear_sgd.pkl', # aici o sa fie salvat modelul
    X_train_data=X_tr_sc, # trimitem datele de train scalate
    X_val_data=X_val_sc, # trimitem datele de validation scalate 
    sample_weight_train=sample_weight_tr # trimitem ponderile pentru exemplele din train, ca modelul sa compenseze dezechilibrul dintre clase 
)

svm_model, history_svm, timp_svm = antreneaza_sgd( # facem acelasi lucru, dar pentru modelul SCM-SGD
    svm_model,
    'SVM-SGD (hinge)', # singura diferenta. modified_hubber este o varianta mai robusta si mai blanda, care poate oferi probabilitati direct
     # hinge este functia de pierdere specifica SVM liniar
    EPOCHS, PATIENCE_SGD, MIN_EPOCI,
    'models/ml_svm_sgd.pkl',
    X_train_data=X_tr_sc,
    X_val_data=X_val_sc,
    sample_weight_train=sample_weight_tr
)

# ---------------------------------------------------------------
# PASUL 9: Calibrare probabilistica
#
# Problema: SGD produce scoruri necalibrate.
# Un model zice "73% spam" dar in realitate poate fi 95%.
# Log loss penalizeaza aceasta nesiguranta.
#
# CalibratorPrefit (isotonic):
#   Mapează monoton P(spam) necalibrat → calibrat (fără cv='prefit').
#   Fit pe același set folosit înainte (validare).
#
# Efectul: Val Loss scade de la ~0.20 la ~0.07-0.09
#
#
#
# Nu mai invatam modelul sa clasifice ham vs spam. Acum le corectam probabilitatile
# ---------------------------------------------------------------
print(f"\n{'='*60}")
print("CALIBRARE PROBABILISTICA")
print(f"{'='*60}")

print("Calibrez Linear-SGD...") # creem si antrenam calibratorul pentru linear SVM 
linear_sgd_cal = CalibratorPrefit(linear_sgd_model) # creem calibratorul pentru linear SGD
linear_sgd_cal.fit(X_cal_sc, y_cal) # antrenam calibratorul pe setul de calibrare
vl_linear_inainte = min(history_linear_sgd['val_loss']) # luam cel mai bun validation loss obtinut inainte de calibrare, din timpul antrenarii modelului
vl_linear_dupa    = log_loss( # verificam cum arata log loss dupa calibrare, pe setul separat de evaluare interna
    y_val_eval, linear_sgd_cal.predict_proba(X_val_eval_sc), labels=[0, 1]
    # predict_proba(...) da probabilitatile calibrate
    # log_loss(...) masoara cat de bune sunt aceste probabilitati
)
print(f"  Val Loss inainte calibrare: {vl_linear_inainte:.4f}")
print(f"  Val Loss dupa calibrare:    {vl_linear_dupa:.4f}")
joblib.dump(linear_sgd_cal, 'models/ml_linear_sgd_calibrated.pkl') # salvam modelul calibrat pe disc
print("  Salvat: models/ml_linear_sgd_calibrated.pkl")

print("\nCalibrez SVM...") # creem si antrenam calibratorul pentru SVM
svm_cal = CalibratorPrefit(svm_model)
svm_cal.fit(X_cal_sc, y_cal)
vl_svm_inainte = min(history_svm['val_loss'])
vl_svm_dupa    = log_loss(
    y_val_eval, svm_cal.predict_proba(X_val_eval_sc), labels=[0, 1]
)
print(f"  Val Loss inainte calibrare: {vl_svm_inainte:.4f}")
print(f"  Val Loss dupa calibrare:    {vl_svm_dupa:.4f}")
joblib.dump(svm_cal, 'models/ml_svm_calibrated.pkl')
print("  Salvat: models/ml_svm_calibrated.pkl")

# ---------------------------------------------------------------
# PASUL 10: Gradient Boosting
#
# Gradient Boosting ≠ Random Forest:
#   RF: arbori INDEPENDENTI, construiti in paralel, voteaza
#   GB: arbori SECVENTIALI, fiecare corecteaza erorile precedentului
#
# De ce e mai bun:
#   RF: fiecare arbore e "bun" independent
#   GB: fiecare arbore "stie" ce a gresit cel anterior si se
#       concentreaza exact pe acele exemple
#
# Parametri:
#   n_estimators=2500: 2500 arbori secventiali
#   learning_rate=0.035: fiecare arbore contribuie 3.5% la rezultat
#     (mic = mai lent dar mai robust, mai putin overfit)
#   max_depth=4: arbori de adancime medie (nu prea complecsi)
#   subsample=0.8: fiecare arbore vede 80% din date (stochastic)
#     → reduce overfitting-ul si adauga diversitate
#   max_features='sqrt': fiecare split vede sqrt(n_features)
#   verbose=1: afiseaza progresul la fiecare 100 arbori
# ---------------------------------------------------------------
# afisam in terminal ca incepem gradient boosting 
print(f"\n{'='*60}")
print("ANTRENEZ: Gradient Boosting")
print(f"  n_estimators={GB_ESTIMATORS} | lr={GB_LR} | max_depth={GB_DEPTH}")
print("  (Poate dura 60-90 minute)")
print(f"{'='*60}")

gb_model = GradientBoostingClassifier( # creem modelul, ce e dupa egal e stabilit in preset
    n_estimators=GB_ESTIMATORS, # numarul de arbori
    learning_rate=GB_LR, # spune cat de mult contribuie fiecare arbore nou. din preset fiecare arbore contribuie 3,5%
    max_depth=GB_DEPTH, # seteaza adancimea arborilor 
    subsample=0.8, # fiecare arbore e antrenat doar pe 80% din date 
    max_features='sqrt', # la fiecare split folosim doar o parte din features-uri
    random_state=42, # facem rezultatul reproductibil
    verbose=1 # afisam progresul in terminal
)

start_gb = time.time() # timestamp inainte de antrenare. pentru calculul duratei
gb_model.fit(X_tr_sc, y_tr, # antrenarea efectiva. fit antreneaza toti arborii intr-o singura comanda
             sample_weight=sample_weight_tr) # sample_weight trimite ponderile pentru exemple, astfel incat clasa mai rara sa conteze mai mult
timp_gb_fit = time.time() - start_gb # calculeaza cat a durat antrenarea

# History simulat pentru GB (pentru grafice uniforme)
# GB nu are epoci — simulam cu staged_predict_proba
# care returneaza probabilitatile la fiecare etapa de arbori
print("\n  Calculez curbele de convergenta GB...")
history_gb  = {'train_loss': [], 'val_loss': [], # dictionarul pentru curbele de convergenta GB
               'train_acc':  [], 'val_acc':  []}
x_values_gb = [] # lista cu numerele de arbori 
best_gb_loss = np.inf # porneste de la infinit, ca sa putem gasi minimul
best_gb_iter = 1 # va memora etapa optima
gb_fara_imbun = 0 # numara cate etape trec fara imbunatatire

# folosim intregul set de train pentru curbele GB,
X_staged_tr = X_tr_sc
y_staged_tr = y_tr

# incepem parcurgerea etapelor modelului
for i, (tr_p, vl_p) in enumerate( # i este indicele etapei, pornind de la 0
    # tr_p = probabilitatile pe train, dupa etapa curenta
    #vl_p = probabilitatile pe validation, dupa etapa curenta
    zip(gb_model.staged_predict_proba(X_staged_tr), # combina doua generatoare in paralel. la fiecare iteratie primim simutan: tr_p si vl_p
        gb_model.staged_predict_proba(X_val_sc))
        # staged_predict_proba = generator care returneaza matricea de probabilitai dupa fiecare arbore adaugat
):
    # calculam metricile pe etapa curenta
    tl_s = log_loss(y_staged_tr,  tr_p) # calculeaza log loss pe subsetul de train dupa i+1 arbori
    vl_s = log_loss(y_val, vl_p) # validation loss
    ta_s = accuracy_score(y_staged_tr,  np.argmax(tr_p, axis=1)) # train accuracy
    va_s = accuracy_score(y_val, np.argmax(vl_p, axis=1)) # validation accuracy

    # salvam valorile in istoric
    history_gb['train_loss'].append(tl_s)
    history_gb['val_loss'].append(vl_s)
    history_gb['train_acc'].append(ta_s)
    history_gb['val_acc'].append(va_s)
    x_values_gb.append(i + 1)

    # urmarim daca validation loss s-a imbunatatit
    if vl_s < (best_gb_loss - 1e-4): # daca da, actualizam cel mai bun loss, salvam etapa optima, resetam contorul fara imbunatatire
        best_gb_loss = vl_s
        best_gb_iter = i + 1
        gb_fara_imbun = 0
    else: # daca nu, crestem contorul gb_fara_imbun
        gb_fara_imbun += 1

    # monitor de platou. oprim doar calculul curbelor daca vedem ca modelul a ajuns in platou si nu mai are sens sa continuam pana la capat
    if (i + 1) >= max(200, GB_PATIENCE) and gb_fara_imbun >= GB_PATIENCE:
        print(f"  [GB early stop monitor] platou la etapa {i+1}, best={best_gb_iter}")
        break

print("  Curbe GB calculate!")

# Refit model final la numarul optim de arbori (reduce overfitting + inferenta mai rapida)
if best_gb_iter < GB_ESTIMATORS:
    print(f"  Refit GB la best iter: {best_gb_iter}/{GB_ESTIMATORS}")
    gb_model = GradientBoostingClassifier(
        n_estimators=best_gb_iter,
        learning_rate=GB_LR,
        max_depth=GB_DEPTH,
        subsample=0.8,
        max_features='sqrt',
        random_state=42,
        verbose=0
    )
    start_refit = time.time()
    gb_model.fit(X_tr_sc, y_tr, sample_weight=sample_weight_tr)
    timp_gb_refit = time.time() - start_refit
else:
    timp_gb_refit = 0.0

# calculam performanta finala a modelului GB 
timp_gb = timp_gb_fit + timp_gb_refit
tr_proba_gb = gb_model.predict_proba(X_tr_sc)
vl_proba_gb = gb_model.predict_proba(X_val_sc)
tl_gb       = log_loss(y_tr,  tr_proba_gb, labels=[0, 1])
vl_gb       = log_loss(y_val, vl_proba_gb, labels=[0, 1])
ta_gb       = accuracy_score(y_tr,  gb_model.predict(X_tr_sc))
va_gb       = accuracy_score(y_val, gb_model.predict(X_val_sc))

# sumarul pentru GB
print(f"\n  Timp fit GB:    {timp_gb_fit/60:.1f} min")
if timp_gb_refit > 0:
    print(f"  Timp refit GB:  {timp_gb_refit/60:.1f} min")
print(f"  Timp total GB:  {timp_gb/60:.1f} min")
print(f"  Best iter GB:   {best_gb_iter}")
print(f"  Val Loss:       {vl_gb:.4f}")
print(f"  Val Acc:        {va_gb*100:.2f}%")
print(f"  Train Acc:      {ta_gb*100:.2f}%")

# salvam modelul final pe disc
joblib.dump(gb_model, 'models/ml_gb.pkl')
print("  Salvat: models/ml_gb.pkl")

# Feature importance GB (util pentru interpretabilitate in lucrare)
print("\n  Calculez feature importance GB...")
try:
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    feat_names = list(tfidf.get_feature_names_out()) + [ # reconstruim lista completa de nume ale features-urilor, in aceeasi ordine in care au fost puse in matrice
        'nr_url_total', 'nr_url_suspect', 'nr_url_scurtat', 'are_url',
        'lung_email', 'nr_cuvinte', 'nr_exclamare',
        'nr_simboluri_ban', 'pct_majuscule', 'nr_trigger_spam'
    ]
    # extragem importanta fiecarui feature si alegem top 20
    importances = gb_model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20] # np.argsort - ordoneaza descrescator
    top_feats = [(feat_names[i], importances[i]) for i in top_idx]

    print("  Top 20 features cele mai importante:")
    for rang, (feat, imp) in enumerate(top_feats, 1):
        print(f"    {rang:2d}. {feat:<30} {imp:.5f}")
    
    # punem numele lungi scurtate, astfel incat graficul sa ramana lizibil
    def _scurt_nume(s, max_len=42):
        s = str(s).replace('\n', ' ')
        return s if len(s) <= max_len else s[: max_len - 1] + '…'

# partea grafica:
# creem figura
# desenam barele orizontale
# scriem valorile langa ele
# setam etichetele
# salvam imaginea
    feats_plot = [_scurt_nume(f) for f, _ in top_feats]
    imps_plot = [i for _, i in top_feats]
    y_fi = np.arange(len(feats_plot))
    imp_max = max(imps_plot)

    fig_w = 11.5
    fig_h = max(7.5, 0.38 * len(feats_plot) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(AX_BG)
    ax.set_axisbelow(True)
    ax.grid(axis='x', linestyle='--', alpha=0.45, color=GRID_COL, zorder=0)
    ax.grid(False, axis='y')

    cmap_fi = plt.get_cmap('YlOrBr')
    tvals = np.linspace(0.28, 0.92, len(imps_plot))
    bar_colors = cmap_fi(tvals)

    bars_fi = ax.barh(
        y_fi, imps_plot, color=bar_colors,
        height=0.72, edgecolor='#FEF3C7', linewidth=0.6, zorder=2
    )
    for bar, val in zip(bars_fi, imps_plot):
        pad = imp_max * 0.012
        ax.text(
            min(val + pad, imp_max * 1.02),
            bar.get_y() + bar.get_height() / 2.0,
            f'{val:.4f}',
            ha='left', va='center', fontsize=8.8,
            color='#78350F', fontweight='semibold', zorder=3
        )

    ax.set_yticks(y_fi)
    ax.set_yticklabels(
        [f'{i+1:2d}.  {lab}' for i, lab in enumerate(feats_plot)],
        fontsize=9.2, color=TEXT
    )
    ax.invert_yaxis()
    ax.set_xlabel('Importanță (feature importance)', fontsize=11.5,
                  fontweight='semibold', labelpad=8)
    ax.set_title(
        'Top 20 feature-uri — Gradient Boosting',
        fontsize=13, fontweight='semibold', pad=16, color=TEXT
    )
    ax.set_xlim(0, imp_max * 1.18)
    ax.tick_params(axis='x', labelsize=9.5)
    ax.spines['left'].set_color('#E7E5E4')
    ax.spines['bottom'].set_color('#E7E5E4')
    plt.tight_layout()
    plt.savefig('grafice/ml_feature_importance.png')
    plt.close()
    print("  Salvat: grafice/ml_feature_importance.png")
except Exception as e:
    print(f"  Feature importance indisponibil: {e}")

# ---------------------------------------------------------------
# PASUL 11: EVALUARE FINALA, MODELUL FINAL SALVAT PE VALIDATION
# ---------------------------------------------------------------
print("\n" + "="*60)
print("EVALUARE FINALA PE VALIDATION, MODELUL FINAL SALVAT")
print("="*60)

# alegem cele 3 modele finale, apoi le evaluam pe acelasi set (X_val_eval_sc) pentru o comparatie corecta
modele_finale = { # dictionar cu modelele finale pe care le comparam
    'Linear-SGD': linear_sgd_cal,
    'SVM-SGD': svm_cal,
    'Gradient Boosting': gb_model
}

pred_finale = {} # clasele prezise
proba_finale = {} # probabilitatile prezise
acc_finale = {} # acuratetea finala
loss_finale = {} # log loss-ul final

for nume, model in modele_finale.items():
    proba = model.predict_proba(X_val_eval_sc) # cerem modelului sa calculeze probabilitatile pentru fiecare email din setul final de evaluare
    pred = np.argmax(proba, axis=1) # transformam probabilitatile in clase finale
# salvam probabilitatile si predictiile in dictionarele create mai sus
    proba_finale[nume] = proba
    pred_finale[nume] = pred
    acc_finale[nume] = accuracy_score(y_val_eval, pred) # calculam acuratetea modelului
    loss_finale[nume] = log_loss(y_val_eval, proba, labels=[0, 1]) # calculam log_loss-ul final

# afisam rezultatele in terminal
    print(f"{nume:<18} Acc={acc_finale[nume]*100:.2f}% | "
          f"Loss={loss_finale[nume]:.4f}")

# ---------------------------------------------------------------
# PASUL 12: GRAFICE
#
# G1 — Loss per epoca (Linear-SGD + SVM + GB, 3 subploturi)
# G2 — Accuracy per epoca (Linear-SGD + SVM + GB, 3 subploturi)
# G3 — Best Validation Loss comparativ (bare)
# G4 — Best Validation Accuracy comparativ (bare)
# G5 — Timp de antrenare comparativ (bare orizontale)
# ---------------------------------------------------------------
print("\n" + "="*60)
print("GENEREZ GRAFICELE")
print("="*60)

os.makedirs('grafice', exist_ok=True)

modele_info = {
    'Linear-SGD': {
        'history':  history_linear_sgd,
        'x_values': list(range(1, len(history_linear_sgd['train_loss'])+1)),
        'x_label':  'Epocă',
        'timp':     timp_linear_sgd,
        'culoare':  CULORI['Linear-SGD'],
        'final_val_loss': loss_finale['Linear-SGD'],
        'final_val_acc':  acc_finale['Linear-SGD'],
    },

    'SVM-SGD': {
        'history':  history_svm,
        'x_values': list(range(1, len(history_svm['train_loss'])+1)),
        'x_label':  'Epocă',
        'timp':     timp_svm,
        'culoare':  CULORI['SVM-SGD'],
        'final_val_loss': loss_finale['SVM-SGD'],
        'final_val_acc':  acc_finale['SVM-SGD'],
    },
    'Gradient Boosting': {
        'history':  history_gb,
        'x_values': x_values_gb,
        'x_label':  'Număr arbori',
        'timp':     timp_gb,
        'culoare':  CULORI['Gradient Boosting'],
        'final_val_loss': loss_finale['Gradient Boosting'],
        'final_val_acc':  acc_finale['Gradient Boosting'],
    }
}
names = list(modele_info.keys())

# ===== GRAFIC 1: Loss per epoca =====
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.patch.set_facecolor(BG)
fig.suptitle(
    f'Convergența ML — Log Loss per epocă  (LR={ETA0}, invscaling)',
    fontsize=13, fontweight='semibold', color=TEXT, y=1.02
)
for idx, (nm, info) in enumerate(modele_info.items()):
    ax = axes[idx]
    deseneaza_subplot(
        ax, info['x_values'],
        info['history']['train_loss'],
        info['history']['val_loss'],
        info['culoare'], info['x_label'],
        ylabel='Log Loss',
        legend_loc='upper right',
        legend_anchor=(0.99, 0.99)
    )
    ax.set_title(nm, fontweight='semibold',
                 color=info['culoare'], pad=10)
plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.92])
plt.savefig('grafice/ml_convergenta_loss.png')
plt.close()
print("Salvat: grafice/ml_convergenta_loss.png")

# ===== GRAFIC 2: Accuracy per epoca =====
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.patch.set_facecolor(BG)
fig.suptitle(
    f'Convergența ML — Accuracy per epocă  (LR={ETA0}, invscaling)',
    fontsize=13, fontweight='semibold', color=TEXT, y=1.02
)
for idx, (nm, info) in enumerate(modele_info.items()):
    ax = axes[idx]
    deseneaza_subplot(
        ax, info['x_values'],
        info['history']['train_acc'],
        info['history']['val_acc'],
        info['culoare'], info['x_label'],
        ylabel='Accuracy', fmt_y=True, show_max=True,
        annotate_below=(nm == 'Gradient Boosting'),
        legend_loc='lower right',
        legend_anchor=(0.99, 0.01)
    )
    ax.set_title(nm, fontweight='semibold',
                 color=info['culoare'], pad=10)
plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.92])
plt.savefig('grafice/ml_convergenta_accuracy.png')
plt.close()
print("Salvat: grafice/ml_convergenta_accuracy.png")

# ===== GRAFIC 3: Matrice de confuzie (model final salvat) =====
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.patch.set_facecolor(BG)
fig.suptitle('Matrice de confuzie — model final salvat (validation)',
             fontsize=13, fontweight='semibold', color=TEXT, y=1.02)
lab = ['Ham (0)', 'Maliţios (1)']
etichete_cm = [['TN', 'FP'], ['FN', 'TP']]
for idx, nm in enumerate(names):
    ax = axes[idx]
    cm = confusion_matrix(y_val_eval, pred_finale[nm], labels=[0, 1])
    im = ax.imshow(cm, cmap='Blues')
    
    # Contur chenar pentru claritate
    for spine in ax.spines.values():
        spine.set_edgecolor('#1E293B')
        spine.set_linewidth(0.5)
    
    ax.set_title(nm, fontsize=13, fontweight='semibold',
                 color=modele_info[nm]['culoare'], pad=12)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(lab, fontsize=10)
    ax.set_yticklabels(lab, fontsize=10)
    ax.set_xlabel('Predicție', fontsize=10, fontweight='semibold')
    ax.set_ylabel('Adevărat', fontsize=10, fontweight='semibold')
    
    cm_max = cm.max() if cm.max() > 0 else 1
    cm_total = cm.sum()
    for i in range(2):
        for j in range(2):
            # culoare adaptiva: alb pe celule inchise, negru pe celule deschise
            txt_color = 'white' if cm[i, j] > cm_max * 0.5 else '#1E293B'
            pct = (cm[i, j] / cm_total * 100) if cm_total > 0 else 0
            ax.text(j, i, f"{cm[i, j]}\n{etichete_cm[i][j]}\n({pct:.1f}%)",
                    ha='center', va='center', color=txt_color,
                    fontsize=9.5, fontweight='bold', linespacing=1.3)
    
    acc_txt = acc_finale[nm] * 100
    ax.text(0.5, -0.18, f'Acc final: {acc_txt:.2f}%',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=10, color=TEXT, fontweight='semibold')
plt.tight_layout()
plt.savefig('grafice/ml_matrice_confuzie_validation.png')
plt.close()
print("Salvat: grafice/ml_matrice_confuzie_validation.png")

# ===== GRAFIC 4: Validation Loss (model final salvat) =====
fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(AX_BG)
ax.set_axisbelow(True)

best_vl  = [modele_info[n]['final_val_loss'] for n in names]
culori_b = [modele_info[n]['culoare'] for n in names]
x_pos    = np.arange(len(names))

bars = ax.bar(x_pos, best_vl, color=culori_b,
              alpha=0.88, edgecolor='white',
              linewidth=0.8, zorder=3, width=0.42)
for bar, val in zip(bars, best_vl):
    ax.text(bar.get_x() + bar.get_width()/2.,
            val + max(best_vl)*0.015,
            f'{val:.4f}',
            ha='center', va='bottom',
            fontsize=12, fontweight='bold', color=TEXT)

ax.set_xlabel('Model', fontsize=12, fontweight='semibold')
ax.set_ylabel('Validation Loss (model final)', fontsize=12,
              fontweight='semibold')
ax.set_title('Comparație Validation Loss — model final salvat',
             fontweight='semibold', pad=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(names, fontsize=11)
ax.set_ylim(0, max(best_vl) * 1.28)
plt.tight_layout()
plt.savefig('grafice/ml_best_val_loss.png')
plt.close()
print("Salvat: grafice/ml_best_val_loss.png")

# ===== GRAFIC 5: Validation Accuracy (model final salvat) =====
fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(AX_BG)
ax.set_axisbelow(True)

best_va = [modele_info[n]['final_val_acc'] for n in names]

bars = ax.bar(x_pos, best_va, color=culori_b,
              alpha=0.88, edgecolor='white',
              linewidth=0.8, zorder=3, width=0.42)
for bar, val in zip(bars, best_va):
    ax.text(bar.get_x() + bar.get_width()/2.,
            val + 0.0008,
            f'{val*100:.2f}%',
            ha='center', va='bottom',
            fontsize=12, fontweight='bold', color=TEXT)

ax.set_xlabel('Model', fontsize=12, fontweight='semibold')
ax.set_ylabel('Validation Accuracy (model final)', fontsize=12,
              fontweight='semibold')
ax.set_title('Comparație Validation Accuracy — model final salvat',
             fontweight='semibold', pad=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(names, fontsize=11)
ymin = max(0, min(best_va) - 0.03)
ax.set_ylim(ymin, 1.005)
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f'{v*100:.1f}%'))
plt.tight_layout()
plt.savefig('grafice/ml_best_val_accuracy.png')
plt.close()
print("Salvat: grafice/ml_best_val_accuracy.png")

# ===== GRAFIC 6: Timp de antrenare =====
fig, ax = plt.subplots(figsize=(10.5, 5.2))
fig.patch.set_facecolor(BG)
ax.set_facecolor(AX_BG)
ax.set_axisbelow(True)
ax.grid(axis='x', linestyle='--', alpha=0.45, color=GRID_COL, zorder=0)
ax.grid(False, axis='y')

timpi = [modele_info[n]['timp'] for n in names]
t_max = max(timpi)
y_pos = np.arange(len(names))

bars = ax.barh(
    y_pos, timpi, color=culori_b,
    height=0.58, edgecolor='#F1F5F9', linewidth=1.0, zorder=2
)
for bar, t, nm in zip(bars, timpi, names):
    w = bar.get_width()
    cy = bar.get_y() + bar.get_height() / 2.0
    x_right = w + t_max * 0.018
    # Pe bare foarte scurte (SGD), eticheta albă nu încape — mutăm totul la dreapta
    if w >= t_max * 0.12:
        ax.text(
            w / 2.0, cy,
            f'{t/60:.1f} min',
            ha='center', va='center',
            fontsize=10.5, fontweight='bold', color='white',
            zorder=4
        )
        ax.text(
            x_right, cy,
            f'({t:.0f} s)',
            ha='left', va='center',
            fontsize=9.5, color='#64748B', zorder=3
        )
    else:
        ax.text(
            x_right, cy,
            f'{t/60:.1f} min  ({t:.0f} s)',
            ha='left', va='center',
            fontsize=10.5, fontweight='bold', color=TEXT,
            zorder=3
        )

ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=11.5, fontweight='semibold')
ax.invert_yaxis()
ax.set_xlabel('Timp (secunde)', fontsize=12, fontweight='semibold')
ax.set_title('Timp de antrenare — comparație modele',
             fontweight='semibold', pad=16, fontsize=13)
ax.set_xlim(0, t_max * 1.28)
ax.tick_params(axis='x', labelsize=10)
plt.tight_layout()
plt.savefig('grafice/ml_timp_antrenare.png')
plt.close()
print("Salvat: grafice/ml_timp_antrenare.png")

# ---------------------------------------------------------------
# SUMAR FINAL
# ---------------------------------------------------------------
print("\n" + "="*65)
print("SUMAR FINAL — ANTRENARE ML")
print("="*65)
print(f"{'Model':<22} {'FinalValLoss':>12} {'FinalValAcc':>12} "
      f"{'Timp':>10}")
print("-"*60)
status_general = {}
for nm, info in modele_info.items():
    bvl = info['final_val_loss']
    bva = info['final_val_acc']
    bta = max(info['history']['train_acc'])
    t   = info['timp']

    gap_pct = (bta - bva) * 100.0
    if gap_pct > 2.0:
        status = "OVERFIT risk"
    elif bva < 92.0 / 100.0:
        status = "UNDERFIT risk"
    else:
        status = "OK"
    status_general[nm] = (status, gap_pct, bta, bva)

    print(f"{nm:<22} {bvl:>11.4f}  {bva*100:>10.2f}%  "
          f"{t/60:>7.1f} min")



print("\nDiagnostic generalizare (automat):")
for nm in names:
    status, gap_pct, bta, bva = status_general[nm]
    print(f"  - {nm:<18} {status:<13} | gap train-val={gap_pct:+.2f}pp "
          f"(train={bta*100:.2f}% vs val={bva*100:.2f}%)")
print("  Praguri: OVERFIT daca gap > 2.0pp | UNDERFIT daca ValAcc < 92%")

print(f"\nDupa calibrare:")
print(f"  Linear-SGD  Val Loss: {vl_linear_inainte:.4f} → {vl_linear_dupa:.4f}")
print(f"  SVM         Val Loss: {vl_svm_inainte:.4f} → {vl_svm_dupa:.4f}")

print("\nModele salvate:")
print("  models/ml_linear_sgd.pkl             (Linear-SGD necalibrat)")
print("  models/ml_linear_sgd_calibrated.pkl  (Linear-SGD calibrat)")
print("  models/ml_svm_sgd.pkl                (SVM necalibrat)")
print("  models/ml_svm_calibrated.pkl         (SVM calibrat)")
print("  models/ml_gb.pkl                     (Gradient Boosting)")
print("  models/feature_scaler.pkl            (MaxAbsScaler)")

print("\nGrafice salvate: grafice/ml_*.png (incl. ml_feature_importance.png)")

# ---------------------------------------------------------------
# Salvare rezultate pentru comparație cu DL
# ---------------------------------------------------------------
print("\nSalvez rezultate pentru comparație cu DL...")
os.makedirs('results', exist_ok=True)

# Creăm DataFrame cu rezultatele finale
ml_results_data = []
for nm in names:
    ml_results_data.append({
        'model': nm,
        'val_accuracy': acc_finale[nm],
        'val_loss': loss_finale[nm],
        'train_time_sec': modele_info[nm]['timp']
    })

df_ml_summary = pd.DataFrame(ml_results_data)
df_ml_summary.to_csv('results/ml_validation_summary.csv', index=False)
print("Salvat: results/ml_validation_summary.csv")

print("\n=== ANTRENARE ML COMPLETA! ===")