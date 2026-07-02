# =============================================================
# TRAIN_DL_TOTAL.PY — VERSIUNEA FINALA
# =============================================================
# Scopul acestui fisier:
#   Antrenare Deep Learning pentru 2 arhitecturi neuronale:
#   1. BiLSTM — Bidirectional Long Short-Term Memory
#      → intelege ordinea si contextul cuvintelor din email
#      → procesesaza secventa de la stanga la dreapta SI
#        de la dreapta la stanga simultan
#   2. CNN 1D — Convolutional Neural Network pe secvente text
#      → detecteaza tipare locale (expresii spam) oriunde in email
#      → rapid, paralel, complementar LSTM
#
# Grafice generate:
#   - Loss + Accuracy per epoca (identice cu ML)
#   - Confusion Matrix
#   - Comparatie ML vs DL (grafic special)
#   - Timp antrenare
#
# =============================================================

import numpy as np # pentru calcule matriciale
import pandas as pd # pentru date tabelare. Ca sa citim CSV-uri si functia de medie din smooth
import matplotlib.pyplot as plt # pentru grafice
import matplotlib.ticker as mticker # formateaza axele
import joblib # salveaza si incarca obiecte Python pe disk. Folosit pentru scaler si tokenizer
import time # masoara durata antrenarii
import os # operatii pe sistemul de fisiere
import warnings # suprima mesajele de avertisment din sklearn si TensorFlow care aglomereaza terminalul
import random # bibblioteca standar Python pentru numere aleatoare - necesara pentru seed
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # afiseaza doar erorile, nu mesajele informative despre harcware si memorie

# TensorFlow / Keras
import tensorflow as tf # ruleaza modelele neuronale
from tensorflow import keras # importa keras. keras ofera o interfata mai simpla pentru retelele neuronale
from tensorflow.keras import layers, Model, Input, regularizers # importa elementele principale pentru construirea arhitecturilor
from tensorflow.keras.preprocessing.text import Tokenizer # importa tokenizer-ul care transforma cuvintele in indici
from tensorflow.keras.preprocessing.sequence import pad_sequences # importa functia care aduce secventele la aceeasi lungime
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # importa cele 3 callback-uri pentru controlul antrenarii
)

from sklearn.metrics import accuracy_score, log_loss, confusion_matrix # importa metricile pentru evaluare
from sklearn.utils.class_weight import compute_class_weight # importa functia care calculeaza ponderile pentru clase. Utila la dezechilibru intre spam si ham

# ---------------------------------------------------------------
# SEED pentru reproductibilitate completă
# ---------------------------------------------------------------
SEED = 42 # defineste valoarea fixa pentru generarea aleatoare
os.environ['PYTHONHASHSEED'] = str(SEED) # fixeaza comportamentul hash-urilor Python
random.seed(SEED) # fixeaza generatorul aleator din python
np.random.seed(SEED) # fixeaza generatorul aleator din NumPy
tf.keras.utils.set_random_seed(SEED) # fixeaza generatorul aleator folosit de TensorFlow si Keras
# Seed-urile de mai sus asigură reproductibilitate practică suficientă
# enable_op_determinism() ar bloca paralelismul CPU → timp 3x mai mare

# ---------------------------------------------------------------
# CONFIGURARE CPU
# Fara GPU — configuram TensorFlow sa foloseasca CPU optimal
# ---------------------------------------------------------------
tf.config.threading.set_intra_op_parallelism_threads(0)  # 0 = auto - detect; decide câte thread-uri folosește PENTRU o singură operație mare
tf.config.threading.set_inter_op_parallelism_threads(0) # decide câte operații independente pot rula simultan.
print(f"TensorFlow versiune: {tf.__version__}") # afiseaza versiunea TensorFlow
print(f"Dispozitive disponibile: {[d.name for d in tf.config.list_logical_devices()]}") # afiseaza dispozitivele disponibile (CPU sau GPU)

# ---------------------------------------------------------------
# STILUL GLOBAL AL GRAFICELOR
# Identic cu train_ml_total.py pentru consistenta vizuala
# ---------------------------------------------------------------
CULORI_DL = {
    'BiLSTM': '#0891B2',   # cyan-700
    'CNN1D':  '#7C3AED',   # violet-700
}

# Culorile ML pentru graficul comparativ
CULORI_ML = {
    'Linear-SGD':        '#0F766E',
    'SVM-SGD':           '#4338CA',
    'Gradient Boosting': '#C2410C',
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
# HIPERPARAMETRI DL
#
# Parametri optimizați pentru PERFORMANȚĂ MAXIMĂ (nu viteză):
#
# MAX_WORDS = 20000
#   Vocabular complet — nu pierdem cuvinte importante.
#   Mai mult context semantic = performanță mai bună.
#
# MAX_LEN = 200
#   Secvențe complete — captăm tot contextul emailului.
#
# EMBED_DIM = 128
#   Embedding bogat — reprezentări semantice mai bogate.
#
# EPOCHS_DL = 60
#   Suficient pentru convergență completă cu LR mai mic.
#   Early stopping va opri când e optim.
#
# BATCH_SIZE = 128
#   Batch mai mare cu LR=0.0005 → gradient mai stabil → curbe mai netede.
#   Reduce zgomotul în convergență, similar cu ML.
#
# LEARNING_RATE = 0.0005
#   LR mai mic decât standard (0.001) → convergență mai lentă dar mai stabilă.
#   Reduce overfit-ul, îmbunătățește generalizarea.
#   ReduceLROnPlateau va ajusta automat când e nevoie.
# ---------------------------------------------------------------
MAX_WORDS     = 20000 # modelul pastreaza maximum 20 000 de cuvinte in vocabular
MAX_LEN       = 200 # fiecare model este adus la 200 de tokens
EMBED_DIM     = 128 # fiecare cuvant devine un vector de 128 de valori
EPOCHS_DL     = 60 # modelul poate rula maxim 60 de epoci
BATCH_SIZE    = 128 # modelul proceseaza 128 de emailuri intr-un batch
LEARNING_RATE = 0.0005 # adam porneste cu learning_rate 0.0005, deci antrenarea va fi mai stabila
NUMERIC_COLS  = [ # defineste cele 10 caracteristici numerice folosite impreuna cu textul
    'nr_url_total', 'nr_url_suspect', 'nr_url_scurtat', 'are_url',
    'lung_email', 'nr_cuvinte', 'nr_exclamare',
    'nr_simboluri_ban', 'pct_majuscule', 'nr_trigger_spam'
] 

print("=" * 65)
print("TRAIN_DL_TOTAL.PY — Antrenare Deep Learning")
print("=" * 65)
print(f"\nHiperparametri DL:")
print(f"  Vocabular:    {MAX_WORDS:,} cuvinte") # afiseaza nr maxim de cuvinte
print(f"  Lungime seq:  {MAX_LEN} tokens (padding/truncate)") # afiseaza lungimea secventei
print(f"  Embedding:    {EMBED_DIM} dimensiuni") # dimensiunea embedding-urilor
print(f"  Epoci max:    {EPOCHS_DL}") # numarul maxim de epoci
print(f"  Batch size:   {BATCH_SIZE}") # batch_size-ul
print(f"  LR initial:   {LEARNING_RATE}") # learning rate-ul initial

# Cream folderele pentru salvare modele si grafice
os.makedirs('models', exist_ok=True)
os.makedirs('grafice', exist_ok=True)

# ---------------------------------------------------------------
# PASUL 1: Incarcam datele CSV produse de preprocess_total.py
#
# DL NU foloseste matricea TF-IDF (X_train_sparse.npz).
# Are nevoie de textul procesat in format CSV pentru propria
# tokenizare Keras, diferita de TF-IDF.
#
# De ce tokenizare diferita?
#   TF-IDF:  transforma textul in frecvente → pierde ordinea
#   Keras:   transforma textul in indici → pastreaza ordinea
#   Aceasta ordine e esentiala pentru LSTM si CNN 1D.
# ---------------------------------------------------------------
print("\n" + "="*60)
print("PASUL 1: Incarcare date CSV")
print("="*60)

df_train = pd.read_csv('data/train/train_data.csv') # citeste setul de antrenare din CSV

# Curatam valorile lipsa — pot aparea in textul procesat
df_train['text_procesat'] = df_train['text_procesat'].fillna('') # inlcuieste valorile lipsa din coloana text_procesat cu text gol

print(f"Train: {len(df_train):,} emailuri") # nr de emailuri din train
print("Test set: sigilat pentru test_total.py (nu se incarca aici)")

# Extragem textul si etichetele
X_text_train = df_train['text_procesat'].values # extrage textele procesate
y_train      = df_train['label'].values.astype(np.float32) # extrage etichetele si le convertaete la float32

# Extragem features numerice — cele 10 din preprocess
X_num_train  = df_train[NUMERIC_COLS].values.astype(np.float32)

print(f"Features numerice: {X_num_train.shape[1]} coloane") # afiseaza cate caracteristici numerice exista

# ---------------------------------------------------------------
# PASUL 2: Validation split 90/10
#
# Identic cu train_ml_total.py pentru comparabilitate.
# Folosim random_state=42 pentru acelasi split.
# ---------------------------------------------------------------
print("\n" + "="*60)
print("PASUL 2: Validation split 90/10")
print("="*60)

from sklearn.model_selection import train_test_split # importa functia pentru impartirea datelor
# impartem datele in train si validation, 10% devine validation; _num_val # caracteristici numerice pentru validare
(X_text_tr , X_text_val,
 X_num_tr,  X_num_val,
 y_tr,      y_val) = train_test_split(
    X_text_train, X_num_train, y_train,
    test_size=0.1,
    random_state=42, # asigura aceeasi impartire la fiecare rulare
    stratify=y_train # pastram proportia claselor in train si validation 
)

print(f"Train efectiv: {len(X_text_tr):,} emailuri") # nr de mailuri din train
print(f"Validation:    {len(X_text_val):,} emailuri") # nr de mailuri din validation

# ---------------------------------------------------------------
# PASUL 3: Normalizare features numerice
#
# Cream un scaler NOU doar pentru cele 10 features numerice.
# Nu putem folosi feature_scaler.pkl de la ML deoarece acela
# a fost antrenat pe 15.010 features (TF-IDF + numerice).
# Aici avem doar 10 features numerice.
#
# De ce scaler separat?
#   ML: 15.010 features (15.000 TF-IDF + 10 numerice) → MaxAbsScaler
#   DL: 10 features numerice (textul e tokenizat separat) → MaxAbsScaler nou
#   Dimensiuni diferite → scalere separate, dar aceeași metodă (MaxAbsScaler)
# ---------------------------------------------------------------
print("\n" + "="*60)
print("PASUL 3: Normalizare features numerice")
print("="*60)
from sklearn.preprocessing import MaxAbsScaler as MaxAbsScalerDL # importa MaxAbsSCaler si ii da numele MaxAbsScalerDL
scaler_dl = MaxAbsScalerDL() # creeaza scaler-ul
X_num_tr_sc  = scaler_dl.fit_transform(X_num_tr).astype(np.float32) # invata valorile maxime din train si transforma train-ul
X_num_val_sc = scaler_dl.transform(X_num_val).astype(np.float32) # transforma validation folosind aceleasi valori invatate din train
joblib.dump(scaler_dl, 'models/dl_numeric_scaler.pkl') # salveaza scalerul pentru testare si utilizare ulterioara
print("Scaler DL nou creat si salvat: models/dl_numeric_scaler.pkl")
print("Features numerice normalizate la [-1, 1] prin MaxAbsScaler") # afiseaza ca normalizarea s-a realizat

# ---------------------------------------------------------------
# PASUL 4: Tokenizare Keras
#
# Tokenizarea = procesul de transformare a textului in indici.
#
# fit_on_texts(X_text_tr):
#   Construieste vocabularul din textul de antrenare.
#   Numara frecventa fiecarui cuvant.
#   Atribuie indici: cuvantul cel mai frecvent → index 1,
#                    al doilea → index 2, etc.
#
# texts_to_sequences:
#   "free money click" → [342, 891, 127]
#   (fiecare cuvant devine indexul sau din vocabular)
#
# pad_sequences(maxlen=200, padding='post', truncating='post'):
#   'post' = adaugam/trunchiim la SFARSITUL secventei.
#   Email de 50 cuvinte → [342, 891, ..., 0, 0, ..., 0] (200 elem)
#   Email de 300 cuvinte → primele 200 cuvinte, restul ignorat.
#
#   De ce padding='post' si nu 'pre'?
#   LSTM si CNN procesesaza de la inceput.
#   Padding la sfarsit = informatia reala e la inceput → mai bine.
# ---------------------------------------------------------------
print("\n" + "="*60)
print("PASUL 4: Tokenizare Keras")
print("="*60)

tokenizer = Tokenizer( # creem tokenizer-ul Keras
    num_words=MAX_WORDS,   # pastreaza doar top 20.000 cuvinte
    oov_token='<OOV>'      # token special pentru cuvinte necunoscute
                           # OOV = Out Of Vocabulary
                           # cuvintele din test care nu apar in train
                           # primesc indexul <OOV> in loc de eroare
)

# fit DOAR pe train — previne data leakage; construieste vocabularul doar din train efectiv
tokenizer.fit_on_texts(X_text_tr)

# Convertim textul la secvente de indici
seq_tr  = tokenizer.texts_to_sequences(X_text_tr) # pentru train
seq_val = tokenizer.texts_to_sequences(X_text_val) # pentru validation

# Padding la lungime fixa
X_seq_tr  = pad_sequences(seq_tr,  maxlen=MAX_LEN,
                           padding='post', truncating='post') # aducem secventele de train la lungimea 200
                           # maxlen stabileste lungimea maxima
                           # padding = post adauga zero-uri la final
                           # truncating = post taie textul dupa primele 200 de tokens
X_seq_val = pad_sequences(seq_val, maxlen=MAX_LEN, # aducem secventele de validation la lungimea 200 
                           padding='post', truncating='post')

vocab_size = min(MAX_WORDS, len(tokenizer.word_index)) + 1 # calculeaza dimensiunea reala a vocabularului folosit de model. trebuie sa o stim pentru embedding layer
print(f"Vocabular construit: {len(tokenizer.word_index):,} cuvinte unice")
print(f"Vocabular folosit:   {vocab_size:,} (top {MAX_WORDS:,} + OOV)") # afiseaza cate cuvinte unice exista
print(f"Shape secvente train: {X_seq_tr.shape}")
print(f"  = {X_seq_tr.shape[0]:,} emailuri × {X_seq_tr.shape[1]} tokens")

# Salvam tokenizer-ul pentru test_total.py si productie
os.makedirs('models', exist_ok=True)
joblib.dump(tokenizer, 'models/dl_tokenizer.pkl')
print("Salvat: models/dl_tokenizer.pkl") # confirma salvarea 

# ---------------------------------------------------------------
# PASUL 5: Ponderi de clasa
#
# Identic cu train_ml_total.py — echilibrare clase.
# ---------------------------------------------------------------
classes = np.array([0, 1]) # defineste clasele posibile 
cw_array = compute_class_weight('balanced', classes=classes, y=y_tr) # calculeaza ponderile claselor
cw_dict  = dict(zip(classes, cw_array)) # transforma ponderile intr-un dictionar pentru Keras
print(f"\nPonderi clase: ham={cw_dict[0]:.3f}, spam={cw_dict[1]:.3f}")

# ---------------------------------------------------------------
# PASUL 6: Functii helper pentru grafice
# ---------------------------------------------------------------

def smooth(values, window=5): # defineste o functie pentru netezirea curbelor
    """Medie mobila pentru netezirea curbelor."""
    s = pd.Series(values, dtype=float) # transforma lista de valori intr-o serie Pandas
    return s.rolling(window=window, min_periods=1).mean().tolist() # calculeaza media mobila si lista netezita pe 5 valori consecutive


def deseneaza_convergenta(ax, history_keras, culoare, xl='Epocă',
                           ylabel='', fmt_y=False, show_max=False):
                           # ax - axa pe care se deseneaza
                           # history_keras - obiectul returnat de model.fit
                           # culoare - culoarea modelului
                           # fmt_y = False - controleaza daca axa Y se formeaza ca procent
                           # show_max = False - indica daca se deseneaza accuracy
    """
    Deseneaza curbele de convergenta dintr-un history Keras.
    
    IMPORTANT: Marchează întotdeauna epoca cu val_loss MINIM,
    indiferent dacă graficul e de loss sau accuracy.
    Asta asigură consistență metodologică — modelul final e ales după val_loss.
    """
    if show_max: # verifica daca graficul este pentru accuracy
        raw_tr = history_keras.history['accuracy']
        raw_vl = history_keras.history['val_accuracy']
    else: # daca nu, e pentru loss
        raw_tr = history_keras.history['loss']
        raw_vl = history_keras.history['val_loss']

    xv = list(range(1, len(raw_tr) + 1)) # creeaza lista epocilor
    w     = max(2, min(7, len(xv) // 5)) # alege fereastra pentru netezire
    sm_tr = smooth(raw_tr, window=w) # netezeste curba train
    sm_vl = smooth(raw_vl, window=w) # netezeste curba validation

    # Curbe raw cu transparenta mai mare pentru a vedea zgomotul natural
    ax.plot(xv, raw_tr, color=culoare,  alpha=0.2, linewidth=0.9) # deseneaza curba bruta de train
    ax.plot(xv, raw_vl, color=COL_VAL, alpha=0.2, linewidth=0.9) # deseneaza curba bruta de validation

    ltr, = ax.plot(xv, sm_tr, color=culoare, linewidth=2.5,
                   label='Train (smoothed)', zorder=5) # deseneaza curba netezita de train
    lvl, = ax.plot(xv, sm_vl, color=COL_VAL, linewidth=2.5,
                   linestyle='--', label='Validation (smoothed)', zorder=5) # deseneaza curba netezita de validation


    # CONSISTENȚĂ: Marchează întotdeauna best val_loss epoch
    # (nu best val_accuracy), fiindcă modelul final e ales după val_loss
    best_loss_idx = int(np.argmin(history_keras.history['val_loss'])) # gaseste epoca unde validation loss este minim. Modelul este ales dupa validation loss, nu dupa accuracy
    opt_idx = best_loss_idx # seteaza epoca optima (index)
    opt_val = raw_vl[opt_idx] # ia valoarea corespunzatoare epocii optime (valoarea)
    
    # Eticheta depinde de tipul graficului
    if show_max: # daca eticheta este de accuracy, eticheta se formateaza ca procent
        lbl = f'{opt_val*100:.2f}%' # eticheta procentuala
    else: # pentru loss, eticheta ramane numar zecimal
        lbl = f'{opt_val:.4f}'

    ax.scatter(xv[opt_idx], opt_val, color=COL_VAL, # marcheaza epoca optima pe grafic
               s=100, zorder=10, edgecolors='white', linewidth=1.5)
               # xv[opt_idx] - coordonata X
               # opt_val - coordonata y
               # s = 100 dimensiunea punctului
               # zorder = 10 - ordinea de afisare ( punctul va fi desenat deasupra altor elemente)
               # edgecolors - culoarea conturului
               # linewidth = grosimea punctului

    # Pozitionam eticheta deasupra sau dedesubt in functie de spatiu
    y_range = max(raw_vl) - min(raw_vl) if len(raw_vl) > 1 else 1 # calculeaza variatia valorilor de validation
    offset_y = 10 if opt_val < (min(raw_vl) + y_range * 0.75) else -16 # alege pozitia etichetei
    va_txt   = 'bottom' if offset_y > 0 else 'top'
    ax.annotate(lbl, xy=(xv[opt_idx], opt_val), # scrie eticheta langa punctul optim
                xytext=(10, offset_y), textcoords='offset points',
                fontsize=9, color=COL_VAL, fontweight='bold',
                va=va_txt,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=COL_VAL, alpha=0.95, linewidth=1.0))

    # Marcam si epoca optima pe axa X cu o linie verticala subtire
    ax.axvline(x=xv[opt_idx], color=COL_VAL, linestyle=':',
               linewidth=1.2, alpha=0.5, zorder=2)

    ax.set_xlabel(xl, fontsize=11, fontweight='semibold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='semibold')
    if fmt_y:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f'{v*100:.1f}%'))
    
    # Legenda cu pozitie adaptiva
    if show_max:
        leg_loc = 'lower right'
        leg_anchor = (0.99, 0.01)
    else:
        leg_loc = 'upper right'
        leg_anchor = (0.99, 0.99)
    
    leg = ax.legend(handles=[ltr, lvl], fontsize=9, loc=leg_loc,
                    bbox_to_anchor=leg_anchor, frameon=True,
                    fancybox=True, framealpha=0.97,
                    edgecolor='#CBD5E1', borderaxespad=0.4)
    leg.set_zorder(2000)
    try:
        leg.set_in_layout(False)
    except Exception:
        pass
    ax.set_facecolor(AX_BG)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)

# ---------------------------------------------------------------
# PASUL 7: Arhitectura BiLSTM
#
# ARHITECTURA COMPLETA:
#
# Input 1: secventa de text (200 tokens)
#   → Embedding(vocab_size, 128, mask_zero=True)
#       Transforma fiecare index intr-un vector de 128 numere.
#       mask_zero=True: ignora padding (zerouri) in calcule LSTM.
#       Greutatile embedding sunt ANTRENABILE — se ajusteaza.
#   → SpatialDropout1D(0.35)
#       Varianta speciala de Dropout pentru secvente.
#       Dezactiveaza 35% din dimensiunile embedding-ului
#       (nu token-uri individuale, ci dimensiuni intregi).
#       Previne co-adaptarea feature-urilor embedding.
#   → Bidirectional(LSTM(80, return_sequences=True))
#       LSTM cu 80 unitati in fiecare directie.
#       return_sequences=True = returneaza output la FIECARE pas,
#       nu doar la final. Necesar pentru al doilea LSTM.
#       Bidirectional: dubleaza la 160 outputuri (80 fwd + 80 bwd).
#   → Dropout(0.35)
#   → Bidirectional(LSTM(40))
#       Al doilea LSTM, mai mic (40 unitati per directie = 80 total).
#       return_sequences=False (implicit) = returneaza doar ultimul pas.
#       Rezuma intreaga secventa intr-un vector de 80 numere.
#   → output_text: vector de 80 numere
#
# Input 2: features numerice (10 valori)
#   → Dense(32, activation='relu')
#       Strat fully-connected cu 32 neuroni si activare ReLU.
#       Invata combinatii neliniare ale celor 10 features.
#   → BatchNormalization()
#       Normalizeaza outputul stratului Dense.
#       Stabilizeaza antrenarea, permite LR mai mare.
#   → output_numeric: vector de 32 numere
#
# Concatenare: [output_text (80) | output_numeric (32)] = 112 numere
#   → Dense(56, activation='relu')
#   → Dropout(0.4)
#   → Dense(1, activation='sigmoid')
#       Un singur neuron de iesire cu sigmoid.
#       Output: probabilitate spam ∈ (0, 1)
# ---------------------------------------------------------------

def construieste_bilstm(vocab_size, embed_dim, max_len, n_numeric): # definim functia care construieste modelul BiLSTM
    """
    Construieste arhitectura BiLSTM cu input dublu:
    text (secventa) + features numerice.

    Returneaza un model Keras compilat, gata de antrenare.
    """

    # --- Ramura text ---
    input_text = Input( # creeaza intrarea textuala, forma este de 200 tokens
        shape=(max_len,),
        name='input_text'
        # shape=(200,) = secventa de 200 indici intregi
    )

    x = layers.Embedding( # adaugam stratul de embedding
        input_dim=vocab_size, # dimensiunea vocabularului
        output_dim=embed_dim,    # dimensiunea vectorului fiecarui cuvant (128 dimensiuni)
        mask_zero=True, # ignoram zerourile de padding in BiLSTM
        embeddings_regularizer=regularizers.l2(1e-5), # aplica regularizare L2 pe embedding-uri
        # penalizeaza valorile mari din matricea embedding pentru a preveni overfitting-ul
        name='embedding_lstm' # da nume stratului
    )(input_text)
    # shape: (batch, 200, 128)

    x = layers.SpatialDropout1D( # aplica SpatialDropout. Dezactiveaza 35% din dimensiunile embedding-ului
        rate=0.35,               # regularizare moderată
        name='spatial_dropout'
    )(x)

    x = layers.Bidirectional( # adauga un strat LSTM bidirectional
        layers.LSTM( # creeaza un LSTM cu 80 de unitati pe directie
            units=80,            # echilibrat: 80 (160 bidirectional)
            return_sequences=True, # returneaza iesirea pentru fiecare token. e necesar pentru al doilea LSTM
            dropout=0.3, # aplica dropout pe intrarile LSTM
            recurrent_dropout=0.2, # aplica dropout pe conexiunile recurente
            kernel_regularizer=regularizers.l2(1e-4), # aplica l2 pe greutatile principale ale LSTM
            recurrent_regularizer=regularizers.l2(1e-4) # aplica L2 pe greutatile recurente
        ),
        name='bilstm_1' # numeste primul strat BiLSTM
    )(x)
    # Output shape: (batch, 200, 160) — 160 = 80×2

    x = layers.Dropout(0.35, name='dropout_lstm1')(x) # aplica dropout dupa primul BiLSTM

    x = layers.Bidirectional( # adaugam al doilea strat BiLSTM
        layers.LSTM( # creeaza un LSTM cu 40 de unitati pe directie
            units=40,            # echilibrat: 40 (80 bidirectional)
            return_sequences=False, # returneaza doar reprezentarea finala a secventei
            dropout=0.3,
            recurrent_dropout=0.2,
            kernel_regularizer=regularizers.l2(1e-4),
            recurrent_regularizer=regularizers.l2(1e-4)
        ),
        name='bilstm_2'
    )(x)
    # Output shape: (batch, 80) — 80 = 40×2

    output_text = x  # vector de 80 numere per email
                    # iesirea are 80 de valori, 40 inainte si 40 inapoi
                    # salveaza reprezentarea textului

    # --- Ramura features numerice ---
    input_numeric = Input( # creeaza intrarea pentru caracteristicile numerice
        shape=(n_numeric,), # forma este 10, deoarece avem 10 caracteristici
        name='input_numeric'
        # shape=(10,) = cele 10 features numerice normalizate
    )

    y = layers.Dense( # aplicam un strat Dense cu 32 de neuroni pe caracteristicile numerice
        units=32,
        activation='relu', # introduce neliniaritate 
        kernel_regularizer=regularizers.l2(1e-4), # reduce riscul de overfitting
        name='dense_numeric'
    )(input_numeric)

    y = layers.BatchNormalization(name='bn_numeric')(y) # normalizeaza iesirea stratului numeric
    y = layers.Dropout(0.3, name='dropout_numeric')(y) # aplica dropout de 30% pe ramura numerica
    output_numeric = y  # vector de 32 numere per email
                        # salveaza reprezentarea numerica

    # --- Concatenare si clasificare finala ---
    combined = layers.Concatenate(name='concatenare')(
        [output_text, output_numeric] # combina reprezentarea textului cu reprezentarea numerica
    )
    # combined shape: (batch, 112) = 80 din text + 32 din numeric

    z = layers.Dense( # aplica un strat Dense cu 56 de neuroni pe informatia combinata 
        units=56,                # echilibrat: 56
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4),
        name='dense_combined'
    )(combined)

    z = layers.Dropout(0.4, name='dropout_final')(z)
    # Dropout 40% — mai agresiv la final pentru regularizare puternica

    output = layers.Dense( # creeaza iesirea finala. un singur neuron produce probabilitatea de spam
        units=1,
        activation='sigmoid',    # sigmoid: output ∈ (0, 1)
        name='output_sigmoid'
    )(z)
    # output = probabilitatea ca emailul sa fie spam

    # Construim modelul cu doua inputuri si un output
    model = Model(
        inputs=[input_text, input_numeric],
        outputs=output,
        name='BiLSTM_model'
    )

    # Compilare — alegem optimizer, loss si metrica
    model.compile( # configureaza modelul pentru antrenare 
        optimizer=keras.optimizers.Adam( # foloseste Adam cu learning rate 0.0005
            learning_rate=LEARNING_RATE
            # Adam = Adaptive Moment Estimation
            # Ajusteaza automat LR per parametru
            # Combina avantajele RMSprop si Momentum
        ),
        loss='binary_crossentropy', # foloseste functia de pierdere pentru clasificare binara
        # binary_crossentropy = log loss pentru clasificare binara
        # identical cu log_loss din sklearn
        metrics=['accuracy'] # monitorizeaza acuratetea
    )

    return model


# ---------------------------------------------------------------
# PASUL 8: Arhitectura CNN 1D
#
# ARHITECTURA COMPLETA:
#
# Input 1: secventa de text (200 tokens)
#   → Embedding(vocab_size, 128)
#       Identic cu BiLSTM — vectori semantici antrenabili.
#       mask_zero=False pentru CNN (nu avem LSTM).
#   → SpatialDropout1D(0.3)
#   → Conv1D(96, kernel_size=3, activation='relu', padding='same')
#       96 filtre, fereastra de 3 cuvinte.
#       Detecteaza trigrame caracteristice de spam.
#       padding='same': outputul are aceeasi lungime ca inputul.
#   → Conv1D(80, kernel_size=4, activation='relu', padding='same')
#       80 filtre, fereastra de 4 cuvinte — expresii mai lungi.
#   → Conv1D(64, kernel_size=5, activation='relu', padding='same')
#       64 filtre, fereastra de 5 cuvinte — context mai larg.
#   → GlobalMaxPooling1D()
#       Extrage valoarea maxima din fiecare filtru pe toata secventa.
#       "Oriunde in email exista acest tipar, il detectam."
#       Output: vector de 64 numere (64 = ultimul conv)
#   → Dense(56, activation='relu')
#   → Dropout(0.35)
#   → output_text: vector de 56 numere
#
# Input 2: features numerice (10 valori) — identic cu BiLSTM
#   → Dense(32, activation='relu')
#   → BatchNormalization()
#   → Dropout(0.3)
#   → output_numeric: vector de 32 numere
#
# Concatenare: [output_text (56) | output_numeric (32)] = 88 numere
#   → Dense(56, activation='relu')
#   → Dropout(0.4)
#   → Dense(1, activation='sigmoid')
# ---------------------------------------------------------------

def construieste_cnn1d(vocab_size, embed_dim, max_len, n_numeric): # functia care construieste CNN1d
    """
    Construieste arhitectura CNN 1D cu input dublu:
    text (secventa) + features numerice.

    CNN 1D e complementar BiLSTM:
    - BiLSTM: context secvential, dependente pe termen lung
    - CNN 1D: tipare locale, rapid, paralel
    """

    # --- Ramura text ---
    input_text = Input(shape=(max_len,), name='input_text')

    x = layers.Embedding( # creeaza intrarea textului, cu 200 tokens
        input_dim=vocab_size, # cate randuri are matricea embedding
        output_dim=embed_dim,    # 128 dimensiuni. cate nr are fiecare vector
        mask_zero=False, # CNN1D nu foloseste masca pentru padding
        embeddings_regularizer=regularizers.l2(1e-5), # aplica L2 pe embedding-uri
        name='embedding_cnn' # numeste stratul
    )(input_text)
    # shape: (batch, 200, 128)

    x = layers.SpatialDropout1D(0.3, name='spatial_dropout')(x)

    # Trei straturi Conv1D cu ferestre diferite
    # Arhitectură echilibrată: 96→80→64
    x = layers.Conv1D( # aplica primul strat convolutional
        filters=96,  # 96 filtre pentru trigrame. cate tipare diferite cauta simultan
        kernel_size=3, # dimensionarea ferestrei - cate cuvinte "citeste" filtrul odata
        activation='relu', # functia de activare aplicata dupa calculul convolutiei
        padding='same', # controleaza lungimea output-ului fata de input
        kernel_regularizer=regularizers.l2(1e-4), # penalizare L2 pe greutatile filtrelor
        name='conv1d_3gram' # numele stratului pentru identificare
    )(x)

    x = layers.BatchNormalization(name='bn_conv1')(x) # normalizeaza iesirea primului Conv1D
    x = layers.Dropout(0.3, name='dropout_conv1')(x) # aplica dropout dupa primul Conv1D

    x = layers.Conv1D( # al doilea strat convolutional
        filters=80,              # 80 filtre pentru 4-grame
        kernel_size=4, # filtru vede 4 cuvinte
        activation='relu', #
        padding='same',
        kernel_regularizer=regularizers.l2(1e-4),
        name='conv1d_4gram'
    )(x)

    x = layers.BatchNormalization(name='bn_conv2')(x) # normalizeaza iesire celui de al doilea Conv1D
    x = layers.Dropout(0.3, name='dropout_conv2')(x) # aplica dropout fupa al doilea Conv1D

    x = layers.Conv1D( # aplica al treilea strat convolutional
        filters=64,              # 64 filtre pentru 5-grame. filtrul vede 5 cuvinte
        kernel_size=5,
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(1e-4),
        name='conv1d_5gram'
    )(x)

    x = layers.GlobalMaxPooling1D(name='global_max_pool')(x) # ia valoarea maxima detectata de fiecare filtru pe toata secventa
    # Output: (batch, 64)

    x = layers.Dense( # aplica un strat Dense pe prezentarea textului
        units=56,                # echilibrat: 56
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4),
        name='dense_text'
    )(x)
    x = layers.Dropout(0.35, name='dropout_text')(x) # aplica dropout de 35%

    output_text = x  # salveaza reprezentarea textuala finala 

    # --- Ramura features numerice ---
    input_numeric = Input(shape=(n_numeric,), name='input_numeric') # creeaza intrarea numerica 

    y = layers.Dense( # aplica strat Dense
        units=32, # cu 32 de neuroni
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4),
        name='dense_numeric'
    )(input_numeric)
    y = layers.BatchNormalization(name='bn_numeric')(y) # normalizeaza iesirea numerica
    y = layers.Dropout(0.3, name='dropout_numeric')(y) # aplica dropout de 30%
    output_numeric = y # salveaza reprezentarea numerica

    # --- Concatenare si clasificare ---
    combined = layers.Concatenate(name='concatenare')(
        [output_text, output_numeric] 
    ) # combina textul si caracteristicile numerice
    # combined shape: (batch, 88) = 56 din text + 32 din numeric

    z = layers.Dense( # aplica un strat Dense
        units=56, # cu 56 de neuroni
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4),
        name='dense_combined'
    )(combined)
    z = layers.Dropout(0.4, name='dropout_final')(z) # aplica dropout de 40%

    output = layers.Dense(1, activation='sigmoid', # produce probabilitatea finala de spam
                          name='output_sigmoid')(z)

    model = Model( # construieste modelul CNN1D
        inputs=[input_text, input_numeric],
        outputs=output,
        name='CNN1D_model'
    )

    model.compile( # configureaza modelul pentru antrenare
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), # foloseste Adam cu learning rate 0.0005
        loss='binary_crossentropy', # foloseste pierderea pentru clasificare binara
        metrics=['accuracy'] # monitorizeaza acuratetea
    )

    return model


# ---------------------------------------------------------------
# PASUL 9: Functia de antrenare DL
#
# Callbacks folosite:
#
# EarlyStopping:
#   monitor='val_loss' — urmarim validation loss
#   patience=7 — asteptam 7 epoci fara imbunatatire
#   min_delta=0.0001 — imbunatatire minima semnificativa
#   start_from_epoch=3 — incepem monitorizarea dupa epoca 3
#   restore_best_weights=True — restauram automat cel mai bun model
#
# ReduceLROnPlateau:
#   Daca val_loss nu scade in 3 epoci → LR = LR × 0.5
#   min_lr=1e-6 — nu scadem sub aceasta valoare
#   Specificul DL: LR adaptiv pe parcursul antrenarii
#
# ModelCheckpoint:
#   Salveaza automat cel mai bun model pe disk dupa fiecare epoca.
#   Protectie: daca antrenarea e intrerupta, putem relua.
# ---------------------------------------------------------------

def antreneaza_dl(model, model_name, X_seq_train, X_num_train_sc,
                  y_train_data, X_seq_val_data, X_num_val_sc,
                  y_val_data, fisier_model): # definim functia generala de antrenare. primeste: modelul, numele modelului, textul train, caracteristicile numerice train, etichetele train, textul validation, caracteristicile numerice validation, etichetele validation
    """
    Antreneaza un model DL Keras cu callbacks automate.

    Returneaza: (model, history, timp_antrenare)
    """
    print(f"\n{'='*60}")
    print(f"ANTRENEZ: {model_name}")
    print(f"  Arhitectura: {model.name}")
    print(f"  Parametri totali: {model.count_params():,}")
    print(f"{'='*60}")
    # model.summary() — comentat pentru a evita eroarea de console width
    # Poți vedea arhitectura în cod sau rulează manual model.summary() în notebook

    # Callbacks — actiuni automate in timpul antrenarii
    callbacks = [ # creem lista de callback-uri
        EarlyStopping( # opreste antrenarea cand modelul nu mai imbunatateste validation loss
            monitor='val_loss', # urmareste validation_loss
            patience=7,               # mai răbdător cu LR mic. asteapta 7 epoci fara imbunatatire
            min_delta=0.0001,         # prag mai permisiv pentru progres mic per epocă. considera imbunatatire doar o scadere de cel putin 0.0001
            restore_best_weights=True, # restaureaza cele mai bune greutati
            start_from_epoch=5,       # începe monitorizarea după epoca 5 (LR mic = start mai lent)
            verbose=1 # afiseaza mesaje 
        ),
        ReduceLROnPlateau( # reduce learning rate-ul cand modelul stagneaza
            monitor='val_loss',
            factor=0.5, # imparte learning rate la 2
            patience=3, # asteapta 3 epoci fara imbunatatire
            min_lr=1e-6,  # nu scade sub 1e-6
            verbose=1
        ),
        ModelCheckpoint( # salveaza cel mai bun model
            filepath=fisier_model, # fisierul in care se salveaza modelul
            monitor='val_loss', # salvarea se face dupa cel mai mic validation loss
            save_best_only=True, # salveaza doar cel mai bun model
            verbose=0
        )
    ]

    start = time.time() # porneste cronometrul

    history = model.fit( # antreneaza modelul
        x=[X_seq_train, X_num_train_sc],  # lista de doua inputuri
        y=y_train_data, # trimite etichete reale
        validation_data=( # trimite datele de validation
            [X_seq_val_data, X_num_val_sc],  
            y_val_data
        ),
        epochs=EPOCHS_DL, # seteaza maximum 60 de epocoi
        batch_size=BATCH_SIZE, # seteaza batch size la 128
        class_weight=cw_dict,          # ponderi de clasa
        callbacks=callbacks, # activam callback-urile
        verbose=1                      # afisam progresul per epoca
    )

    timp = time.time() - start # calculeaza timpul antrenarii

    # Determinăm epoca optimă (best epoch)
    best_epoca = int(np.argmin(history.history['val_loss'])) + 1 # gaseste epoca unde validation loss este minim
    best_val_loss = float(np.min(history.history['val_loss'])) # ia cea mai buna valoare de validation loss
    best_val_acc = float(history.history['val_accuracy'][best_epoca - 1]) # ia acuratetea de validation din epoca optima

    # Evaluare finala pe validation (cu modelul restaurat la best epoch)
    y_pred_proba = model.predict(
        [X_seq_val_data, X_num_val_sc],
        verbose=0
    ).flatten() # flatten - transforma rezultatul intr-un vector simplu
    y_pred = (y_pred_proba >= 0.5).astype(int) # transforma probabilitatile in clase; probabilitate < 0.5 => clasa 0

    val_acc  = accuracy_score(y_val_data, y_pred) # calculeaza acuratetea
    val_loss = log_loss(y_val_data, y_pred_proba) # calculeaza log loss

    print(f"\n  Timp total:    {timp/60:.1f} min")
    print(f"  Epoci rulate:  {len(history.history['loss'])}")
    print(f"  Best epoca:    {best_epoca}")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Val Acc best:  {best_val_acc*100:.2f}%")
    print(f"  Val Loss final:{val_loss:.4f}")
    print(f"  Val Acc final: {val_acc*100:.2f}%")
    print(f"  Salvat:        {fisier_model}")

    return model, history, timp, val_acc, val_loss, y_pred_proba # returneaza rezultatele


# ---------------------------------------------------------------
# PASUL 10: Antrenare BiLSTM
# ---------------------------------------------------------------
print("\n" + "="*60)
print("PASUL 10: Antrenare BiLSTM")
print("="*60)

bilstm_model = construieste_bilstm( # construieste modelul BiLSTM
    vocab_size=vocab_size, # trimitem dimensiunea vocabularului
    embed_dim=EMBED_DIM, # trimitem dimensiunea embedding-ului 
    max_len=MAX_LEN, # trimitem lungimea secventei
    n_numeric=len(NUMERIC_COLS) # trimitem nr caracteristicilor numerice
)

(bilstm_model, history_bilstm, timp_bilstm,
 acc_bilstm, loss_bilstm, proba_bilstm) = antreneaza_dl( # antreneaza modelul BiLSTM
    bilstm_model,
    'BiLSTM Bidirectional',
    X_seq_tr, X_num_tr_sc, y_tr, # datele de train
    X_seq_val, X_num_val_sc, y_val, # datele de validation
    'models/dl_bilstm.keras' # fisierul in care se salveaza cel mai bun model BiLSTM
)

# ---------------------------------------------------------------
# PASUL 11: Antrenare CNN 1D
# ---------------------------------------------------------------
print("\n" + "="*60)
print("PASUL 11: Antrenare CNN 1D")
print("="*60)

cnn1d_model = construieste_cnn1d( # construieste CNN 1D. aceiasi parametrii ca la BiLSTM
    vocab_size=vocab_size, 
    embed_dim=EMBED_DIM,
    max_len=MAX_LEN,
    n_numeric=len(NUMERIC_COLS)
)

(cnn1d_model, history_cnn1d, timp_cnn1d,
 acc_cnn1d, loss_cnn1d, proba_cnn1d) = antreneaza_dl(
    cnn1d_model,
    'CNN 1D',
    X_seq_tr, X_num_tr_sc, y_tr,
    X_seq_val, X_num_val_sc, y_val,
    'models/dl_cnn1d.keras'
)

# ---------------------------------------------------------------
# PASUL 12: EVALUARE FINALA PE VALIDATION
# ---------------------------------------------------------------
print("\n" + "="*60)
print("EVALUARE FINALA PE VALIDATION")
print("="*60)

modele_dl = { # creem un dictionar care retine rezultatele celor doua modele
    'BiLSTM': { # salvam pentru BiLSTM
        'history':  history_bilstm, # evolutia pe epoci
        'timp':     timp_bilstm, # durata antrenarii
        'acc':      acc_bilstm, # acuratetea
        'loss':     loss_bilstm, # pierderea
        'proba':    proba_bilstm, # probabilitati estimate
        'pred':     (proba_bilstm >= 0.5).astype(int), # clasele finale
        'culoare':  CULORI_DL['BiLSTM'], # culoarea pentru grafice
    },
    'CNN1D': { # la fel si pentru CNN 1D
        'history':  history_cnn1d,
        'timp':     timp_cnn1d,
        'acc':      acc_cnn1d,
        'loss':     loss_cnn1d,
        'proba':    proba_cnn1d,
        'pred':     (proba_cnn1d >= 0.5).astype(int),
        'culoare':  CULORI_DL['CNN1D'],
    }
}

for nume, info in modele_dl.items(): # parcurge fiecare model
    print(f"{nume:<10} Acc={info['acc']*100:.2f}% | Loss={info['loss']:.4f}") # afiseaza acuratetea si loss-ul fievcarui model

# ---------------------------------------------------------------
# PASUL 13: GRAFICE DL
# ---------------------------------------------------------------

# cod pentru graifce 
print("\n" + "="*60)
print("GENEREZ GRAFICELE DL")
print("="*60)

# ===== GRAFIC 1: Loss per epoca =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor(BG)
fig.suptitle('Convergența DL — Log Loss per epocă',
             fontsize=13, fontweight='semibold', color=TEXT, y=1.02)

for idx, (nm, info) in enumerate(modele_dl.items()):
    ax = axes[idx]
    deseneaza_convergenta(ax, info['history'], info['culoare'],
                          ylabel='Log Loss')
    ax.set_title(nm, fontweight='semibold',
                 color=info['culoare'], pad=10)

plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.92])
plt.savefig('grafice/dl_convergenta_loss.png')
plt.close()
print("Salvat: grafice/dl_convergenta_loss.png")

# ===== GRAFIC 2: Accuracy per epoca =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor(BG)
fig.suptitle('Convergența DL — Accuracy per epocă',
             fontsize=13, fontweight='semibold', color=TEXT, y=1.02)

for idx, (nm, info) in enumerate(modele_dl.items()):
    ax = axes[idx]
    deseneaza_convergenta(ax, info['history'], info['culoare'],
                          ylabel='Accuracy', fmt_y=True, show_max=True)
    ax.set_title(nm, fontweight='semibold',
                 color=info['culoare'], pad=10)

plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.92])
plt.savefig('grafice/dl_convergenta_accuracy.png')
plt.close()
print("Salvat: grafice/dl_convergenta_accuracy.png")

# ===== GRAFIC 3: Confusion Matrix DL =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.patch.set_facecolor(BG)
fig.suptitle('Matrice de confuzie — DL (validation)',
             fontsize=13, fontweight='semibold', color=TEXT, y=1.02)
lab = ['Ham', 'Maliţios']
etichete_cm = [['TN', 'FP'], ['FN', 'TP']]

for idx, (nm, info) in enumerate(modele_dl.items()):
    ax = axes[idx]
    cm = confusion_matrix(y_val, info['pred'], labels=[0, 1])
    im = ax.imshow(cm, cmap='Blues')
    
    # Contur chenar pentru claritate
    for spine in ax.spines.values():
        spine.set_edgecolor('#1E293B')
        spine.set_linewidth(0.5)
    
    ax.set_title(nm, fontsize=13, fontweight='semibold',
                 color=info['culoare'], pad=12)
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
    ax.text(0.5, -0.18, f'Acc: {info["acc"]*100:.2f}%',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=10, color=TEXT, fontweight='semibold')

plt.tight_layout()
plt.savefig('grafice/dl_matrice_confuzie.png')
plt.close()
print("Salvat: grafice/dl_matrice_confuzie.png")

# ===== GRAFIC 4: Validati Loss comparativ DL =====
fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(AX_BG)
ax.set_axisbelow(True)

names_dl   = list(modele_dl.keys())
losses_dl  = [modele_dl[n]['loss'] for n in names_dl]
culori_dl  = [modele_dl[n]['culoare'] for n in names_dl]
x_pos      = np.arange(len(names_dl))

bars = ax.bar(x_pos, losses_dl, color=culori_dl,
              alpha=0.88, edgecolor='white', linewidth=0.8,
              zorder=3, width=0.42)
for bar, val in zip(bars, losses_dl):
    ax.text(bar.get_x() + bar.get_width()/2.,
            val + max(losses_dl)*0.015,
            f'{val:.4f}', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color=TEXT)

ax.set_xlabel('Model DL', fontsize=12, fontweight='semibold')
ax.set_ylabel('Validation Loss', fontsize=12, fontweight='semibold')
ax.set_title('Comparație Validation Loss — modele DL',
             fontweight='semibold', pad=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(names_dl, fontsize=11)
ax.set_ylim(0, max(losses_dl) * 1.28)
plt.tight_layout()
plt.savefig('grafice/dl_best_val_loss.png')
plt.close()
print("Salvat: grafice/dl_best_val_loss.png")

# ===== GRAFIC 5: Validation Accuracy comparativ DL =====
fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(AX_BG)
ax.set_axisbelow(True)

accs_dl = [modele_dl[n]['acc'] for n in names_dl]

bars = ax.bar(x_pos, accs_dl, color=culori_dl,
              alpha=0.88, edgecolor='white', linewidth=0.8,
              zorder=3, width=0.42)
for bar, val in zip(bars, accs_dl):
    ax.text(bar.get_x() + bar.get_width()/2.,
            val + 0.0008,
            f'{val*100:.2f}%', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color=TEXT)

ax.set_xlabel('Model DL', fontsize=12, fontweight='semibold')
ax.set_ylabel('Validation Accuracy', fontsize=12, fontweight='semibold')
ax.set_title('Comparație Validation Accuracy — modele DL',
             fontweight='semibold', pad=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(names_dl, fontsize=11)
ymin = max(0, min(accs_dl) - 0.03)
ax.set_ylim(ymin, 1.005)
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f'{v*100:.1f}%'))
plt.tight_layout()
plt.savefig('grafice/dl_best_val_accuracy.png')
plt.close()
print("Salvat: grafice/dl_best_val_accuracy.png")

# ===== GRAFIC 6: Timp antrenare DL =====
fig, ax = plt.subplots(figsize=(9, 4.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(AX_BG)
ax.set_axisbelow(True)
ax.grid(axis='x', linestyle='--', alpha=0.45, color=GRID_COL)
ax.grid(False, axis='y')

timpi_dl = [modele_dl[n]['timp'] for n in names_dl]
y_pos    = np.arange(len(names_dl))

bars = ax.barh(y_pos, timpi_dl, color=culori_dl,
               height=0.55, edgecolor='white', linewidth=1.0, zorder=2)
t_max_dl = max(timpi_dl)
for bar, t in zip(bars, timpi_dl):
    w_bar = bar.get_width()
    cy    = bar.get_y() + bar.get_height() / 2.
    lbl   = f'{t/60:.1f} min ({t:.0f}s)'
    if w_bar >= t_max_dl * 0.25:
        # eticheta alba in interiorul barei
        ax.text(w_bar / 2., cy, lbl,
                ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', zorder=4)
    else:
        # eticheta in dreapta barei
        ax.text(w_bar + t_max_dl * 0.02, cy, lbl,
                ha='left', va='center',
                fontsize=10, fontweight='bold', color=TEXT, zorder=3)

ax.set_yticks(y_pos)
ax.set_yticklabels(names_dl, fontsize=11.5, fontweight='semibold')
ax.invert_yaxis()
ax.set_xlabel('Timp (secunde)', fontsize=12, fontweight='semibold')
ax.set_title('Timp antrenare — modele DL',
             fontweight='semibold', pad=14)
ax.set_xlim(0, max(timpi_dl) * 1.28)
plt.tight_layout()
plt.savefig('grafice/dl_timp_antrenare.png')
plt.close()
print("Salvat: grafice/dl_timp_antrenare.png")

# ===== GRAFIC 7: COMPARATIE ML vs DL — grafic special =====
# Acesta este graficul care sintetizeaza intreaga comparatie
# dintre abordarea ML clasica si Deep Learning.
# Este graficul central pentru capitolul comparativ din licenta.
print("\nGenerez graficul comparativ ML vs DL...")

# Rezultatele ML — folosim valorile reale din rularea anterioară
# IMPORTANT: Dacă rerulezi train_ml_total.py și obții valori diferite,
# actualizează aceste valori aici pentru consistență în graficul comparativ!
ml_names  = ['Linear-SGD', 'SVM-SGD', 'Gradient Boosting']
ml_accs   = [0.9750, 0.9657, 0.9820]
ml_losses = [0.0750, 0.0866, 0.0835]
ml_timpi  = [239, 236, 149]
print("Rezultate ML incarcate din valorile reale ale rularii.")

dl_names  = list(modele_dl.keys())
dl_accs   = [modele_dl[n]['acc']  for n in dl_names]
dl_losses = [modele_dl[n]['loss'] for n in dl_names]
dl_timpi  = [modele_dl[n]['timp'] for n in dl_names]

all_names  = ml_names + dl_names
all_accs   = ml_accs  + dl_accs
all_losses = ml_losses + dl_losses

# Generăm graficul comparativ cu rezultatele ML reale
if len(ml_names) > 0:
    # Culori: ML = culori din CULORI_ML, DL = culori din CULORI_DL
    all_culori_acc = (
        [CULORI_ML.get(n, '#64748B') for n in ml_names] +
        [modele_dl[n]['culoare'] for n in dl_names]
    )
    all_culori_loss = all_culori_acc

    # Figura cu 2 subploturi: Accuracy si Loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle('Comparație completă — ML clasic vs Deep Learning',
                 fontsize=14, fontweight='semibold', color=TEXT, y=1.02)

    # Subplot 1 — Accuracy
    x_pos_all = np.arange(len(all_names))

    bars1 = ax1.bar(x_pos_all, all_accs, color=all_culori_acc,
                    alpha=0.88, edgecolor='white', linewidth=0.8,
                    zorder=3, width=0.6)

    # Evidențiem cel mai bun model cu o bordură aurie
    best_acc_idx = np.argmax(all_accs)
    bars1[best_acc_idx].set_edgecolor('#F59E0B')
    bars1[best_acc_idx].set_linewidth(2.5)

    for bar, val in zip(bars1, all_accs):
        ax1.text(bar.get_x() + bar.get_width()/2.,
                 val + 0.0005,
                 f'{val*100:.2f}%', ha='center', va='bottom',
                 fontsize=8.5, fontweight='bold', color=TEXT)

    # Linie verticala de separare ML / DL
    sep_x = len(ml_names) - 0.5
    ax1.axvline(x=sep_x, color='#94A3B8',
                linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)

    # Etichetele sus, deasupra graficului, pentru aspect profesional
    y_lbl_top = max(all_accs) + (max(all_accs) - min(all_accs)) * 0.10
    ax1.text(len(ml_names)/2 - 0.5, y_lbl_top,
             'ML clasic', ha='center', fontsize=10, color='#475569',
             fontstyle='italic', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='#F1F5F9',
                       edgecolor='#94A3B8', alpha=0.95, linewidth=1.2))
    ax1.text(len(ml_names) + len(dl_names)/2 - 0.5, y_lbl_top,
             'Deep Learning', ha='center', fontsize=10, color='#475569',
             fontstyle='italic', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='#F1F5F9',
                       edgecolor='#94A3B8', alpha=0.95, linewidth=1.2))

    ax1.set_xticks(x_pos_all)
    ax1.set_xticklabels(all_names, rotation=15, ha='right', fontsize=9.5)
    ax1.set_ylabel('Validation Accuracy', fontsize=11, fontweight='semibold')
    ax1.set_title('Accuracy', fontweight='semibold', pad=10)
    ymin1 = max(0, min(all_accs) - 0.02)
    ymax1 = max(all_accs) + (max(all_accs) - min(all_accs)) * 0.15
    ax1.set_ylim(ymin1, ymax1)
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f'{v*100:.1f}%'))
    ax1.set_facecolor(AX_BG)
    ax1.set_axisbelow(True)

    # Subplot 2 — Loss
    bars2 = ax2.bar(x_pos_all, all_losses, color=all_culori_loss,
                    alpha=0.88, edgecolor='white', linewidth=0.8,
                    zorder=3, width=0.6)

    # Evidențiem cel mai bun model (loss minim) cu o bordură aurie
    best_loss_idx = np.argmin(all_losses)
    bars2[best_loss_idx].set_edgecolor('#F59E0B')
    bars2[best_loss_idx].set_linewidth(2.5)

    for bar, val in zip(bars2, all_losses):
        ax2.text(bar.get_x() + bar.get_width()/2.,
                 val + max(all_losses)*0.012,
                 f'{val:.4f}', ha='center', va='bottom',
                 fontsize=8.5, fontweight='bold', color=TEXT)

    ax2.axvline(x=sep_x, color='#94A3B8',
                linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)

    # Etichetele sus, deasupra graficului
    y_lbl2_top = max(all_losses) + (max(all_losses) - 0) * 0.10
    ax2.text(len(ml_names)/2 - 0.5, y_lbl2_top,
             'ML clasic', ha='center', fontsize=10, color='#475569',
             fontstyle='italic', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='#F1F5F9',
                       edgecolor='#94A3B8', alpha=0.95, linewidth=1.2))
    ax2.text(len(ml_names) + len(dl_names)/2 - 0.5, y_lbl2_top,
             'Deep Learning', ha='center', fontsize=10, color='#475569',
             fontstyle='italic', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='#F1F5F9',
                       edgecolor='#94A3B8', alpha=0.95, linewidth=1.2))

    ax2.set_xticks(x_pos_all)
    ax2.set_xticklabels(all_names, rotation=15, ha='right', fontsize=9.5)
    ax2.set_ylabel('Validation Loss', fontsize=11, fontweight='semibold')
    ax2.set_title('Log Loss', fontweight='semibold', pad=10)
    ymax2 = max(all_losses) + (max(all_losses) - 0) * 0.15
    ax2.set_ylim(0, ymax2)
    ax2.set_facecolor(AX_BG)
    ax2.set_axisbelow(True)

    plt.tight_layout(rect=[0.01, 0.03, 0.99, 0.92])
    plt.savefig('grafice/comparatie_ml_vs_dl.png')
    plt.close()
    print("Salvat: grafice/comparatie_ml_vs_dl.png")
else:
    print("Graficul comparativ ML vs DL nu a fost generat (lipsesc rezultate ML).")

# ---------------------------------------------------------------
# SUMAR FINAL
# ---------------------------------------------------------------
print("\n" + "="*65)
print("SUMAR FINAL — ANTRENARE DL")
print("="*65)
print(f"{'Model':<12} {'ValLoss':>10} {'ValAcc':>10} "
      f"{'Epoci':>8} {'Timp':>10}")
print("-"*55)
for nm, info in modele_dl.items():
    epoci = len(info['history'].history['loss'])
    print(f"{nm:<12} {info['loss']:>10.4f}  "
          f"{info['acc']*100:>9.2f}%  "
          f"{epoci:>7}  {info['timp']/60:>7.1f} min")

print(f"\nDiagnostic overfit (gap train-val accuracy la best epoca):")
for nm, info in modele_dl.items():
    best_idx = int(np.argmin(info['history'].history['val_loss']))
    tr_acc_best = info['history'].history['accuracy'][best_idx]
    vl_acc_best = info['history'].history['val_accuracy'][best_idx]
    gap = (tr_acc_best - vl_acc_best) * 100
    status = 'OVERFIT risk' if gap > 3.0 else 'OK'
    print(f"  {nm:<12} {status:<14} | gap={gap:+.2f}pp "
          f"(train={tr_acc_best*100:.2f}% val={vl_acc_best*100:.2f}%)")

print("\nModele salvate:")
print("  models/dl_bilstm.keras    (BiLSTM Bidirectional)")
print("  models/dl_cnn1d.keras     (CNN 1D)")
print("  models/dl_tokenizer.pkl (Tokenizer Keras)")
print("  models/dl_numeric_scaler.pkl (Scaler features numerice)")

print("\nGrafice DL salvate:")
print("  grafice/dl_convergenta_loss.png")
print("  grafice/dl_convergenta_accuracy.png")
print("  grafice/dl_matrice_confuzie.png")
print("  grafice/dl_best_val_loss.png")
print("  grafice/dl_best_val_accuracy.png")
print("  grafice/dl_timp_antrenare.png")
print("  grafice/comparatie_ml_vs_dl.png   ← GRAFICUL PRINCIPAL")

print("\n=== ANTRENARE DL COMPLETA! ===")