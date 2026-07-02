# =============================================================
# PREPROCESS_TOTAL.PY — VERSIUNEA FINALA
# =============================================================
# Scopul acestui fisier:
#   1. Incarca si uneste 3 dataset-uri: Enron + Nazario_5 + Nigerian_5
#   2. Standardizeaza coloanele
#   3. Extrage 4 features URL (nr linkuri, domenii suspecte etc.)
#   4. Extrage 6 features structurale (lungime, majuscule, trigger words)
#   5. Curata textul emailurilor (lowercase, stopwords, stemming)
#   6. Imparte datele in Train (80%) si Test (20%) stratificat
#   7. Aplica TF-IDF (15.000 features, trigrame, sublinear_tf)
#   8. Salveaza matricile sparse + CSV-urile pentru DL
#
# Features finale per email: 15.010
#   - 15.000 features TF-IDF (text procesat)
#   - 4 features URL
#   - 6 features structurale
# =============================================================

import pandas as pd # pentru datele in format tabelar
import numpy as np # pentru operatii matematice rapide.
import re # pentru regresii (cautare si inlocuire de tipare in text)
import nltk # pentru procesarea textului (Natural Language Toolkit, ofera lista de stopwords si functiile de tokenizare, stemming etc.)
import joblib # pentru salvarea modelelor. il folosim pentru a salva vectorizatorul TF-IDF
import os # pentru operatii pe sistem
from nltk.corpus import stopwords # importa lista de cuvinte comune in engleza care nu aduc informatie utila pentru clasificare
from nltk.stem import PorterStemmer # pentru stemming   (reduce cuvintele la forma de baza)
from sklearn.model_selection import train_test_split # pentru splitul datelor (80% train, 20% test)
from sklearn.feature_extraction.text import TfidfVectorizer # pentru vectorizarea textului
import scipy.sparse as sp # pentru matrici sparse (stocheaza doar valorile nenule, economisind memorie)
from tqdm import tqdm # pentru progres bar (afisam o bara de progres in terminal)

# ---------------------------------------------------------------
# PASUL 0: Descarcam resursele NLTK
# ---------------------------------------------------------------
print("Descarc resursele NLTK...")
nltk.download('stopwords', quiet=True) # descarcam lista de stopwords, un fisier cu lista completa de stopwords in engleza 
nltk.download('punkt',     quiet=True) # descarcam tokenizatorul ( va scrie emailul intr-un vector de cuvinte)
print("Gata!")# quiet = True → nu afiseaza mesaje suplimentare in terminal despre descarcare

# ---------------------------------------------------------------
# PASUL 1: Incarcam cele 3 dataset-uri
#
# Structuri diferite — standardizam la format comun:
#   text_brut | label (0/1) | urls_raw | sursa
#
# ENRON:      Message ID | Subject | Message  | Spam/Ham | Date
# NAZARIO_5:  sender | receiver | date | subject | body | label | urls
# NIGERIAN_5: sender | receiver | date | subject | body | label | urls
# ---------------------------------------------------------------
print("\n" + "="*60)
print("PASUL 1: Incarcare dataset-uri")
print("="*60)

CALE_ENRON    = 'data_2/raw_2/enron_spam_data-master/enron_spam_data/enron_spam_data.csv' # dataset-ul original de emailuri spam/ham
CALE_NAZARIO  = 'data_2/raw_2/phishing_ieee/Nazario_5.csv' # dataset-ul original de emailuri phishing
CALE_NIGERIAN = 'data_2/raw_2/phishing_ieee/Nigerian_5.csv' # dataset-ul original de emailuri phishing

# --- Enron ---
df_enron = pd.read_csv(CALE_ENRON) # incarcam dataset-ul original de emailuri spam/ham
df_enron_std = pd.DataFrame({ # cream un dataframe standardizat pentru dataset-ul original de emailuri spam/ham
    'text_brut': (df_enron['Subject'].fillna('') + ' ' + # adaugam subiectul si mesajul in text_brut
                  df_enron['Message'].fillna('')), # adaugam subiectul si mesajul in text_brut
    'label':     df_enron['Spam/Ham'].map({'spam': 1, 'ham': 0}), # adaugam label-ul (spam/ham) in label
    'urls_raw':  '', # adaugam urls_raw in urls_raw
    'sursa':     'enron' # adaugam sursa in sursa (enron)
})
print(f"Enron:    {len(df_enron_std):,} emailuri | " # printam numarul de emailuri in Enron
      f"spam={df_enron_std['label'].sum():,} | " # printam numarul de emailuri spam in Enron
      f"ham={(df_enron_std['label']==0).sum():,}") # printam numarul de emailuri ham in Enron

# --- Nazario_5 ---
df_naz = pd.read_csv(CALE_NAZARIO) # incarcam dataset-ul original de emailuri de phishing (Nazario_5)
df_naz_std = pd.DataFrame({ # cream un dataframe standardizat pentru dataset-ul original de emailuri de phishing (Nazario_5)
    'text_brut': (df_naz['subject'].fillna('') + ' ' + # adaugam subiectul si mesajul in text_brut
                  df_naz['body'].fillna('')),  
    'label':     df_naz['label'].astype(int), # adaugam label-ul (spam/ham) in label
    'urls_raw':  df_naz['urls'].fillna(''), # adaugam urls_raw in urls_raw
    'sursa':     'nazario' # adaugam sursa in sursa (nazario)
})
print(f"Nazario:  {len(df_naz_std):,} emailuri | " # printam numarul de emailuri in Nazario_5
      f"spam={df_naz_std['label'].sum():,} | " # printam numarul de emailuri spam in Nazario_5
      f"ham={(df_naz_std['label']==0).sum():,}") # printam numarul de emailuri ham in Nazario_5

# --- Nigerian_5 ---
df_nig = pd.read_csv(CALE_NIGERIAN) # incarcam dataset-ul original de emailuri de phishing (Nigerian_5)
df_nig_std = pd.DataFrame({ # cream un dataframe standardizat pentru dataset-ul original de emailuri de phishing (Nigerian_5)
    'text_brut': (df_nig['subject'].fillna('') + ' ' + # fillna('') = inlocuieste valorile lipsa cu ''
                  df_nig['body'].fillna('')), # adaugam subiectul si mesajul in text_brut
    'label':     df_nig['label'].astype(int), # adaugam label-ul (spam/ham) in label
    'urls_raw':  df_nig['urls'].fillna(''), # adaugam urls_raw in urls_raw
    'sursa':     'nigerian' # adaugam sursa in sursa (nigerian)
})
print(f"Nigerian: {len(df_nig_std):,} emailuri | " # printam numarul de emailuri in Nigerian_5
      f"spam={df_nig_std['label'].sum():,} | " # printam numarul de emailuri spam in Nigerian_5
      f"ham={(df_nig_std['label']==0).sum():,}") # printam numarul de emailuri ham in Nigerian_5

# ---------------------------------------------------------------
# PASUL 2: Unim cele 3 dataset-uri
# ---------------------------------------------------------------
print("\n" + "="*60)
print("PASUL 2: Unire dataset-uri")
print("="*60)

df = pd.concat([df_enron_std, df_naz_std, df_nig_std], # unim cele 3 dataframe-uri in df (concatenare)
               ignore_index=True) # reseteaza indexul randurilor in df

# Eliminam newline-urile din text_brut
# Cauzeaza probleme in CSV — un email apare pe mai multe randuri
df['text_brut'] = df['text_brut'].str.replace('\n', ' ', regex=False) # inlocuieste newline-urile cu spatii in text_brut
df['text_brut'] = df['text_brut'].str.replace('\r', ' ', regex=False) # inlocuieste carriage return-urile cu spatii in text_brut

df = df.dropna(subset=['text_brut', 'label']) # eliminam randurile care au valori lipsa in coloanele text_brut sau label
df = df[df['text_brut'].str.strip() != ''].copy() # eliminam randurile care au text_brut gol sau doar spatii (str.strip() elimina spatiile de la inceputul si sfarsitul textului)

print(f"Dataset unit: {len(df):,} emailuri total")
print(f"  Spam (1): {df['label'].sum():,} ({df['label'].mean()*100:.1f}%)")
print(f"  Ham  (0): {(df['label']==0).sum():,} ({(df['label']==0).mean()*100:.1f}%)")
print(f"  Surse: {df['sursa'].value_counts().to_dict()}")

# ---------------------------------------------------------------
# PASUL 3: Extragere features URL
#
# De ce? URL-urile sunt indicator puternic de spam/phishing:
#   - Spam contine mai multe linkuri
#   - Phishing foloseste domenii suspecte (.ru, .tk, IP-uri)
#   - Linkuri scurtate (bit.ly) ascund destinatia reala
#
# 4 features extrase:
#   nr_url_total  = cate linkuri contine emailul
#   nr_url_suspect = linkuri cu domenii suspecte sau IP
#   nr_url_scurtat = linkuri scurtate
#   are_url        = flag binar 0/1
# ---------------------------------------------------------------
print("\n" + "="*60)
print("PASUL 3: Extragere features URL")
print("="*60)

DOMENII_SUSPECTE = [ # domenii frecvent folosite in spam/phishing
    '.ru', '.cn', '.tk', '.ml', '.ga', '.cf', '.xyz',
    '.top', '.click', '.win', '.loan', '.download',
    '.zip', '.mov'
]

URL_SCURTATE = [ # servicii populare de scurtare URL folosite in spam/phishing
    'bit.ly', 'tinyurl.com', 't.co', 'goo.gl',
    'ow.ly', 'buff.ly', 'short.link', 'rebrand.ly'
]

# expresia regulata pentru a extrage URL-urile din text
PATTERN_URL = re.compile( # cauta secvente care incep cu http://, https:// sau www. si continua pana la primul spatiu sau caracter special
    r'https?://[^\s<>"\']+|www\.[^\s<>"\']+',
    re.IGNORECASE # ignora majusculele/minusculele in cautare (www.Example.com = www.example.com)
)
PATTERN_IP = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}') # detecteaza adrese IPv4 in URL-uri

def extrage_urls(text): # functie care extrage URL-urile dintr-un text folosind expresia regulata definita mai sus
    """Extrage toate URL-urile dintr-un string de text."""
    if pd.isna(text) or str(text).strip() == '': # daca textul este NaN sau gol, returneaza o lista goala
        return []
    return PATTERN_URL.findall(str(text)) # foloseste expresia regulata pentru a gasi toate URL-urile din text si returneaza-le ca o lista

def analizeaza_urls(row): # functie care analizeaza URL-urile dintr-un rand al dataframe-ului si extrage cele 4 features numerice despre URL-uri
    """
    Calculeaza 4 features numerice despre URL-urile dintr-un email.
    Enron: extragem din text_brut cu regex
    Nazario/Nigerian: combinam urls_raw cu text_brut
    """
    if row['sursa'] == 'enron':
        urls = extrage_urls(row['text_brut']) # pentru emailurile din Enron, extragem URL-urile doar din text_brut folosind functia extrage_urls
    else:
        urls = list(set(
            extrage_urls(row['urls_raw']) + # pentru emailurile din Nazario si Nigerian, combinam URL-urile extrase din urls_raw cu cele extrase din text_brut (unele URL-uri pot fi in urls_raw, altele in text_brut)
            extrage_urls(row['text_brut'])
        ))

    nr_total   = len(urls) # numarul total de URL-uri din email
    nr_suspect = sum(  # numara cate URL-uri sunt suspecte, adica contin domenii suspecte sau adrese IP
        1 for url in urls 
        if any(dom in url.lower() for dom in DOMENII_SUSPECTE) # daca URL-ul contine vreunul din domeniile suspecte (ignora majusculele/minusculele)
        or bool(PATTERN_IP.search(url)) # sau daca URL-ul contine o adresa IP (detectata cu expresia regulata pentru IP-uri)
    )
    nr_scurtat = sum( # numara cate URL-uri sunt scurtate, adica contin servicii populare de scurtare URL
        1 for url in urls
        if any(s in url.lower() for s in URL_SCURTATE) # daca URL-ul contine vreunul din serviciile de scurtare URL (ignora majusculele/minusculele)
    )
    are_url = 1 if nr_total > 0 else 0

    return pd.Series({ # returneaza cele 4 features despre URL-uri ca o serie dataframe
        'nr_url_total':   nr_total,
        'nr_url_suspect': nr_suspect,
        'nr_url_scurtat': nr_scurtat,
        'are_url':        are_url
    })

tqdm.pandas(desc="Analizare URL-uri") # bara de progres pentru aplicarea functiei analizeaza_urls pe fiecare rand al dataframe-ului
print("Analizez URL-urile... (1-2 minute)")
url_features = df.progress_apply(analizeaza_urls, axis=1) # aplica functia analizeaza_urls pe fiecare rand al dataframe-ului df si returneaza un nou dataframe url_features cu cele 4 features despre URL-uri
df = pd.concat([df.reset_index(drop=True), # concateneaza dataframe-ul original df cu noul dataframe url_features pe orizontala (axis=1), resetand indexul pentru a se potrivi
                url_features.reset_index(drop=True)], axis=1)

print(f"Features URL extrase!")
print(f"  Cu cel putin un URL:  {df['are_url'].sum():,} emailuri")
print(f"  Cu URL-uri suspecte:  {(df['nr_url_suspect']>0).sum():,} emailuri")
print(f"  Cu URL-uri scurtate:  {(df['nr_url_scurtat']>0).sum():,} emailuri")

# ---------------------------------------------------------------
# PASUL 4: Extragere features structurale 
#
# Aceste 6 features captureaza caracteristici structurale
# ale emailului — independente de continutul textual.
# Impreuna cu TF-IDF, ofera o reprezentare mai completa.
#
# 6 features extrase:
#   lung_email       = lungimea in caractere
#   nr_cuvinte       = numarul de cuvinte
#   nr_exclamare     = nr semne de exclamare (spam = urgenta falsa)
#   nr_simboluri_ban = nr simboluri monetare ($, £, €)
#   pct_majuscule    = procentul de litere mari (spam scris cu CAPS)
#   nr_trigger_spam  = cuvinte cunoscute din spam
# ---------------------------------------------------------------
print("\n" + "="*60)
print("PASUL 4: Extragere features structurale")
print("="*60)

# Cuvinte care apar frecvent in spam/phishing
# Prezenta lor e un indicator puternic
TRIGGER_SPAM = [ # cuvinte care apar frecvent in emailurile de spam/phishing, prezenta lor creste probabilitatea ca emailul sa fie spam
    'free', 'win', 'winner', 'won', 'prize', 'claim',
    'urgent', 'act now', 'click here', 'limited time',
    'congratulations', 'selected', 'offer', 'discount',
    'money', 'cash', 'earn', 'income', 'investment',
    'nigerian', 'prince', 'inheritance', 'transfer',
    'verify', 'confirm', 'account', 'suspended', 'login',
    'password', 'bank', 'credit', 'loan', 'debt',
    'million', 'billion', 'dollar', 'reward', 'bonus'
]

def extrage_features_structurale(row): # scoate indicatori numerici din forma emeilului, nu din sensul cuvintelor 
    """
    Extrage 6 features numerice din structura emailului.
    Aceste features sunt independente de TF-IDF si
    adauga informatie structurala valoroasa modelelor.
    """
    text = str(row['text_brut']) if not pd.isna(row['text_brut']) else '' # luam textul brut al emialului din coloana text_brut

    # Lungimea emailului
    # Spam comercial = lung (reclame, promotii)
    # Phishing = scurt (urgenta, actiune rapida)
    lung_email = len(text) # calculam lungimea totala a emailului, adica numarul de caractere 

    # Numarul de cuvinte
    nr_cuvinte = len(text.split()) # numaram cate cuvinte are emailul

    # Semne de exclamare — spam creeaza urgenta falsa
    # "ACT NOW!!! LIMITED TIME OFFER!!!"
    nr_exclamare = text.count('!') # numaram cate semne de exclamare apar

    # Simboluri monetare — spam financiar si fraude
    nr_simboluri_ban = (text.count('$') + # numaram simbolurile monetare 
                        text.count('£') +
                        text.count('€'))

    # Procentul de litere mari
    # SPAM SCRIS CU MAJUSCULE = indicator clasic recunoscut de filtere
    litere = [c for c in text if c.isalpha()] # extragem doar literele din text, ignorand cifrele, simbolurile si spatiile
    pct_majuscule = (sum(1 for c in litere if c.isupper()) / # cate litere sunt majuscule 
                     max(len(litere), 1)) # o protectie impotriva diviziunii la zero (daca nu sunt litere, consideram procentul de majuscule ca fiind 0)

    # Cuvinte trigger — prezenta lor creste probabilitatea de spam
    text_lower = text.lower() # convertim textul la lowercase pentru a face cautarea de cuvinte trigger insensibila la majuscule/minuscule
    nr_trigger = sum(1 for t in TRIGGER_SPAM if t in text_lower) # numaram cate cuvinte trigger din lista TRIGGER_SPAM apar in textul emailului (in lowercase)

    return pd.Series({ # returnam cele 6 features structurale ca o serie dataframe
        'lung_email':       lung_email,
        'nr_cuvinte':       nr_cuvinte,
        'nr_exclamare':     nr_exclamare,
        'nr_simboluri_ban': nr_simboluri_ban,
        'pct_majuscule':    round(pct_majuscule, 4),
        'nr_trigger_spam':  nr_trigger
    })

tqdm.pandas(desc="Extragere features structurale") # bara de progres pentru aplicarea functiei extrage_features_structurale pe fiecare rand al dataframe-ului
print("Extrag features structurale... (1-2 minute)")
struct_features = df.progress_apply(extrage_features_structurale, axis=1) # aplica functia extrage_features_structurale pe fiecare rand al dataframe-ului df si returneaza un nou dataframe struct_features cu cele 6 features structurale
df = pd.concat([df.reset_index(drop=True), # concateneaza dataframe-ul original df cu noul dataframe struct_features pe orizontala (axis=1), resetand indexul pentru a se potrivi
                struct_features.reset_index(drop=True)], axis=1)

print(f"Features structurale extrase!")
print(f"  Lungime medie email:  {df['lung_email'].mean():.0f} caractere")
print(f"  Nr mediu cuvinte:     {df['nr_cuvinte'].mean():.0f}")
print(f"  Nr mediu exclamari:   {df['nr_exclamare'].mean():.2f}")
print(f"  Pct majuscule mediu:  {df['pct_majuscule'].mean()*100:.1f}%")
print(f"  Nr mediu trigger:     {df['nr_trigger_spam'].mean():.2f}")

# ---------------------------------------------------------------
# PASUL 5: Curatare text
#
# a) Lowercase        — FREE = free (acelasi token)
# b) Eliminare speciale — !!!, $$$ = zgomot → spatiu
# c) Tokenizare       — "free money" → ["free", "money"]
# d) Eliminare stopwords — the, is, a, are... nu ajuta + adaugam termeni specifici dataset-ului care nu sunt indicatori reali de spam
# e) Stemming         — running→run, emails→email
# ---------------------------------------------------------------
print("\n" + "="*60)
print("PASUL 5: Curatare text")
print("="*60)

stemmer    = PorterStemmer() # algoritm clasic de stemming care reduce cuvintele la forma de baza (running → run, emails → email)
stop_words = set(stopwords.words('english')) # set de cuvinte comune in engleza care nu aduc informatie utila pentru clasificare (the, is, a, are etc.)

# Adaugam termeni specifici dataset-ului care nu sunt indicatori reali de spam
stop_words.update([
    'enron', 'ect', 'vinc', 'corp',
    'houston', 'hou', 'ga',
    'louis', 'subject', 'thank', 'email'
])

def curata_text(text): # functie care curata si normalizeaza textul unui email, aplicand lowercase, eliminare de caractere speciale, tokenizare, eliminare de stopwords si stemming
    """Curata si normalizeaza textul unui email."""
    if pd.isna(text) or str(text).strip() == '': # daca textul este NaN sau gol, returneaza un string gol
        return ''
    text = str(text).lower() # convertim textul in litere mici (lowercase) pentru a face procesarea ulterioara insensibila la majuscule/minuscule
    text = re.sub(r'[^a-z\s]', ' ', text) # inlocuim orice caracter care nu este o litera (a-z) sau un spatiu cu un spatiu (eliminare de caractere speciale, cifre, simboluri etc.)
    cuvinte = text.split() # impartim textul curatat in cuvinte (tokenizare) folosind spatiul ca delimitator
    cuvinte_curate = [ 
        stemmer.stem(cuv) # aplicam stemming pe fiecare cuvant pentru a reduce la forma de baza (running → run, emails → email)
        for cuv in cuvinte 
        if cuv not in stop_words and len(cuv) > 2 # eliminam stopwords si cuvintele foarte scurte (1-2 litere) care nu aduc informatie utila
    ]
    return ' '.join(cuvinte_curate) # reunim cuvintele curatate intr-un string final, separate prin spatii

tqdm.pandas(desc="Curatare text") # bara de progres pentru aplicarea functiei curata_text pe fiecare rand al dataframe-ului
print("Aplic curatarea textului... (2-3 minute)")
df['text_procesat'] = df['text_brut'].progress_apply(curata_text) # aplicam functia curata_text pe fiecare rand al coloanei text_brut si salvam rezultatul in noua coloana text_procesat
df = df[df['text_procesat'].str.strip() != ''].copy() # eliminam randurile care au text_procesat gol sau doar spatii (str.strip() elimina spatiile de la inceputul si sfarsitul textului)
print(f"Curatare finalizata! Emailuri ramase: {len(df):,}")

# ---------------------------------------------------------------
# PASUL 6: Split Train/Test 80/20 stratificat
#
# stratify=y → proportia spam/ham identica in ambele seturi
# random_state=42 → acelasi split la fiecare rulare
# ---------------------------------------------------------------
print("\n" + "="*60)
print("PASUL 6: Split Train/Test 80/20")
print("="*60)

# Toate coloanele numerice — 4 URL + 6 structurale = 10 features
NUMERIC_COLS = [
    'nr_url_total',   'nr_url_suspect', 'nr_url_scurtat', 'are_url',
    'lung_email',     'nr_cuvinte',     'nr_exclamare',
    'nr_simboluri_ban','pct_majuscule', 'nr_trigger_spam'
]

df['label'] = df['label'].astype(int) # asiguram ca label-ul este de tip int (0/1) pentru a evita probleme la antrenarea modelelor ML/DL

df_train, df_test = train_test_split( # impartim datele in set de antrenare (train) si set de testare (test) folosind functia train_test_split din scikit-learn
    df,
    test_size=0.2, # 20 % din date pentru test, 80% pentru train
    random_state=42, # folosim un seed fix pentru a avea acelasi split la fiecare rulare (reproducibilitate)
    stratify=df['label'] # asiguram ca proportia de spam/ham este aceeasi in ambele seturi (stratificare pe label)
)

df_train = df_train.reset_index(drop=True) # resetam indexul pentru setul de antrenare, eliminand indexul vechi (drop=True)
df_test  = df_test.reset_index(drop=True) # resetam indexul pentru setul de testare, eliminand indexul vechi (drop=True)

# Marcam split-ul in dataset_final
df_train['split'] = 'train' # adaugam o coloana 'split' in df_train si setam valoarea 'train' pentru toate randurile, pentru a marca ca aceste randuri fac parte din setul de antrenare
df_test['split']  = 'test' # adaugam o coloana 'split' in df_test si setam valoarea 'test' pentru toate randurile, pentru a marca ca aceste randuri fac parte din setul de testare
df_complet = pd.concat([df_train, df_test], ignore_index=True) # concatenam df_train si df_test inapoi intr-un singur dataframe df_complet, pentru a putea salva un singur fisier CSV care contine atat datele de antrenare, cat si cele de testare, cu o coloana 'split' care indica setul din care face parte fiecare rand
os.makedirs('data/full', exist_ok=True)
df_complet.to_csv('data/full/dataset_final.csv', index=False)
print("Salvat: data/full/dataset_final.csv")

print(f"Train: {len(df_train):,} emailuri | "
      f"spam={df_train['label'].sum():,} | "
      f"ham={(df_train['label']==0).sum():,}")
print(f"Test:  {len(df_test):,} emailuri  | "
      f"spam={df_test['label'].sum():,} | "
      f"ham={(df_test['label']==0).sum():,}")

X_text_train = df_train['text_procesat'] # textul procesat (curatat) al emailurilor din setul de antrenare, care va fi folosit pentru vectorizarea TF-IDF
X_text_test  = df_test['text_procesat'] # textul procesat (curatat) al emailurilor din setul de testare, care va fi folosit pentru vectorizarea TF-IDF (doar transform, nu fit!)
y_train      = df_train['label'] # etichetele (0/1) pentru setul de antrenare, care vor fi folosite pentru antrenarea modelelor ML/DL
y_test       = df_test['label'] # etichetele (0/1) pentru setul de testare, care vor fi folosite pentru evaluarea modelelor ML/DL

X_num_train  = df_train[NUMERIC_COLS].values.astype(np.float32) # extragem valorile numerice (features URL + structurale) din setul de antrenare si le convertim la tipul float32 pentru a economisi memorie
X_num_test   = df_test[NUMERIC_COLS].values.astype(np.float32) # extragem valorile numerice (features URL + structurale) din setul de testare si le convertim la tipul float32 pentru a economisi memorie

# ---------------------------------------------------------------
# PASUL 7: TF-IDF Vectorization
#
# Fata de versiunea anterioara:
#   max_features: 15.000 (reprezentare mai bogata, dar mai mult timp de antrenare)
#   ngram_range:  (1,3)  (adaugam trigrame pentru expresii tipice de spam)
#   sublinear_tf: True   (logaritm pe frecventa pentru a reduce dominanta cuvintelor foarte frecvente)
#
# sublinear_tf=True:
#   Un cuvant care apare de 10x in email nu e de 10x mai
#   important decat unul care apare de 1x. Scala logaritmica
#   reduce dominanta cuvintelor foarte frecvente.
#   TF_nou = 1 + log(TF_vechi)
#
# Trigrame (1,3):
#   "click here now" ca un singur token e mai informativ
#   decat "click", "here", "now" separat
#   Prinde expresii tipice de spam de 3 cuvinte
#
# fit_transform pe train = invata vocabularul + transforma
# transform pe test      = doar transforma (nu fit!)
# NU fit pe test = previne data leakage
# ---------------------------------------------------------------
print("\n" + "="*60)
print("PASUL 7: TF-IDF Vectorization (15k features, trigrame)")
print("="*60)

tfidf = TfidfVectorizer( # vectorizator TF-IDF pentru a transforma textul procesat al emailurilor in matrice de caracteristici numerice bazate pe frecventa cuvintelor, ajustata pentru importanta lor in intregul corpus
    max_features=15000,  # mai multe features → reprezentare mai bogata
    ngram_range=(1, 3),  # unigrame + bigrame + trigrame
    min_df=2, # ignora cuvintele care apar in mai putin de 2 emailuri (reduce zgomotul)
    sublinear_tf=True,   # 1+log(TF) pentru a reduce dominanta cuvintelor foarte frecvente
    analyzer='word' # analizam cuvintele (nu caracterele)
)

print("Aplic TF-IDF pe datele de train...")
X_tfidf_train = tfidf.fit_transform(X_text_train) # invata vocabularul din textul de antrenare si transforma textul de antrenare in matrice TF-IDF (fiecare rand reprezinta un email, fiecare coloana reprezinta un cuvant sau o expresie de 1-3 cuvinte, iar valorile sunt scorurile TF-IDF care reflecta importanta relativa a fiecarui cuvant/expresie in emailuri)
X_tfidf_test  = tfidf.transform(X_text_test) # transforma textul de testare in matrice TF-IDF folosind vocabularul invatat din textul de antrenare (nu invata un vocabular nou, pentru a preveni data leakage)

print(f"Matrice TF-IDF train: {X_tfidf_train.shape[0]:,} x "
      f"{X_tfidf_train.shape[1]:,}")

# ---------------------------------------------------------------
# PASUL 8: Concatenare TF-IDF + features numerice
#
# Fiecare email = 15.000 TF-IDF + 10 numerice = 15.010 total
# sp.hstack = concatenare orizontala matrici sparse
# csr_matrix = Compressed Sparse Row = format eficient memorie
# ---------------------------------------------------------------
print("\n" + "="*60)
print("PASUL 8: Concatenare TF-IDF + features numerice")
print("="*60)

X_num_train_sp = sp.csr_matrix(X_num_train) # convertim array-ul numpy al features numerice (4 URL + 6 structurale) in format sparse
X_num_test_sp  = sp.csr_matrix(X_num_test) # convertim array-ul numpy al features numerice (4 URL + 6 structurale) in format sparse

X_train_final = sp.hstack([X_tfidf_train, X_num_train_sp]) # concatenare orizontala - adauga coloane noi la dreapta matricei TF-IDF 
X_test_final  = sp.hstack([X_tfidf_test,  X_num_test_sp]) # concatenare orizontala - adauga coloane noi la dreapta matricei TF-IDF pentru test

print(f"Matrice finala train: {X_train_final.shape[0]:,} x "
      f"{X_train_final.shape[1]:,}")
print(f"  = 15.000 TF-IDF + 10 features numerice (4 URL + 6 structurale)")

# ---------------------------------------------------------------
# PASUL 9: Salvare
# ---------------------------------------------------------------
print("\n" + "="*60)
print("PASUL 9: Salvare")
print("="*60)

os.makedirs('data/train', exist_ok=True)
os.makedirs('data/test',  exist_ok=True)
os.makedirs('models',     exist_ok=True)
os.makedirs('grafice',    exist_ok=True)

# Matricile sparse pentru train_ml_total.py
sp.save_npz('data/train/X_train_sparse.npz', X_train_final)
sp.save_npz('data/test/X_test_sparse.npz',   X_test_final)
print("Salvat: X_train_sparse.npz si X_test_sparse.npz")

# Etichetele
np.save('data/train/y_train.npy', y_train.values)
np.save('data/test/y_test.npy',   y_test.values)
print("Salvat: y_train.npy si y_test.npy")

# Vectorizatorul TF-IDF
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
print("Salvat: models/tfidf_vectorizer.pkl")

# CSV-urile pentru train_dl_total.py
# DL are nevoie de textul procesat pentru propria tokenizare
COLS_FINALE = [
    'text_procesat', 'label',
    'nr_url_total',   'nr_url_suspect', 'nr_url_scurtat', 'are_url',
    'lung_email',     'nr_cuvinte',     'nr_exclamare',
    'nr_simboluri_ban','pct_majuscule', 'nr_trigger_spam'
]

df_train[COLS_FINALE].to_csv('data/train/train_data.csv', index=False)
df_test[COLS_FINALE].to_csv('data/test/test_data.csv',    index=False)
print("Salvat: train_data.csv si test_data.csv")

# ---------------------------------------------------------------
# SUMAR FINAL
# ---------------------------------------------------------------
print("\n" + "="*60)
print("SUMAR FINAL")
print("="*60)
print(f"Dataset unit:         {len(df):,} emailuri")
print(f"  Enron:              {(df['sursa']=='enron').sum():,}")
print(f"  Nazario:            {(df['sursa']=='nazario').sum():,}")
print(f"  Nigerian:           {(df['sursa']=='nigerian').sum():,}")
print(f"Train set:            {X_train_final.shape[0]:,} emailuri")
print(f"Test set:             {X_test_final.shape[0]:,} emailuri")
print(f"Features per email:   {X_train_final.shape[1]:,}")
print(f"  TF-IDF:             15.000")
print(f"  URL features:       4")
print(f"  Structurale:        6")
print(f"\n=== PREPROCESARE COMPLETA! ===")