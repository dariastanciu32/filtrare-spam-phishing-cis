# Filtrare automată a mesajelor spam și phishing

Acest repository conține codul sursă utilizat pentru dezvoltarea și evaluarea unei soluții de filtrare automată a mesajelor spam și phishing, realizată în cadrul proiectului de diplomă „Particularități privind asigurarea sprijinului CIS al unităților de nivel tactic în condițiile războiului hibrid”.

## Fișiere principale

- `preprocess_total.py` - constituirea bazei de date, preprocesarea mesajelor și construirea vectorilor de caracteristici;
- `train_ml_total.py` - antrenarea modelelor clasice de învățare automată;
- `train_dl_total.py` - antrenarea modelelor de învățare profundă;
- `test_total.py` - evaluarea finală a modelelor pe setul de testare.

## Modele utilizate

- Linear-SGD;
- SVM-SGD;
- Gradient Boosting;
- BiLSTM;
- CNN 1D.

## Observație

Seturile de date și modelele salvate nu sunt incluse în repository, deoarece pot avea dimensiuni mari. Codul sursă reflectă fluxul experimental descris în lucrare.

## Autor

Daria-Maria Stanciu  
Academia Forțelor Terestre „Nicolae Bălcescu” din Sibiu  
2026