# Speech Disfluency Feature Extraction

This repository contains feature extraction pipelines for speech disfluency classification.

## Feature Extractors

The following feature extraction approaches are implemented:

* **MFCC** — Mel-Frequency Cepstral Coefficients
* **MFCC + SDC** — MFCC combined with Shifted Delta Cepstral features
* **SFFCC + SDC** — Spectral Flux-based Cepstral Coefficients combined with SDC
* **Prosodic** — Prosodic speech features
* **Prosodic + Acoustic** — Combined prosodic and acoustic features
* **Wav2Vec2 Embeddings** — Transformer-based speech representations
* **Whisper Embeddings** — Speech representations extracted from Whisper
* **Conformer Embeddings** — Speech representations extracted using a Conformer-based model

## Classification

The extracted features are intended for frame-level speech disfluency classification using:

* Random Forest (RF)
* Deep Neural Network (DNN)
* Bidirectional LSTM (BiLSTM)

## Classification Tasks

1. **Fluent vs Disfluent**
2. **Fluent vs Individual Disfluency Classes**
3. **Multi-class Disfluency Classification**

### Disfluency Classes

* Filled Pause (I)
* Prolongation (PR)
* Phrase Repetition (PhR)
* Word Repetition (WR)
* Part-word Repetition (PWR)
* Pause (P)

# Best Model per Experiment

## mfcc feature extractor-classifier

## mfcc+sdc feature extractor-classifier

| Experiment | Best Model | Weighted F1 |
|---|---|---:|
| Exp1 – Fluent vs Disfluent | BiLSTM | 0.9020 |
| Exp2 – Fluent vs Filled Pause (I) | BiLSTM | 0.9660 |
| Exp2 – Fluent vs Prolongation (PR) | BiLSTM | 1.0000 |
| Exp2 – Fluent vs Phrase Repetition (PhR) | BiLSTM | 0.9893 |
| Exp2 – Fluent vs Word Repetition (WR) | BiLSTM | 0.9496 |
| Exp2 – Fluent vs Part-word Repetition (PWR) | BiLSTM | 0.9934 |
| Exp2 – Fluent vs Pause (P) | BiLSTM | 0.9988 |
| Exp3 – 6-class Disfluency | BiLSTM | 0.3389 |

## sfcc+sdc feature extractor-classifier

## prosodic+acoustic feature extractor-classifier

| Experiment | Best Model | Weighted F1 |
|---|---|---:|
| Exp1 – Fluent vs Disfluent | Random Forest | 0.7845 |
| Exp2 – Fluent vs Filled Pause (I) | Random Forest | 0.9508 |
| Exp2 – Fluent vs Prolongation (PR) | Random Forest | 0.9667 |
| Exp2 – Fluent vs Phrase Repetition (PhR) | Random Forest | 0.9197 |
| Exp2 – Fluent vs Word Repetition (WR) | Random Forest | 0.9623 |
| Exp2 – Fluent vs Part-word Repetition (PWR) | BiLSTM | 0.9459 |
| Exp2 – Fluent vs Pause (P) | BiLSTM | 0.9967 |
| Exp3 – 6-class Disfluency | Random Forest | 0.2258 |

## wav2vec2 feature extractor-classifier

| Experiment | Best Model | Weighted F1 |
|---|---|---:|
| Exp1 – Fluent vs Disfluent | BiLSTM | 0.9006 |
| Exp2 – Fluent vs Filled Pause (I) | BiLSTM | 0.9651 |
| Exp2 – Fluent vs Prolongation (PR) | BiLSTM | 1.0000 |
| Exp2 – Fluent vs Phrase Repetition (PhR) | BiLSTM | 0.9903 |
| Exp2 – Fluent vs Word Repetition (WR) | BiLSTM | 0.9496 |
| Exp2 – Fluent vs Part-word Repetition (PWR) | BiLSTM | 0.9931 |
| Exp2 – Fluent vs Pause (P) | BiLSTM | 0.9988 |
| Exp3 – 6-class Disfluency | BiLSTM | 0.4956 |

## whisper embeddings

## conformer embeddings

| Experiment | Best Model | Weighted F1 |
|---|---|---:|
| Exp1 – Fluent vs Disfluent | Random Forest | 0.9180 |
| Exp2 – Fluent vs Filled Pause (I) | Random Forest | 0.9769 |
| Exp2 – Fluent vs Prolongation (PR) | Random Forest | 0.9667 |
| Exp2 – Fluent vs Phrase Repetition (PhR) | Random Forest | 0.9741 |
| Exp2 – Fluent vs Word Repetition (WR) | Random Forest | 0.9807 |
| Exp2 – Fluent vs Part-word Repetition (PWR) | Random Forest | 0.9905 |
| Exp2 – Fluent vs Pause (P) | Random Forest | 0.9910 |
| Exp3 – 6-class Disfluency | Random Forest | 0.9174 |
