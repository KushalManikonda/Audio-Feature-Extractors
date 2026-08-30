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
