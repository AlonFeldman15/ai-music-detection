# Real vs. AI-Generated Music Detection

Can deep learning tell human-made music from AI-generated music, and does that hold up on
data it has never seen? This project compares three PyTorch models on 10-second music clips
under a cost-sensitive objective, then stress-tests them on external datasets.

Deep Learning course final project, M.Sc., Tel Aviv University (2026).
Team: Alon Feldman and Maayan Bortz.

## Highlights
- **Data:** 9,879 songs from the SONICS dataset (5,000 real, 4,879 AI-generated), resampled to
  16 kHz mono and split into 177,855 ten-second clips. Train/val/test split at the song level
  to prevent leakage between clips of the same song.
- **Models:**
  - LSTM trained from scratch on 64-bin log-Mel spectrograms
  - Mel-CNN: frozen PANNs Cnn14 (16 kHz) backbone with a classification head
  - Attention model: frozen MERT-v1-95M music transformer with a classification head
- **Cost-sensitive evaluation:** missing AI-generated music costs 5x more than flagging real
  music. Decision thresholds are selected on validation data to minimize that cost.
- **Hyperparameter search:** manual grid plus Optuna.

## Results

| Model | SONICS test F1 | SONICS cost | External F1 (GTZAN + FakeMusicCaps) |
|---|---|---|---|
| Attention (MERT transfer) | **0.972** | **0.034** | 0.532 |
| LSTM (from scratch) | 0.951 | 0.069 | **0.682** |
| Mel-CNN (PANNs transfer) | 0.893 | 0.179 | 0.523 |

**Key finding:** the best in-domain model was not the best generalizer. MERT led on SONICS,
but the scratch LSTM held up best on unseen sources. In-domain accuracy alone is not enough
evidence of robust AI-music detection, and calibrated, cost-aware thresholds matter.

## From design to results
The project started with a design review (`docs/design_review.pdf`) that set the module
structure, the two architecture families and a hypothesis: a temporal model would beat a
spectral CNN. What changed along the way:
- **Hypothesis:** confirmed in-domain (the MERT temporal model was best on SONICS), but the
  external test showed the scratch LSTM generalized best.
- **Cost framing:** the proposal prioritized precision. The final evaluation weighted missed
  AI tracks 5x more, matching the compliance and royalty risk platforms actually face.
- **Sampling:** a flat clip pool was replaced with song-level sampling to avoid correlated clips
  and leakage.

## Repository
- `data_preparation.ipynb`: data download, song-level split and 10-second segmentation
- `real_vs_ai_music_detection.ipynb`: main pipeline (dataset, models, training, HPO,
  evaluation, reporting), with outputs
- `docs/design_review.pdf`: initial design review (proposal before training)
- `docs/final_report.pdf`: full project report
- `docs/presentation.pdf`: project presentation

## Running
Built for Google Colab with a T4 GPU. Datasets are not included. Mount Google Drive and
replace `PROJECT_ROOT` in the notebook paths with the folder holding the data.

**Stack:** Python, PyTorch, torchaudio, PANNs, MERT (Hugging Face), scikit-learn, Optuna, pandas
