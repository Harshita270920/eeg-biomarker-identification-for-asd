# Live EEG participant demo

Choose a participant and click Run live pipeline. The backend streams actual completed stages; it does not replay saved predictions.

## Run locally

Python 3.11 or 3.12:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python live_server.py
```

Open http://localhost:8002. Localhost only works on the computer running the server. For public access deploy a Render Python Web Service with build `pip install -r requirements.txt` and start `HOST=0.0.0.0 python live_server.py` (free instance).

## Panel demonstration

1. Select a participant. They are held out from training.
2. Watch five-fold cross-fitting of covariance SVM, spectral SVM and covariance Extra Trees.
3. Each fit tunes settings using three inner folds. Imputation, feature selection and scaling are fitted within those training folds.
4. Ensemble weights are learned from training-only cross-fitted scores.
5. The three models are refitted on all other 55 participants and score the held-out participant. The weighted score uses a fixed 0.5 threshold.
6. Open Backend result JSON and show live_server.py and pipeline.py to explain the backend.

Raw EEG preprocessing and extraction of 1,050 covariance + 240 spectral features are already completed and cached in features.npz. This demo starts from those features, not raw EEG. No saved scores or weights are loaded. Participant names/filenames have been replaced with ordinal identifiers. Labels are the existing project's research labels and are not independently verified clinical diagnoses. Scores are not calibrated diagnostic probabilities. This is the project's ensemble extension, not Han's original microstate classifier.

The original historical cohort accuracy is 44/56 (78.57%); one live demonstration does not establish a new cohort accuracy. Fresh results can differ across software versions. Local smoke test: participant 1 score 0.5003234179159688, 18 completed base fits, approximately 4.25 seconds on this laptop. Hosted timing depends on available compute.

Only one run is accepted at a time. Keep the page open while it runs. The included render.yaml is for a standalone deployment of this folder.
