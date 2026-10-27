# EEG research dashboard

A static dashboard for saved subject-wise EEG classification results. No backend, package installation or runtime dependencies.

## Render deployment

Push this folder to a Git repository. In Render choose New > Static Site, connect that repository, and use:

- Build command: `echo Static dashboard ready`
- Publish directory: `.`

Alternatively, create a Render Blueprint from the included render.yaml.

Open index.html directly for offline use. The dropdown compares five saved models; the outcome filter isolates errors or correct predictions; Export CSV downloads the selected model's complete 56-participant results.

The dashboard contains ordinal participant numbers only, not original filenames. Results are exploratory saved research predictions and are not clinical diagnoses. Source: three_view_ensemble/loocv_all_predictions.json and summary.json.
