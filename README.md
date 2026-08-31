# Superpave Gyratory Compaction Curve Predictor

A CatBoost multi-output model that predicts **both coefficients** of the logarithmic
densification law for asphalt concrete from binder properties, compaction temperature
and aggregate-gradation descriptors:

$$\mathrm{AV}(N_g) = a \cdot \ln(N_g) + b$$

where `a` is the slope (densification rate) and `b` is the intercept (air voids at
one gyration). Predicting both coefficients reconstructs the full air-void curve, so
the app also reports %G<sub>mm</sub>, the Superpave criteria checks at
N<sub>ini</sub> / N<sub>des</sub> / N<sub>max</sub>, and the gyration count needed to
reach the 4 % design air-void target.

---

## Live demo

<!-- replace with your own URL once deployed -->

**https://YOUR-APP-NAME.streamlit.app**

---

## Features

| Tab                  | What it does                                                                                                                             |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Prediction**       | Slope and intercept from eight mixture inputs, the reconstructed AV and %G<sub>mm</sub> curves, Superpave criteria chips, and CSV export |
| **Sensitivity**      | Live one-at-a-time sweep of any input, run through the deployed model, plus the resulting family of compaction curves                    |
| **Batch prediction** | Upload a CSV/XLSX of many mixtures, get predictions and derived design quantities back as Excel                                          |
| **Model card**       | Algorithm, hold-out and leave-one-out performance, SHAP importance, and intended-use limits                                              |

Inputs outside the training range are flagged rather than silently extrapolated.

---

## Model

|              |                                                                      |
| ------------ | -------------------------------------------------------------------- |
| Algorithm    | `Pipeline(StandardScaler → MultiOutputRegressor(CatBoostRegressor))` |
| Responses    | Slope, Intercept                                                     |
| Inputs       | η, P<sub>b</sub>%, T<sub>R</sub> (°R), PG limits, λ, κ, GA, TX       |
| Observations | 279                                                                  |
| Validation   | 75/25 hold-out split, stratified on PG limits + leave-one-out CV     |

Standardisation sits **inside** the pipeline, so it is refitted within every
cross-validation fold and no information leaks from validation data into training.
The deployed artefact is refitted on all 279 observations after validation.

Performance figures are printed in the app's Model card tab and stored in
`model_metadata.json`.

---

## Repository layout

```
.
├── app.py                    # Streamlit application
├── compaction_model.pkl      # fitted sklearn Pipeline (pickle)
├── compaction_model.joblib   # same model, joblib format
├── model_metadata.json       # feature names/ranges, performance, SHAP, sensitivity
├── sample_batch_input.csv    # example file for the batch tab
├── requirements.txt
├── .gitignore
└── README.md
```

`model_metadata.json` is the single source of truth for feature names and order.
The app never hard-codes column names — it builds the input frame from that file,
which matters here because two training columns carry non-obvious spellings
(`η` and `lambda ` with a trailing space).

---

## Run locally

```bash
git clone https://github.com/ahmednawaad/Superpave-Gyratory-Compaction-Curve-Predictor-app.git
cd Superpave-Gyratory-Compaction-Curve-Predictor-app

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

-----

## Deploy to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **New app** → pick the repository, branch `main`, main file `app.py`.
3. **Deploy**. The first build takes a few minutes while CatBoost compiles.
4. Paste the resulting URL into the *Live demo* section above.

### Deployment notes

- **Pin your versions.** `requirements.txt` pins `scikit-learn` and `catboost` to the
  versions the model was pickled with. A different scikit-learn minor version can fail
  to unpickle the pipeline or, worse, load it with changed behaviour.
- **Model size.** If the `.pkl` ever exceeds GitHub's 100 MB limit, use
  [Git LFS](https://git-lfs.com) or fetch the artefact from a release asset at startup.
- **Cold starts.** `@st.cache_resource` keeps the model in memory between reruns; the
  first request after the app sleeps will be slower.

---

## Batch input format

Column names must match the training data exactly. Download the template from the
Batch prediction tab, or start from `sample_batch_input.csv`.

```csv
η,Pb%,TR_Rankine,PG limits,lambda ,kappa,GA,TX
0.826,5.49,748.78,-1672,6.601,0.970,3005.4,295.5
```

---

## Intended use and limits

This model supports **screening and preliminary mixture design** inside the range of
its training data. It is not a substitute for laboratory gyratory compaction testing,
and it has not been validated against mixtures, binder grades, or gradations outside
that range. Extrapolated inputs are flagged in the interface.

---

## Citation

If this tool supports published work, please cite the accompanying manuscript.
A `CITATION.cff` can be added once the paper has a DOI.

---

## License

MIT — see `LICENSE`.
