# Air Raid Alert Forecast (Kyiv)

A Python project that estimates the probability that an air raid alert starts in Kyiv within the next hour, using only information available at the forecast time.

**Main result:** on the untouched last two months of data, the model ranks risky hours **~1.8x better than random guessing**. It is a *relative risk indicator*, not an exact predictor.

## Task

At every hour `t` when no alert is active in Kyiv, predict `P(a new alert starts in (t, t+1h])`.

Why this is hard:
- Alerts are rare (about 6% of eligible hours), so plain accuracy is meaningless. I use PR-AUC and "lift" (how many times better than random guessing).
- Data leakage is easy to introduce. Everything below only uses information from before `t`.

## Data

[ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) (`official_data_uk.csv`), alerts from 2022-03-15 to 2026-09-07 (the latest version available at the time of work).
The file is **not** included in the repo. 
I also wanted to combine the two files and found another file that contained more recent data, but I thought it wasn't worth doing because I didn't have enough time to update and combine the file with the previous version.
(https://github.com/JustJimer/ukraine-air-alert-stats/blob/main/archive/alerts_in_ua.csv) (`archive/alerts_in_ua.csv`)

Notes on the data:
- For Kyiv city: 4141 records = **2267 real alerts**. Before 2026 each alert was recorded twice (exact duplicates), so I deduplicate first.
- Timestamps are in UTC; calendar features use Europe/Kyiv time.

## Results

Test = the last 2 months (2026-07-07 to 2026-09-07), opened once, after all decisions were made.
1346 eligible hours, 143 alert starts, base rate 10.6%.

| Model | PR-AUC | Lift vs random | Brier |
|---|---|---|---|
| Constant (train average) | 0.106 | 1.00 | 0.0971 |
| Hour of day | 0.131 | 1.24 | 0.0966 |
| Only "alert active in Kyiv oblast" | 0.146 | 1.37 | 0.0938 |
| **Random Forest + Platt calibration** | **0.247** | **2.33** | **0.0918** |

How to read it:
- **Lift** = PR-AUC divided by the base rate. 2.33 means the model finds alert starts 2.3x better than chance.
- The 10% of hours the model rates riskiest contain **26.6%** of all alert starts (random choice would give +-10%); the top 20% contain **38.5%**.
- In the top 5% riskiest hours, the true alert rate reaches 36.8% (3.5x higher than the base rate).

## How it works

1. **Hourly table** (`main.py`): turns the list of alerts into one row per hour with the target "does a new alert start in the next hour". Hours with an active alert are removed.
2. **Features** (`features.py`), all from the past only:
   - Kyiv: alert starts in the last 1/3/6/24/72 hours, hours since the last alert ended, its duration;
   - other oblasts: whether an alert is active there, whether one started in the last hour, number of oblasts in alert;
   - calendar: hour of day (sin/cos), day of week.
3. **Model** (`pipeline.py`): LightGBM, trained on data from 2024-01-01 (earlier periods looked like a different regime).
4. **Calibration**: Platt scaling fitted on out-of-sample predictions of six walk-forward windows.
5. **Evaluation** (`final_eval.py`): time-based split with a gap between train and test, one run on the test period.

## Quick start

```bash
pip install -r requirements.txt
```

1. Download `official_data_uk.csv` from the dataset repo above into a `data/` folder.
2. Predict (uses the saved model in `artifacts/`):

```bash
python predict.py --time "2026-08-21 18:00"
```

Time is Kyiv local time. The output is a probability between 0 and 1.

To reproduce everything: `python final_eval.py` (retrains and re-evaluates; takes a few minutes).

## What went wrong and what I changed

- **Trivial target.** "Will there be an alert" is almost always "yes" if an alert is already running. I changed the target to "will a *new* alert start" and only use hours without an active alert.
- **Duplicated data.** Half of the records before 2026 were exact duplicates. Found via the even monthly counts, fixed by deduplication.
- **Calendar features barely helped** (lift 1.2-1.3). Alerts depend on what the enemy does, not on the clock. Features from the past and from neighbouring oblasts gave the real gain.
- **`month` feature dropped.** With ~2.5 years of data it can memorise specific months; removing it did not hurt.
- **Overconfident probabilities.** Raw model probabilities were about 25% too high on validation; Platt scaling fixed that.
- **Leakage check.** `test_no_leakage.py` recomputes features from data truncated at time `t` and confirms they are identical (25 random moments, 0 differences).

## Limitations

- **Relative risk, not exact probability.** On the test period the model *underestimates* probabilities (e.g. predicted 13.5% vs 19.8% observed in the top group), because alert frequency rose from ~6% to ~10.6% of hours. The ranking of hours stayed correct, the absolute numbers did not. Re-calibrating on fresh data would help.
- **Optimistic validation.** Walk-forward windows suggested ~3x lift, the test gave 1.8x. I report the test number.
- **Few events.** 143 positives in the test, so all numbers are approximate.
- **Training start (2024-01-01)** was chosen by reasoning about regimes, not by a clear validation signal (differences were within noise).
- One city, one forecast horizon (1 hour), no text/media data.

## Project structure

```
├── artifacts/
│   └── model.joblib          # Final serialized model bundle (Random Forest + Platt, compress=3)
├── data/                     # Local data directory (git-ignored)
│   └── official_data_uk.csv  # Verified alert dataset (2022-03-15 to 2026-09-07)
│
│── [ Core Pipeline & Production ]
├── main.py                   # Data ingestion, target formulation, and hourly time-grid aggregation
├── features.py               # Feature engineering (regional alerts, spatial lags, cyclical time of day)
├── pipeline.py               # Architecture definition, walk-forward OOF validation, Platt logit transform
├── final_eval.py             # Full training, calibrator fitting, out-of-time evaluation, and model export
├── predict.py                # Command-line interface for risk inference at any specified timestamp
├── test_no_leakage.py        # Automated regression test verifying strict zero-leakage temporal isolation   
│
│── [ Research & Hypothesis Scripts ]
├── diagnose.py               # Data audit: checking duplicate events, interval overlaps, and missing boundaries   
├── baseline.py               # Naive heuristic benchmarks (constant baseline, hour-of-day rate) and window selection   
├── train.py                  # Initial validation prototype: Logistic Regression vs LightGBM and Gain importance   
├── compare_models.py         # Comparative benchmarking across model families (Linear vs Tree Ensembles vs RF)   
├── evaluate_folds.py         # 6-fold walk-forward validation and feature ablation testing for calendar month   
├── calibration.py            # Diagnostic reliability curves and empirical probability analysis   
├── calibrate.py              # Prototyping Platt Scaling via logistic regression over out-of-fold logits   
├── build_combined.py         # Source stitching experiment: overlap consistency audit and distribution shift analysis   
│
│── [ Configuration & Documentation ]
├── requirements.txt          # Frozen dependency specifications   
└── README.md                 # Complete project documentation and benchmark report   

## AI usage

The project was built in dialogue with Claude and Gemini; 