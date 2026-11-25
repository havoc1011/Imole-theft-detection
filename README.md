# Ìmọ́lẹ̀ — Finding Electricity Theft from the Transformer Down

**Probabilistic localisation of electricity theft using transformer phase energy balance and prepaid meter data in Nigerian distribution networks.**

Ìmọ́lẹ̀ (Yoruba: *light*) helps a DisCo find where electricity is being stolen. It works down from the
distribution transformer (DT) to the phase, then the street, then the household. It also tells the field
survey team **whether** to go out this month and **when**.

> Status: research prototype for the INARIC 2026 challenge. All results below come from a **simulated**
> Lagos transformer community and a public labelled dataset. Real performance has to be confirmed on a pilot transformer.

## How it works

| Layer | What it does | Data it uses |
|---|---|---|
| 1. Transformer phase balance | Gap = phase meter − sum of prepaid meters − normal technical loss. Bayes' rule turns the gap into P(theft) for each phase, then each street segment | DT phase meters (CTs), optional street observer meters, prepaid meters |
| 2. Customer risk ranking | Gradient-boosting model ranks households inside the suspicious segment (drops by time of day, solar signature, low vs neighbours, how well their missing usage matches the gap) | Prepaid meter history, supply hours |
| 3. Monthly survey trigger and validation | **At most one survey a month, and only if the AI flags a segment.** It picks a high-gap day (randomly among the top 3) and the best time window; night-only patterns get a night operation. Inspectors confirm with a clamp meter at the service drop. | Layer 1–2 output, supply forecast |

The AI runs independently from the meter data. The survey team moves only when the AI signals.
Survey outcomes can optionally be fed back later as labels to retrain the model.

## Key results (simulation: 60 unseen transformers, 5,400 customers; 40% of transformers theft-free)

| Measure | Result |
|---|---|
| Flagged street segments | hold 25% of customers but 85% of thieves (~11 households per segment) |
| Illegal unmetered connections located to the right street and phase | 29 of 29 |
| Thieves found when inspecting the riskiest 5% of customers | prepaid only 68% → + phase meters 77% → + street meters 83% |
| Inspection visits wasted on theft-free transformers | 4.3% → 0% with the transformer balance |
| Field visits catching theft in progress | 99% with AI-chosen day and time vs 50% with random timing |

The results held when the test was repeated on 120 fresh transformers. The hardest cases are partial bypass and meter
tampering, which resemble honest households cutting usage. The AI narrows the search to the street, and the field check confirms.

A separate baseline on the public SGCC dataset (42,372 real customers, 3,615 confirmed thieves) shows that **monthly**
consumption keeps about 98% of the detection power of daily data (AUC 0.787 vs 0.802, 5-fold CV). See `experiments/sgcc_baseline/`.

## Repository layout

```
src/                  imole_sim.py (digital twin), imole_detect.py (detector, planner, field check)
experiments/          run_experiment.py (train on simulated DTs, evaluate)
  sgcc_baseline/      monthly-vs-daily study on the public SGCC dataset
notebooks/            Imole_theft_localisation.ipynb (run in Google Colab, ~1 minute)
results/              saved outputs of both studies
docs/                 method notes and the validation protocol
data/                 how to obtain data (no data is committed)
tests/                smoke tests
```

## Quick start

```bash
pip install -r requirements.txt
python experiments/run_experiment.py results/simulation 160 60   # about 15 seconds
pytest -q
```

Or open `notebooks/Imole_theft_localisation.ipynb` in Google Colab and choose **Runtime → Run all**.

For the SGCC baseline, download the dataset first (see `data/README.md`), then:

```bash
python experiments/sgcc_baseline/theft_detection_model.py data/sgcc/data.csv results/sgcc_baseline
python experiments/sgcc_baseline/cross_validation.py data/sgcc/data.csv results/sgcc_baseline/cv.json
```

## Responsible use

A risk score is **never proof of theft**. Every flag must go through the validation protocol in
`docs/validation_protocol.md` (data quality check, vending history, physical check, the customer's right to respond)
before any back-billing or disconnection.

## Limits

- The main results are simulated; the SGCC baseline is real data from China, not Nigeria.
- Theft upstream of the transformer meter, or staff collusion that falsifies readings, is outside what the balance can see.
- Field checks only catch a bypass that is active during the visit; this is why visit timing matters.

## Roadmap

1. Streamlit dashboard (upload data → hotspots → risk list → monthly survey plan → inspection form).
2. Pilot on one Eko DisCo transformer with near-100% prepaid metering, under a data-sharing agreement compliant with the Nigeria Data Protection Act 2023.
3. Retrain on Nigerian inspection outcomes; add AMR/AMI hourly data and meter tamper alerts.

## Author

[Your name], Department of Electrical/Electronic Engineering, University of Lagos.
