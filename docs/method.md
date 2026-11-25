# Method notes

## Layer 1: transformer and street balance
For phase p (or street segment s, phase p) over time t:

    Gap(t) = MeasuredEnergy(t) − Σ PrepaidMeters(t) − EstimatedTechnicalLoss(t)

Technical loss is estimated as c·P², where c comes from an engineering estimate (assumed within ±20% of the true value).
The non-technical loss share over the last 14 days is turned into a probability with Bayes' rule:

- No theft: share ~ Normal(0, 1.2%), covering meter error and loss-estimate error.
- Theft: share ~ Uniform(0, 60%).
- Prior P(theft) = 0.3.

## Layer 2: customer features
These include drop vs own baseline (overall, night, midday and evening), a solar signature (midday drop minus evening drop),
low usage vs phase peers, the share of zero readings, phase and street loss share and probability, the hourly correlation
between the household's missing usage and the segment gap, the share of the gap explained by the household, and the
deficit vs neighbours relative to the gap (for long-term thieves with no "before" period).
The model is histogram gradient boosting with balanced class weights.

## Layer 3: monthly planner
Expected gap on a candidate day = forecast supply hours × typical stolen kW in working hours.
One day is chosen at random among the top 3, and the 2-hour window with the highest typical gap is selected.
Night-signature households are sent to a night operation instead.

## Digital twin
- One 3-phase DT and about 90 prepaid customers on 4 streets.
- On/off supply (about 15–16 hours a day), evening-peak household loads and hourly variation.
- Honest look-alikes: solar (10%), vacant (4%), running out of units (30%), travel (10%), lifestyle change (15%).
- Theft (8% of customers on theft transformers; 25% of them long-term): full, partial, tamper, night-only and gradual.
- Illegal unmetered connections on 50% of theft transformers.
- 40% of transformers are theft-free.
- Meter bias of 1% for customers and 0.5% for phase meters, with 1% missing readings.
