# Validation protocol: from AI flag to action

The AI only prioritises where to look. No customer is back-billed or disconnected on a risk score.

## 1. Monthly trigger (automatic)
- The AI runs on the latest meter data.
- If no street segment has P(theft) above 0.5, **no survey that month**.
- Otherwise it issues one survey plan: the day (chosen randomly among the 3 highest-gap days), the time window, the segments, and up to 10 households.
- Households with a night-only pattern are scheduled for a night operation with security support.

## 2. Desk checks (before the visit)
- **Data quality:** confirm that the phase, street and customer meters reported normally (no meter or mapping error, no missing days).
- **Vending history:** steady, high token purchases lower suspicion.
- **Supply hours:** confirm that low usage was not caused by outages on that feeder.
- **Known honest reasons:** solar installation, vacancy, travel, new tenant.

## 3. Field check (unannounced, by authorised DisCo staff only)
- Clamp the service drop **at the pole or service point**, not at the meter input, so that a tap before the meter is not missed.
- Compare the clamp reading with the meter's registered energy over the same 10–15 minutes.
- Photograph the meter seal, terminals and wiring. Record the result in the inspection form.

## 4. Outcome
- **Confirmed** (bypass, tamper or illegal connection found): follow the DisCo's approved procedure and due process.
- **Not confirmed:** record "no theft found". The household is not contacted again on the same evidence.
- In both cases the customer can respond and appeal before any back-billing.

## 5. Feedback (optional)
Confirmed and not-confirmed outcomes are stored, anonymised, as labels to retrain the model on Nigerian data.
