"""Ìmọ́lẹ̀ digital twin: a simulated Lagos distribution-transformer (DT) community.

One 11kV/415V three-phase DT feeds prepaid customers on several streets. Everything
the DisCo could measure is produced (phase meters, optional street "observer" meters,
customer prepaid meters), together with the hidden truth (who steals, how, from when),
so detection can be scored honestly.

All behaviour is SIMULATED. Patterns are chosen to resemble Lagos conditions
(about 15 hours of supply a day, evening peaks, prepaid metering, solar adoption).
"""
import numpy as np

HOD_SHAPE = np.array([0.45, 0.4, 0.38, 0.38, 0.42, 0.6, 0.9, 1.0, 0.85, 0.7, 0.65, 0.65,
                      0.7, 0.7, 0.7, 0.75, 0.85, 1.05, 1.4, 1.6, 1.6, 1.45, 1.1, 0.7])
HOD_SHAPE = HOD_SHAPE / HOD_SHAPE.mean()

THEFT_TYPES = ['full_bypass', 'partial_bypass', 'meter_tamper', 'night_bypass', 'gradual_bypass']


def simulate_dt(seed, n_cust=90, n_streets=4, days=60, theft_rate=0.08, p_illegal=0.5,
                p_solar=0.10, p_vacant=0.04, p_long_term=0.25):
    rng = np.random.default_rng(seed)
    H = days * 24
    hod = np.tile(np.arange(24), days)
    day = np.repeat(np.arange(days), 24)
    weekend = np.isin(day % 7, [5, 6])

    phase = rng.integers(0, 3, n_cust)
    street = rng.integers(0, n_streets, n_cust)

    # Grid supply: on/off Markov chain, about 15-16 hours of supply per day on average
    supply = np.zeros(H, bool); on = True
    for t in range(H):
        on = (rng.random() > 0.08) if on else (rng.random() < 0.15)
        supply[t] = on

    # True household demand (kW) when supply is on
    scale = rng.lognormal(np.log(0.5), 0.5, n_cust)
    personal = np.clip(1 + rng.normal(0, 0.25, (n_cust, 24)), 0.3, None)
    shape = HOD_SHAPE[None, :] * personal
    shape = shape / shape.mean(1, keepdims=True)
    noise = rng.gamma(10, 0.1, (n_cust, H))                    # hour-to-hour variation (cv ~0.3)
    demand = scale[:, None] * shape[:, hod] * noise * np.where(weekend, 1.1, 1.0)

    # Honest behaviour changes that LOOK suspicious (false-flag traps)
    role = np.array(['normal'] * n_cust, dtype=object)
    change_day = np.full(n_cust, -1)
    idx = rng.permutation(n_cust)
    n_thief = max(1, rng.binomial(n_cust, theft_rate)) if theft_rate > 0 else 0
    thieves = idx[:n_thief]
    n_solar = rng.binomial(n_cust, p_solar); solar = idx[n_thief:n_thief + n_solar]
    n_vac = rng.binomial(n_cust, p_vacant); vacant = idx[n_thief + n_solar:n_thief + n_solar + n_vac]
    for i in solar:
        role[i] = 'solar'; d0 = rng.integers(10, 45); change_day[i] = d0
        m = (day >= d0) & (hod >= 9) & (hod <= 16)
        demand[i, m] *= 0.25
    for i in vacant:
        role[i] = 'vacant'; d0 = rng.integers(10, 45); change_day[i] = d0
        demand[i, day >= d0] *= 0.05
    # Everyday honest noise seen on real prepaid meters
    for i in range(n_cust):
        if rng.random() < 0.30:   # ran out of units: dark for a few days, then recharge
            for _ in range(rng.integers(1, 4)):
                d0 = rng.integers(0, days); demand[i, (day >= d0) & (day < d0 + rng.integers(1, 5))] = 0
        if rng.random() < 0.10:   # travel
            d0 = rng.integers(5, days); demand[i, (day >= d0) & (day < d0 + rng.integers(5, 15))] *= 0.05
        if rng.random() < 0.15 and role[i] == 'normal':  # lifestyle / economising / new tenant
            role[i] = 'lifestyle_change'; d0 = rng.integers(10, 45); change_day[i] = d0
            demand[i, day >= d0] *= rng.uniform(0.4, 0.8) if rng.random() < 0.7 else rng.uniform(1.2, 1.6)

    # Theft: share of true demand that the prepaid meter registers
    reg = np.ones((n_cust, H))
    theft_type = np.array([''] * n_cust, dtype=object)
    for i in thieves:
        typ = THEFT_TYPES[rng.integers(len(THEFT_TYPES))]
        theft_type[i] = typ; role[i] = 'thief'
        d0 = 0 if rng.random() < p_long_term else rng.integers(10, 45)
        change_day[i] = d0
        act = day >= d0
        if typ == 'full_bypass':
            reg[i, act] = 0.03
        elif typ == 'partial_bypass':
            reg[i, act] = rng.uniform(0.3, 0.6)          # e.g. AC circuit bypassed
        elif typ == 'meter_tamper':
            reg[i, act] = rng.uniform(0.4, 0.6)          # meter under-registers
        elif typ == 'night_bypass':
            reg[i, act & ((hod >= 22) | (hod < 6))] = 0.0  # steals only at night
        elif typ == 'gradual_bypass':                    # slowly increases theft to avoid sudden drop
            ramp = np.clip((day - d0) / 25, 0, 1)
            reg[i, act] = 1 - 0.7 * ramp[act]

    # Illegal unmetered connections (no meter at all)
    illegal = []
    if rng.random() < p_illegal:
        for _ in range(rng.integers(1, 3)):
            ph, st, d0 = rng.integers(0, 3), rng.integers(0, n_streets), rng.integers(0, 45)
            load = rng.lognormal(np.log(0.6), 0.4) * HOD_SHAPE[hod] * rng.gamma(10, 0.1, H) * (day >= d0)
            illegal.append(dict(phase=ph, street=st, start=d0, load=load))

    true = demand * supply[None, :]
    meter_bias = 1 + rng.normal(0, 0.01, n_cust)
    metered = true * reg * meter_bias[:, None]
    missing = rng.random((n_cust, H)) < 0.01
    metered_obs = np.where(missing, np.nan, metered)

    # Phase and street-segment energy (true flow incl. illegal), with technical losses
    def seg_flow(mask_c, ph, st=None):
        f = true[mask_c].sum(0)
        for il in illegal:
            if il['phase'] == ph and (st is None or il['street'] == st):
                f = f + il['load'] * supply
        return f
    phase_load = np.array([seg_flow(phase == p, p) for p in range(3)])
    c = 0.04 / max(phase_load[phase_load > 0].mean(), 1e-6)    # ~4% average technical loss
    tech = c * phase_load ** 2
    pm_bias = 1 + rng.normal(0, 0.005, 3)
    phase_meter = (phase_load + tech) * pm_bias[:, None] * (1 + rng.normal(0, 0.002, (3, H)))
    street_meter = np.zeros((n_streets, 3, H))
    for s in range(n_streets):
        for p in range(3):
            fl = seg_flow((phase == p) & (street == s), p, s)
            street_meter[s, p] = (fl + c * fl * phase_load[p]) * (1 + rng.normal(0, 0.003))

    return dict(H=H, days=days, hod=hod, day=day, supply=supply, phase=phase, street=street,
                n_streets=n_streets, metered=metered_obs, phase_meter=phase_meter,
                street_meter=street_meter, tech_coef_true=c,
                truth=dict(role=role, theft_type=theft_type, change_day=change_day, illegal=illegal,
                           reg=reg, true=true))
