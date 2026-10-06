"""Starter exit (step 7): the chance the manager pulls the starter before each batter after his first.

    chance = 1 / (1 + exp(-(w0 + x . w)))

x = the base terms (start of an inning, pitch count with bends, batters faced, times through the order,
inning, the first two batters of the first inning) plus the factor families Mark kept on 2026-10-01
(KEPT), all built by design() from a table of decision states (tools/step7_data.py has the history
fields: his last 5 starts, projection, rest, injured list, team hook, bullpen workload, due-up batters).
Fitted on 2021-2024 by tools/step7_build.py; weights in data/model/exit_model.json.

Every input is known before that batter: the game so far, and earlier dates for player and team history.
d may be a DataFrame or a dict of equal-length arrays (the simulation passes one per step).
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

MODEL = Path("data/model/exit_model.json")
KEPT = ["Runs allowed this game", "Trouble this inning", "Score margin", "Hits, walks, HR and strikeouts so far",
        "His usual leash (last 5 starts)", "Projected innings per start", "Rest and last start",
        "Back from IL, early season", "Team's usual hook", "Bullpen workload", "Due-up batters",
        "Month", "Season end x race",   # adopted 2026-10-04 (tools/step7_calendar.py)
        "Debut or from the bullpen"]    # adopted 2026-10-04 evening (tools/step7_rookies.py)
PARKED = ["League hook trend"]
STATE = ["pitches_so_far", "start_inning", "bf", "tto", "inning", "runs_so_far", "runs_inning", "runners_on",
         "pitches_inning", "reached_last3", "margin", "hits_so_far", "bb_so_far", "hr_so_far", "k_so_far", "q", "q_next",
         "same_side"]
PREGAME = ["leash_pitches", "leash_bf", "first_start_ever", "proj_ip_gs", "has_proj", "days_since_start", "last_p",
           "il_return", "starts_this_season", "team_hook", "pen_pitches_1d", "pen_pitches_3d", "lg_share100",
           "month", "days_to_end", "playoff_chance", "days_since_debut", "debut_this_season"]


def hinge(x, k, scale):
    return np.maximum(x - k, 0) / scale


def _g(d, k):
    return np.asarray(d[k], dtype=float)


def base_cols(d):
    pc, s, bf = _g(d, "pitches_so_far"), _g(d, "start_inning"), _g(d, "bf")
    c = {"start of inning": s, "pitches (/100)": pc / 100}
    for k in (60, 75, 85, 95, 105):
        c[f"pitches over {k} (/10)"] = hinge(pc, k, 10)
        c[f"start of inning x pitches over {k}"] = s * hinge(pc, k, 10)
    c["start of inning x pitches (/100)"] = s * pc / 100
    c["batters faced (/10)"] = bf / 10
    c["batters faced over 18 (/5)"] = hinge(bf, 18, 5)
    c["batters faced over 24 (/5)"] = hinge(bf, 24, 5)
    tto = _g(d, "tto")
    c["3rd time through"] = (tto == 3).astype(float)
    c["4th time through+"] = (tto >= 4).astype(float)
    inn = np.minimum(_g(d, "inning"), 9)
    for i in range(2, 10):
        c[f"inning {i}{'+' if i == 9 else ''}"] = (inn == i).astype(float)
    c["first 2 batters, mid-inning"] = ((bf < 3) & (s == 0)).astype(float)
    return c


def families(d):
    pc, bf, s = _g(d, "pitches_so_far"), _g(d, "bf"), _g(d, "start_inning")
    rs = _g(d, "runs_so_far")
    m = np.clip(_g(d, "margin"), -6, 6)
    lp, lb = _g(d, "leash_pitches"), _g(d, "leash_bf")
    pbf = 4.3 * _g(d, "proj_ip_gs")
    early = (_g(d, "starts_this_season") <= 1).astype(float)
    hook = _g(d, "team_hook") / 10
    same = _g(d, "same_side")
    out = {
        "Runs allowed this game": {"runs so far": rs, "runs over 3": hinge(rs, 3, 1)},
        "Trouble this inning": {"runs this inning": _g(d, "runs_inning"),
                                "runners on (mid-inning)": _g(d, "runners_on") * (1 - s),
                                "pitches this inning (/10)": _g(d, "pitches_inning") / 10,
                                "reached of last 3": _g(d, "reached_last3")},
        "Score margin": {"margin (his team ahead +)": m / 3, "blowout (5+ either way)": (np.abs(m) >= 5).astype(float)},
        "Hits, walks, HR and strikeouts so far": {"hits": _g(d, "hits_so_far") / 3, "walks + HBP": _g(d, "bb_so_far") / 3,
                                                  "home runs": _g(d, "hr_so_far"), "strikeouts": _g(d, "k_so_far") / 3},
        "His usual leash (last 5 starts)": {"pitches vs his usual (/10)": (pc - lp) / 10,
                                            "pitches over his usual (/10)": hinge(pc - lp, 0, 10),
                                            "batters vs his usual (/5)": (bf - lb) / 5,
                                            "batters over his usual (/5)": hinge(bf - lb, 0, 5),
                                            "first career start": _g(d, "first_start_ever")},
        "Projected innings per start": {"batters vs projection (/5)": (bf - pbf) / 5,
                                        "batters over projection (/5)": hinge(bf - pbf, 0, 5),
                                        "no projection": 1 - _g(d, "has_proj")},
        "Rest and last start": {"4 days or less since last start": (_g(d, "days_since_start") <= 4).astype(float),
                                "7+ days since last start": (_g(d, "days_since_start") >= 7).astype(float),
                                "pitches last start vs usual (/10)": (_g(d, "last_p") - lp) / 10},
        "Back from IL, early season": {"first 2 games back from IL": _g(d, "il_return"),
                                       "first 2 starts of the season": early,
                                       "early x pitches over 75": early * hinge(pc, 75, 10)},
        "Team's usual hook": {"team starters' pitches vs league (/10)": hook,
                              "team hook x pitches over 85": hook * hinge(pc, 85, 10)},
        "Bullpen workload": {"relievers' pitches yesterday (/50)": _g(d, "pen_pitches_1d") / 50,
                             "relievers' pitches last 3 days (/100)": _g(d, "pen_pitches_3d") / 100},
        "Due-up batters": {"this batter's value": _g(d, "q"), "on-deck batter's value": _g(d, "q_next"),
                           "same side as pitcher": same, "same side x late (pitches over 85)": same * hinge(pc, 85, 10)},
    }
    # calendar (basedball/model/race.py): month (March with April, October with September), and a ramp over the last
    # 30 days of the regular season x the team's playoff chance that morning
    m_ = _g(d, "month")
    h85 = hinge(pc, 85, 10)
    out["Month"] = {}
    for k, name in ((5, "May"), (6, "June"), (7, "July"), (8, "August"), (9, "September")):
        out["Month"][name] = (m_ == k).astype(float)
        out["Month"][f"{name} x pitches over 85"] = (m_ == k) * h85
    L = np.clip(1 - _g(d, "days_to_end") / 30, 0, 1)
    pch = _g(d, "playoff_chance")
    out["Season end x race"] = {"last 30 days x out of the race (1 - chance)": L * (1 - pch),
                                "last 30 days x in (chance)": L * pch,
                                "last 30 days x contested 4 chance (1 - chance)": L * 4 * pch * (1 - pch)}
    # MLB debut and first starts (basedball/model/debut.py): "first career start" lumps debuts (which go longer than
    # predicted) with first starts by pitchers who had pitched in relief (openers, bullpen games: pulled far sooner)
    if "days_since_debut" in (d.columns if hasattr(d, "columns") else d):     # inputs built before 2026-10-05 lack it
        first = _g(d, "first_start_ever") == 1
        deb = _g(d, "days_since_debut") == 0
        out["Debut or from the bullpen"] = {"MLB debut start": (first & deb).astype(float),
                                            "first start after relief work": (first & ~deb).astype(float),
                                            "debuted this season (later starts)": (~first & (_g(d, "debut_this_season") == 1)).astype(float)}
    if "lg_share100" in (d.columns if hasattr(d, "columns") else d):
        sh = (_g(d, "lg_share100") - 0.2) * 10
        out["League hook trend"] = {"league share of 100-pitch starts (last 365 days) x pitches over 85": sh * hinge(pc, 85, 10),
                                    "same x pitches over 95": sh * hinge(pc, 95, 10), "same x start of inning": sh * s,
                                    "league share of 100-pitch starts": sh}
    return out


def design(d, fams=None):
    """-> (X, column names): base terms, then the families in fams (default KEPT)"""
    fams = KEPT if fams is None else fams
    cols = dict(base_cols(d))
    f = families(d)
    for name in fams:
        cols.update({f"{name}: {k}": v for k, v in f[name].items()})
    return np.column_stack(list(cols.values())), list(cols)


def fit(X, y, ridge=1.0):
    """logistic regression, columns scaled to unit spread for the fit; -> (intercept, weights...)"""
    s = np.sqrt((X ** 2).mean(0))
    s = np.where(s > 0, s, 1)
    Xs = X / s
    n = len(y)

    def f(w):
        z = w[0] + Xs @ w[1:]
        p = 1 / (1 + np.exp(-z))
        ll = -(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))).sum()
        g = p - y
        grad = np.concatenate([[g.sum()], Xs.T @ g]) + np.concatenate([[0], ridge * w[1:]])
        return (ll + 0.5 * ridge * (w[1:] ** 2).sum()) / n * 1e3, grad / n * 1e3

    r = minimize(f, np.zeros(X.shape[1] + 1), jac=True, method="L-BFGS-B", options={"maxiter": 2000, "gtol": 1e-9})
    return np.concatenate([[r.x[0]], r.x[1:] / s])


def logit(w, X):
    return w[0] + X @ w[1:]


def predict(w, X):
    return 1 / (1 + np.exp(-logit(w, X)))


def save(w, cols, path=MODEL, **meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"intercept": float(w[0]), "columns": cols, "weights": [float(v) for v in w[1:]], "families": KEPT, **meta},
              open(path, "w"), indent=1)


def load(path=MODEL):
    js = json.load(open(path))
    return np.concatenate([[js["intercept"]], js["weights"]]), js
