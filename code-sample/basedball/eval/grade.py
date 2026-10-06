"""Grade predictions against what happened.

Two kinds of prediction are graded the same way for every model (baselines,
sub-models, the full simulator):

1. Plate appearances: one row per PA (game_key, event_num) with a probability
   column per outcome class in actuals.PA_CLASSES (p_SO, p_BB, ...).
2. Player (or team) game stats: for each stat, one row per player-game
   (game_key, player_id) with a probability for every value on the stat's
   scale: p_0, p_1, ..., the last column being "that value or more".

Measures (lower is better except where noted):
  log_loss      average of -ln(probability given to what happened). The main
                measure: it rewards putting probability on the right outcome
                and punishes confident misses hardest.
  brier_lines   average Brier score at the stat's common prop lines
                ((P(over) - outcome)^2), the error on the bets we'd make.
  mean_pred / mean_actual / bias   average predicted value vs actual.
  calibration   per line: predictions grouped into bins, predicted vs actual
                over rate (written as a separate table).
"""
import numpy as np
import pandas as pd

from .actuals import PA_CLASSES

# Scales (value range, last bin = "or more") and the common prop lines per stat.
# Lines are for scoring only; they are the usual Pick6 thresholds, not live lines.
STATS = {
    # batters
    "H": dict(who="bat", max=5, lines=[0.5, 1.5, 2.5]),
    "TB": dict(who="bat", max=12, lines=[0.5, 1.5, 2.5, 3.5]),
    "HR": dict(who="bat", max=3, lines=[0.5]),
    "R": dict(who="bat", max=4, lines=[0.5, 1.5]),
    "RBI": dict(who="bat", max=6, lines=[0.5, 1.5]),
    "HRR": dict(who="bat", max=10, lines=[0.5, 1.5, 2.5, 3.5]),
    "1B": dict(who="bat", max=5, lines=[0.5, 1.5]),
    "2B": dict(who="bat", max=3, lines=[0.5]),
    "3B": dict(who="bat", max=2, lines=[0.5]),
    "XBH": dict(who="bat", max=4, lines=[0.5]),
    "BB": dict(who="bat", max=4, lines=[0.5]),
    "SO": dict(who="bat", max=5, lines=[0.5, 1.5]),
    "SB": dict(who="bat", max=3, lines=[0.5]),
    "FPTS": dict(who="bat", max=40, lines=[4.5, 6.5, 8.5, 10.5]),
    # starting pitchers
    "P_SO": dict(who="pit", col="SO", max=16, lines=[3.5, 4.5, 5.5, 6.5, 7.5]),
    "P_OUTS": dict(who="pit", col="OUTS", max=27, lines=[14.5, 15.5, 16.5, 17.5, 18.5]),
    "P_ER": dict(who="pit", col="ER", max=10, lines=[1.5, 2.5]),
    "P_H": dict(who="pit", col="H", max=14, lines=[3.5, 4.5, 5.5]),
    "P_BB": dict(who="pit", col="BB", max=8, lines=[1.5, 2.5]),
    "P_PITCHES": dict(who="pit", col="PITCHES", max=130, lines=[79.5, 84.5, 89.5, 94.5]),
    "P_FPTS": dict(who="pit", col="FPTS", min=-15, max=50, lines=[10.5, 14.5, 18.5]),
    # teams
    "T_W": dict(who="team", col="W", max=1, lines=[0.5]),
    "T_R": dict(who="team", col="R", max=15, lines=[2.5, 3.5, 4.5, 5.5]),
    "T_R_F5": dict(who="team", col="R_F5", max=10, lines=[1.5, 2.5]),
    "T_R_INN1": dict(who="team", col="R_INN1", max=5, lines=[0.5]),
}


def scale(stat):
    s = STATS[stat]
    return np.arange(s.get("min", 0), s["max"] + 1)


def to_bins(values, stat):
    """Actual values -> bin index on the stat's scale (fantasy points rounded down)."""
    s = STATS[stat]
    v = np.floor(np.asarray(values, dtype=float))
    return (np.clip(v, s.get("min", 0), s["max"]) - s.get("min", 0)).astype(int)


def prob_cols(stat):
    return [f"p_{v}" for v in scale(stat)]


# ------------------------------------------------------------------ plate appearances
def grade_pa(pred, actual, eps=1e-6):
    """pred: game_key, event_num, p_<class>...; actual: game_key, event_num, pa_class."""
    d = actual[["game_key", "event_num", "pa_class"]].merge(pred, on=["game_key", "event_num"])
    P = d[[f"p_{c}" for c in PA_CLASSES]].to_numpy()
    P = P / P.sum(1, keepdims=True)
    y = d.pa_class.map({c: i for i, c in enumerate(PA_CLASSES)}).to_numpy()
    p_true = np.clip(P[np.arange(len(d)), y], eps, 1)
    out = {"n": len(d), "log_loss": float(-np.log(p_true).mean())}
    for i, c in enumerate(PA_CLASSES):
        hit = (y == i).astype(float)
        out[f"brier_{c}"] = float(((P[:, i] - hit) ** 2).mean())
        out[f"pred_{c}"] = float(P[:, i].mean())
        out[f"act_{c}"] = float(hit.mean())
    return out


# ------------------------------------------------------------------ player / team game stats
def grade_stat(pred, actual, stat, keys=("game_key", "player_id"), eps=1e-6):
    """pred: keys + p_<value> columns; actual: keys + the stat's column."""
    s = STATS[stat]
    col = s.get("col", stat)
    keys = list(keys)
    d = actual[keys + [col]].merge(pred, on=keys)
    if not len(d):
        return None, None
    P = d[prob_cols(stat)].to_numpy(dtype=float)
    P = P / P.sum(1, keepdims=True)
    vals = scale(stat)
    y = to_bins(d[col], stat)
    p_true = np.clip(P[np.arange(len(d)), y], eps, 1)
    actual_v = d[col].to_numpy(dtype=float)
    out = {"stat": stat, "n": len(d), "log_loss": float(-np.log(p_true).mean()),
           "mean_pred": float((P * vals).sum(1).mean()), "mean_actual": float(actual_v.mean())}
    out["bias"] = out["mean_pred"] - out["mean_actual"]
    cal = []
    briers = []
    for line in s["lines"]:
        p_over = P[:, vals > line].sum(1)
        over = (actual_v > line).astype(float)
        briers.append(((p_over - over) ** 2).mean())
        out[f"brier_{line}"] = float(briers[-1])
        bins = np.clip((p_over * 10).astype(int), 0, 9)
        g = pd.DataFrame({"bin": bins, "p": p_over, "o": over}).groupby("bin").agg(
            n=("o", "size"), pred=("p", "mean"), actual=("o", "mean")).reset_index()
        g.insert(0, "line", line)
        cal.append(g)
    out["brier_lines"] = float(np.mean(briers))
    return out, pd.concat(cal).assign(stat=stat)


def summarize(results):
    """results: list of dicts (each with model and split fields) -> DataFrame."""
    return pd.DataFrame([r for r in results if r])


# ------------------------------------------------------------------ over/under picks
PRIORITY = ["TB", "P_SO", "H", "HR", "RBI"]     # accuracy priority, in order (Sep 30)


def proxy_lines(ref_pred, stat):
    """A stand-in for a sportsbook / Pick6 line: for each player-game, the half-point line
    where the reference model's over chance is closest to 50%."""
    vals = scale(stat)
    P = ref_pred[prob_cols(stat)].to_numpy(float)
    P = P / P.sum(1, keepdims=True)
    cands = vals[:-1] + 0.5
    over = np.stack([P[:, vals > c].sum(1) for c in cands], 1)
    return cands[np.abs(over - 0.5).argmin(1)]


def grade_picks(pred, ref_pred, actual, stat, keys=("game_key", "player_id"), edge=0.05):
    """Pick the more likely side of each proxy line; share of picks that were right.
    Also the share right among picks where the model's chance is at least 50% + edge."""
    s = STATS[stat]
    col = s.get("col", stat)
    keys = list(keys)
    ref = ref_pred[keys].copy()
    ref["line"] = proxy_lines(ref_pred, stat)
    d = actual[keys + [col]].merge(pred, on=keys).merge(ref, on=keys)
    if not len(d):
        return None
    vals = scale(stat)
    P = d[prob_cols(stat)].to_numpy(float)
    P = P / P.sum(1, keepdims=True)
    p_over = np.array([row[vals > ln].sum() for row, ln in zip(P, d.line)])
    over = d[col].to_numpy(float) > d.line.to_numpy()
    pick_over = p_over > 0.5
    right = pick_over == over
    conf = np.abs(p_over - 0.5) >= edge
    return {"stat": stat, "n": len(d), "avg_line": float(d.line.mean()), "over_rate": float(over.mean()),
            "pick_accuracy": float(right.mean()), "picks_with_edge": int(conf.sum()),
            "accuracy_with_edge": float(right[conf].mean()) if conf.any() else np.nan}
