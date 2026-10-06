"""What actually happened: plate-appearance outcomes and per-player game lines.

Plate appearances come from the play-by-play (Retrosheet where published, MLB feed
otherwise). Player game lines come from the official MLB box scores when they have
been pulled (data/boxscores), which is what props settle on; the play-by-play
version (pbp_player_games) is kept as a cross-check and a fallback.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.union import load_games, load_pbp

# plate-appearance outcome classes the PA model predicts
PA_CLASSES = ["SO", "BB", "HBP", "1B", "2B", "3B", "HR", "ROE", "GO", "AO"]
_CLASS = {"SO": "SO", "BB": "BB", "IBB": "BB", "HBP": "HBP", "CI": "HBP", "1B": "1B", "2B": "2B", "3B": "3B",
          "HR": "HR", "ROE": "ROE", "GO": "GO", "FC": "GO", "SH": "GO", "FO": "AO", "LO": "AO", "PO": "AO"}

# Pick6 / DraftKings fantasy scoring (pick6.draftkings.com/pick6-rules-and-scoring-mlb)
FPTS_BAT = {"1B": 3, "2B": 5, "3B": 8, "HR": 10, "RBI": 2, "R": 2, "BB": 2, "HBP": 2, "SB": 5}
FPTS_PIT = {"OUTS": 0.75, "SO": 2, "W": 4, "ER": -2, "H": -0.6, "BB": -0.6, "HBP": -0.6,
            "CG": 2.5, "SHO": 2.5, "NH": 5}

BOX = Path("data/boxscores")


def pa_outcomes(seasons, game_type="R", columns=None):
    """One row per plate appearance with its outcome class. columns: load only these play-by-play columns (memory)."""
    if columns is not None:
        columns = sorted(set(columns) | {"is_pa", "game_type", "outcome", "batted_ball"})
    p = load_pbp(seasons, columns=columns)
    p = p[(p.is_pa == 1) & (p.game_type == game_type)].copy()
    cls = p.outcome.map(_CLASS)
    other = cls.isna()
    cls[other] = np.where(p.batted_ball[other] == "G", "GO", "AO")
    p["pa_class"] = cls
    return p


def _bat_totals(d):
    d["H"] = d["1B"] + d["2B"] + d["3B"] + d["HR"]
    d["TB"] = d["1B"] + 2 * d["2B"] + 3 * d["3B"] + 4 * d["HR"]
    d["XBH"] = d["2B"] + d["3B"] + d["HR"]
    d["HRR"] = d["H"] + d["R"] + d["RBI"]
    d["FPTS"] = sum(w * d[k] for k, w in FPTS_BAT.items())
    return d


def pbp_player_games(seasons, game_type="R"):
    """Batter and pitcher game lines rebuilt from the play-by-play (no ER or W)."""
    p = load_pbp(seasons)
    p = p[p.game_type == game_type]
    pa = p[p.is_pa == 1]
    o = pa.outcome
    key = ["season", "game_date", "game_key", "batter_id"]
    bat = pa.assign(**{"1B": o == "1B", "2B": o == "2B", "3B": o == "3B", "HR": o == "HR",
                       "BB": o.isin(["BB", "IBB"]), "HBP": o == "HBP", "SO": o == "SO", "PA": 1}) \
        .groupby(key).agg(team=("bat_team", "first"), opp=("fld_team", "first"),
                          lineup_slot=("lineup_pos", "first"), PA=("PA", "sum"), **{
                              c: (c, "sum") for c in ["1B", "2B", "3B", "HR", "BB", "HBP", "SO"]},
                          RBI=("rbi", "sum")).reset_index().rename(columns={"batter_id": "player_id"})
    # runs, steals and caught stealing by the runner involved
    parts = [pd.DataFrame({"game_key": p.game_key, "player_id": p.batter_id, "R": (p.bat_dest == 4) & (p.is_pa == 1)})]
    for b in (1, 2, 3):
        parts.append(pd.DataFrame({"game_key": p.game_key, "player_id": p[f"base{b}_pre"],
                                   "R": p[f"run{b}_dest"] == 4}))
    runs = pd.concat(parts)
    runs = runs[runs.player_id > 0].groupby(["game_key", "player_id"]).R.sum()
    sb, cs = [], []
    for b, nxt in ((1, "2b"), (2, "3b"), (3, "hm")):
        sb.append(pd.DataFrame({"game_key": p.game_key, "player_id": p[f"base{b}_pre"], "n": p[f"sb_{nxt}"]}))
        cs.append(pd.DataFrame({"game_key": p.game_key, "player_id": p[f"base{b}_pre"], "n": p[f"cs_{nxt}"]}))
    sb = pd.concat(sb).query("player_id > 0").groupby(["game_key", "player_id"]).n.sum()
    cs = pd.concat(cs).query("player_id > 0").groupby(["game_key", "player_id"]).n.sum()
    idx = pd.MultiIndex.from_frame(bat[["game_key", "player_id"]])
    bat["R"] = runs.reindex(idx).fillna(0).to_numpy()
    bat["SB"] = sb.reindex(idx).fillna(0).to_numpy()
    bat["CS"] = cs.reindex(idx).fillna(0).to_numpy()
    bat = _bat_totals(bat)
    first = pa.sort_values("event_num").groupby(["game_key", "bat_team", "lineup_pos"]).batter_id.first()
    starters = set(zip(first.index.get_level_values(0), first.to_numpy()))
    bat["started"] = [int((g, b) in starters) for g, b in zip(bat.game_key, bat.player_id)]

    pk = ["season", "game_date", "game_key", "pitcher_id"]
    pit = p.assign(BF=p.is_pa, H=(p.is_pa == 1) & o.reindex(p.index).isin(["1B", "2B", "3B", "HR"]),
                   HR=(p.is_pa == 1) & (p.outcome == "HR"), BB=(p.is_pa == 1) & p.outcome.isin(["BB", "IBB"]),
                   HBP=(p.is_pa == 1) & (p.outcome == "HBP"), SO=(p.is_pa == 1) & (p.outcome == "SO"),
                   PITCHES=p.pitches.where(p.is_pa == 1, 0)) \
        .groupby(pk).agg(team=("fld_team", "first"), opp=("bat_team", "first"), started=("pitcher_is_starter", "max"),
                         BF=("BF", "sum"), OUTS=("event_outs", "sum"), H=("H", "sum"), HR=("HR", "sum"),
                         BB=("BB", "sum"), HBP=("HBP", "sum"), SO=("SO", "sum"), R_on_mound=("runs_on_play", "sum"),
                         PITCHES=("PITCHES", "sum")).reset_index().rename(columns={"pitcher_id": "player_id"})
    return bat, pit


def box_player_games(seasons, game_type="R", box_dir=BOX):
    """Official box score lines; None for seasons not pulled yet."""
    bats, pits = [], []
    for s in seasons:
        fb, fp = Path(box_dir) / f"batting_{s}.parquet", Path(box_dir) / f"pitching_{s}.parquet"
        if fb.exists() and fp.exists():
            bats.append(pd.read_parquet(fb))
            pits.append(pd.read_parquet(fp))
    if not bats:
        return None, None
    bat = pd.concat(bats, ignore_index=True)
    pit = pd.concat(pits, ignore_index=True)
    bat, pit = bat[bat.game_type == game_type].copy(), pit[pit.game_type == game_type].copy()
    bat = _bat_totals(bat)
    # no-hitter by the pitcher's own team (for fantasy points): complete game with 0 hits
    pit["NH"] = ((pit.CG == 1) & (pit.H == 0)).astype(int)
    pit["FPTS"] = sum(w * pit[k] for k, w in FPTS_PIT.items())
    return bat, pit


def team_games(seasons, game_type="R"):
    """One row per team per game: runs, runs allowed, win, first-5 and first-inning runs."""
    g = load_games(seasons)
    g = g[g.game_type == game_type]
    p = load_pbp(seasons)
    p = p[p.game_type == game_type]
    inn = p.groupby(["game_key", "bat_team", "inning"]).runs_on_play.sum().unstack(fill_value=0)
    rows = []
    for side, opp in (("away", "home"), ("home", "away")):
        t = pd.DataFrame({"season": g.season, "game_date": g.game_date, "game_key": g.game_key,
                          "team": g[f"{side}_team"], "opp": g[f"{opp}_team"], "side": side,
                          "R": g[f"{side}_score"], "RA": g[f"{opp}_score"]})
        rows.append(t)
    t = pd.concat(rows, ignore_index=True)
    t["W"] = (t.R > t.RA).astype(int)
    idx = pd.MultiIndex.from_frame(t[["game_key", "team"]])
    f5 = inn[[c for c in inn.columns if c <= 5]].sum(1)
    t["R_F5"] = f5.reindex(idx).to_numpy()
    t["R_INN1"] = inn[1].reindex(idx).to_numpy() if 1 in inn else np.nan
    return t
