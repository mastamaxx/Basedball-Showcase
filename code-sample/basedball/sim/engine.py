"""The game simulator rebuilt on the sub-models (step 13).

Both teams, batter by batter. Before each batter's turn, in order:
  pitching change   the fielding team's pitcher may be pulled: the starter by the step 7 exit model (with its
                    league tracker), a reliever by the step 8 exit model (with its trackers, 3-batter minimum); a
                    new one is chosen by the step 8 choice model from the game-day relievers not yet used
  substitution      the batter's spot may have been taken over by a substitute (step 9); the substitute bats with
                    the team's bench-average rates
  steal             the lead runner with an open base ahead may try to steal (step 10 attempt and success)
  between pitches   wild pitch, passed ball, balk, pickoff (out or throwing error), defensive indifference, other
                    advances (step 12), runners moved as they move on those plays league-wide
Then the plate appearance (step 6 model with the in-game terms from the simulated state), its pitch count
(step 7), a dropped third strike (step 12), and where the runners end up (step 11 models for the lead runner,
league shares for the rest); runs, RBI and earned runs; extra innings start with a runner on 2nd; walk-offs end
the game. An inning-ending caught stealing or pickoff ends the batter's turn without a plate appearance.

Inputs: the arrays of tools/step13_inputs.py (or the daily builder), for any set of games. Output per game x sim:
starting batters' lines, starting pitcher and bullpen lines, team runs (full game, first 5, 1st inning) and wins.
"""
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from ..eval.actuals import PA_CLASSES
from ..model import baserunning as BR
from ..model import bullpen as BP
from ..model import events as EV
from ..model import exit as EX
from ..model import steals as ST
from ..model import subs as SB

NC = len(PA_CLASSES)
IX = {c: i for i, c in enumerate(PA_CLASSES)}
SO_, BB_, HBP_, S1, S2, S3, HR_, ROE_, GO_, AO_ = (IX[c] for c in ["SO", "BB", "HBP", "1B", "2B", "3B", "HR", "ROE", "GO", "AO"])
HIT = np.zeros(NC, bool)
HIT[[S1, S2, S3, HR_]] = True
REACH = np.zeros(NC, bool)
REACH[[S1, S2, S3, HR_, BB_, HBP_, ROE_]] = True
BSTAT = ["PA", "1B", "2B", "3B", "HR", "BB", "HBP", "SO", "R", "RBI", "SB", "CS"]
CD_BINS = {"H": 6, "TB": 13, "HR": 4, "1B": 6, "2B": 4, "3B": 3, "XBH": 5, "BB": 5, "SO": 6, "RBI": 7}   # 0..max (last = or more)
CD_CLASSES = {"H": ["1B", "2B", "3B", "HR"], "HR": ["HR"], "1B": ["1B"], "2B": ["2B"], "3B": ["3B"], "XBH": ["2B", "3B", "HR"],
              "BB": ["BB"], "SO": ["SO"]}
PSTAT = ["BF", "OUTS", "H", "HR", "BB", "HBP", "SO", "R", "ER", "PITCHES"]
BI = {k: i for i, k in enumerate(BSTAT)}
PI = {k: i for i, k in enumerate(PSTAT)}
IN_GAME = ["runners", "outs", "score", "inning", "tto", "pitch_count", "starter", "next_batter"]
EV_ORDER = ["Pickoff, runner out", "Pickoff throwing error", "Balk", "Wild pitch", "Passed ball", "Defensive indifference",
            "Other advance, foul-fly error"]
UNEARNED_EV = {"Passed ball", "Pickoff throwing error"}
SUB_POS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "P"]
# league shares for runners the step 11 models don't cover (2021-2024 play-by-play, tools/step13 notes)
GO1_BATTER_OUT = 0.771                         # "batter out" outcome of a ground ball with a runner on 1st
GO1_R2 = {0: [0.104, 0.0, 0.002, 0.850, 0.044], 1: [0.079, 0.0, 0.624, 0.297, 0.0], 2: [0.0, 0.0, 0.001, 0.948, 0.051]}
GO1_R3 = {0: [0.245, 0.0, 0.0, 0.130, 0.625], 1: [0.025, 0.0, 0.0, 0.756, 0.219], 2: [0.0, 0.0, 0.0, 0.062, 0.938]}
GO1_R3_LOADED = {0: [0.422, 0.0, 0.0, 0.003, 0.575], 1: [0.139, 0.0, 0.0, 0.648, 0.213], 2: [0.0, 0.0, 0.0, 0.002, 0.998]}
GO_FC_BATTER_SAFE = 0.82                       # a runner thrown out on a ground ball: the batter reaches
AO_R1 = [0.031, 0.942, 0.027, 0.0, 0.0]        # runner on 1st on a fly out (out, stays, to 2nd, ...)
HELD_AT_3RD = 0.964                            # runner from 2nd "held" on a single: at 3rd (else at 2nd)
R2_TO_3RD_WHEN_FREE = 0.85                     # ground ball, runners on 2nd and 3rd: runner on 2nd moves up when 3rd frees
BATTER_EXTRA = {"1B": (0.040, 0.002), "2B": 0.033}   # batter takes an extra base on a hit (throws, misplays)
OTHER_UNEARNED = 0.028                         # otherwise-earned runs ruled unearned for misplays the model doesn't play


def _probs_pitches(params, X, n_cut):
    th0, dl, beta = params[0], params[1:n_cut], params[n_cut:]
    th = np.concatenate([[th0], th0 + np.cumsum(np.exp(dl))])
    cdf = expit(th[None, :] - (X @ beta)[:, None])
    cdf = np.column_stack([np.zeros(len(X)), cdf, np.ones(len(X))])
    return np.diff(cdf, axis=1)


class Models:
    """every sub-model's weights, loaded once"""

    def __init__(self, inputs_dir="out/step13"):
        d = Path(inputs_dir)
        js = json.load(open("data/model/pa_model.json"))
        W = np.array(js["W"])
        spans = {b: tuple(s) for b, s in zip(js["blocks"], js["spans"])}
        self.W_ing = np.vstack([W[spans[b][0]:spans[b][1]] for b in IN_GAME])
        self.hand = json.load(open(d / "hand_tables.json"))
        pp = json.load(open("data/model/pitches_model.json"))
        self.kmax = int(pp.get("kmax", 12))
        self.pitch_params = {IX[c]: np.array(v) for c, v in pp["params"].items()}
        self.ex_w, exj = EX.load()
        self.ex_tr = exj["tracker"]
        self.ex_fams = exj.get("families", EX.KEPT)        # the saved model's own families
        self.cw, self.ew, bj = BP.load()
        self.rel_tr = bj["exit_tracker"]
        self.sb_w, sbj = SB.load()
        self.sb_fams = sbj["families"]
        self.st_w, sj = ST.load()
        self.st_fams = {k: sj[k]["families"] for k in ("attempt", "success")}
        self.st_tracker = sj.get("attempt_tracker")
        self.br_W, brj = BR.load()
        self.br_fams = brj["families"]
        ev = EV.load()["events"]
        self.ev = {}
        for e, spec in ev.items():
            mv = spec["runner moves, 2021-2024 (dest: -1 none, 0 out, 1-3 base, 4 scored)"]
            tab = np.zeros((3, 5))
            for b in range(3):
                dd = mv.get(f"runner on {['1st', '2nd', '3rd'][b]}", {})
                for k_, v in dd.items():
                    if int(k_) >= 0:
                        tab[b, int(k_)] = v
                if tab[b].sum() == 0:
                    tab[b, b + 1] = 1
                tab[b] /= tab[b].sum()
            self.ev[e] = dict(w=np.array(spec["w"]), unit=spec["unit"], fams=list(spec["families"]), moves=tab)
        self.ev_names = list(EV.EVENTS)
        roe = json.load(open("data/model/roe_moves.json"))
        self.roe_bat = np.array([roe["batter"][str(k)] for k in range(5)])
        self.roe_run = np.array([[roe[f"runner on {n}"][str(k)] for k in range(5)] for n in ("1st", "2nd", "3rd")])
        meta = json.load(open(d / "inputs_meta.json"))
        self.meta = meta
        sj = Path("data/model/sim_shocks.json")
        self.shocks = {k: np.array(v) for k, v in json.load(open(sj))["shocks"].items()} if sj.exists() else None


def pregame_tables(A, M):
    """log-probabilities before the in-game terms, for every batter spot (and his substitute) vs every pitcher of
    the other team: (G, 2 batting side, 9, K, 10)"""
    G, K = A["rp"].shape[0], A["rp"].shape[2]
    LC = np.zeros((G, 2, 9, K, NC), np.float32)
    LCs = np.zeros((G, 2, 9, K, NC), np.float32)
    seasons = A["hand_season"]
    for s in (0, 1):
        o = 1 - s
        h = A["pit_hand"][:, o]                                            # (G, K) 0 L, 1 R
        side = A["batter_side"][:, s]                                      # (G, 9) 0 L, 1 R, 2 switch
        eff = np.where(side[:, :, None] == 2, 1 - h[:, None, :], side[:, :, None])   # (G, 9, K)
        hf = np.ones((G, 9, K, NC))
        for season in np.unique(seasons):
            tab = M.hand[str(int(season))]
            ms = (seasons == season)[:, None, None]
            for b, bn in enumerate("LR"):
                for ph, pn in enumerate("LR"):
                    m = ms & (eff == b) & (h[:, None, :] == ph)
                    hf[m] = np.array(tab[f"{bn}|{pn}"])
        lg = A["lg"][:, s][:, :, None, :].astype(np.float64)
        rpx = np.where(A["rp_has"][:, o][:, None, :, None], A["rp"][:, o][:, None, :, :], lg)
        pt = np.where(eff[..., None] == 0, A["ptm"][:, o][:, None, :, 0, :], A["ptm"][:, o][:, None, :, 1, :])
        pl = np.where(h[:, None, :, None] == 0, A["platb"][:, s][:, :, None, 0, :], A["platb"][:, s][:, :, None, 1, :])
        c = A["rb"][:, s][:, :, None, :] * rpx / lg * hf * A["pf"][:, s][:, :, None, :]
        c /= c.sum(-1, keepdims=True)
        LC[:, s] = np.log(np.clip(c, 1e-12, 1)) + A["eta"][:, s][:, :, None, :] + pl + pt
        cs = A["rb_bench"][:, s][:, None, None, :] * rpx / lg * hf * A["pf"][:, s][:, :, None, :]
        cs /= cs.sum(-1, keepdims=True)
        LCs[:, s] = np.log(np.clip(cs, 1e-12, 1)) + A["eta_sub"][:, s][:, :, None, :] + pt
    return LC, LCs


def _bins(x, edges):
    return [((x >= lo) & (x <= hi)) for lo, hi in edges]


def _draw(rng, p):
    """one draw per row of a probability matrix"""
    c = np.cumsum(p, 1)
    return np.minimum((rng.random(len(p))[:, None] > c).sum(1), p.shape[1] - 1)


class Game:
    """the state of every lane (game x sim) of a chunk"""

    def __init__(self, A, M, n_sims, rng, shocks=None, shock_scale=1.0, dists=True):
        self.A, self.M, self.rng = A, M, rng
        G = len(A["rb"])
        self.G, self.S = G, n_sims
        L = G * n_sims
        self.L = L
        self.gl = np.repeat(np.arange(G), n_sims)
        K = A["rp"].shape[2]
        self.K = K
        self.LC, self.LCs = pregame_tables(A, M)
        self.alive = np.ones(L, bool)
        self.half = np.zeros(L, np.int8)
        self.inning = np.ones(L, np.int16)
        self.outs = np.zeros(L, np.int8)
        self.occ = np.full((L, 3), -1, np.int8)
        self.osub = np.zeros((L, 3), bool)
        self.oresp = np.zeros((L, 3), np.int8)
        self.oearn = np.ones((L, 3), bool)
        self.score = np.zeros((L, 2), np.int16)
        self.slot = np.zeros((L, 2), np.int8)
        self.npa = np.zeros((L, 2, 9), np.int8)
        self.repl = np.zeros((L, 2, 9), bool)
        self.subs_used = np.zeros((L, 2), np.int8)
        self.last_inn = np.zeros((L, 2, 9), np.int16)
        self.prev_reached = np.zeros((L, 2, 9), bool)
        self.last_pit = np.zeros((L, 2, 9), np.int8)
        self.cur = np.zeros((L, 2), np.int8)
        self.used = np.zeros((L, 2, K), bool)
        self.used[:, :, 0] = True
        self.bf_k = np.zeros((L, 2, K), np.int16)          # batters faced by each candidate pitcher
        self.napp = np.zeros((L, 2), np.int8)
        self.first_change = np.ones((L, 2), bool)
        self.start_inn = np.ones((L, 2), bool)
        f = lambda: np.zeros((L, 2), np.float32)
        self.sp = {k: f() for k in ("bf", "pc", "pc_inn", "runs", "h", "bb", "hr", "k")}
        self.sp_last3 = np.zeros((L, 2, 3), bool)
        self.rp = {k: f() for k in ("bf", "pc", "outs", "runs", "entry", "clean")}
        self.runs_inn = np.zeros(L, np.float32)
        self.err_outs = np.zeros(L, np.int8)                   # outs the fielders should have made this half (errors)
        self.por = np.zeros((L, 2), np.int8)
        self.bat = np.zeros((L, 2, 9, len(BSTAT)), np.int16)
        self.pit = np.zeros((L, 2, 2, len(PSTAT)), np.int16)
        self.sp_out = np.zeros((L, 2), bool)                   # starter pulled
        self.inn_runs = np.zeros((L, 2, 10), np.int16)         # runs in innings 1-9, then all extra innings
        self.sched = A["innings_sched"][self.gl].astype(np.int16)
        gr = A.get("ghost_runner")                             # extra-inning runner on 2nd (regular season only)
        self.ghost = np.ones(L, bool) if gr is None else np.asarray(gr, bool)[self.gl]
        # starting batters' count stats as distributions given each simulated game's plate appearances (the chance
        # of each outcome at each of his plate appearances, convolved): far less simulation noise than counts
        self.cd = {}
        self.dists = dists
        for nm, nb in (CD_BINS.items() if dists else []):
            z = np.zeros((L, 2, 9, nb), np.float32)
            z[..., 0] = 1
            self.cd[nm] = z
        # game-to-game swings beyond the plate-appearance model (tools/step13_dispersion.py): a shift shared by the
        # whole game, one per batting team, one for each starter's outing; scale multiplies their spread
        self.shock_g = self.shock_t = self.shock_p = None
        scales = shock_scale if isinstance(shock_scale, dict) else {k: shock_scale for k in ("whole game", "batting team", "starter")}
        if shocks is not None and max(scales.values()) > 0:
            def draw(name, shape):
                cov = np.asarray(shocks[name]) * scales[name] ** 2
                w, v = np.linalg.eigh(cov)
                f = v * np.sqrt(np.maximum(w, 0))
                # centred so each outcome's average chance is unchanged (E[exp(shift)] = 1)
                return (rng.standard_normal(shape + (NC,)) @ f.T - np.diag(cov) / 2).astype(np.float32)
            self.shock_g = draw("whole game", (L,))
            self.shock_t = draw("batting team", (L, 2))
            self.shock_p = draw("starter", (L, 2))

    # ------------------------------------------------------------------ small helpers
    def team(self, a):
        bt = self.half[a].astype(int)
        return bt, 1 - bt

    def cur_pc(self, a, ft):
        k = self.cur[a, ft]
        return np.where(k == 0, self.sp["pc"][a, ft], self.rp["pc"][a, ft])

    def side_vs(self, a, bt, spot, k, ft):
        """the batter's side against this pitcher (0 L / 1 R) and whether it's the same as the pitcher's hand"""
        A, g = self.A, self.gl[a]
        hand = A["pit_hand"][g, ft, k]
        side = A["batter_side"][g, bt, spot]
        eff = np.where(side == 2, 1 - hand, side)
        same = (side != 2) & (side == hand)
        return eff, hand, same

    def q_of(self, a, bt, spot):
        g = self.gl[a]
        return np.where(self.repl[a, bt, spot], self.A["bench"][g, bt, 2], self.A["q"][g, bt, spot])

    # ------------------------------------------------------------------ runs, outs, runners
    def score_runs(self, a, bt, ft, scorers, play_earned, rbi_mask=None):
        """scorers: (n, 3) bool, runners on 1st/2nd/3rd who score; credits R, team runs, pitcher R / ER"""
        n = scorers.sum(1)
        if not n.any():
            return n
        g = self.gl[a]
        for b in range(3):
            m = scorers[:, b]
            if not m.any():
                continue
            aa, bb_, ff = a[m], bt[m], ft[m]
            sp_ = self.occ[aa, b]
            st = ~self.osub[aa, b]
            self.bat[aa[st], bb_[st], sp_[st], BI["R"]] += 1
            resp = self.oresp[aa, b]
            slot_ = np.minimum(resp, 1)
            self.pit[aa, ff, slot_, PI["R"]] += 1
            # earned: the runner didn't reach on an error, didn't score on one, and the inning as it should have
            # gone (errors counted as outs) wasn't already over
            er = self.oearn[aa, b] & play_earned[m] & (self.outs[aa] + self.err_outs[aa] < 3)
            er &= self.rng.random(len(aa)) >= OTHER_UNEARNED
            self.pit[aa[er], ff[er], slot_[er], PI["ER"]] += 1
        self.add_runs(a, bt, ft, n)
        if rbi_mask is not None:
            st = ~self.repl[a, bt, self.slot[a, bt]]
            r = np.where(rbi_mask, n, 0)
            self.bat[a[st], bt[st], self.slot[a[st], bt[st]], BI["RBI"]] += r[st].astype(np.int16)
        return n

    def add_runs(self, a, bt, ft, n):
        before_lead = self.score[a, bt] > self.score[a, ft]
        self.score[a, bt] += n.astype(np.int16)
        self.runs_inn[a] += n
        self.inn_runs[a, bt, np.minimum(self.inning[a], 10) - 1] += n.astype(np.int16)
        k = self.cur[a, ft]
        self.sp["runs"][a, ft] += np.where(k == 0, n, 0)
        self.rp["runs"][a, ft] += np.where(k > 0, n, 0)
        took = ~before_lead & (self.score[a, bt] > self.score[a, ft])
        self.por[a[took], bt[took]] = self.cur[a[took], bt[took]]

    def add_outs(self, a, ft, n):
        k = self.cur[a, ft]
        slot_ = np.minimum(k, 1)
        self.pit[a, ft, slot_, PI["OUTS"]] += n.astype(np.int16)
        self.rp["outs"][a, ft] += np.where(k > 0, n, 0)
        self.outs[a] += n.astype(np.int8)

    def set_bases(self, a, new_occ, new_sub, new_resp, new_earn):
        self.occ[a], self.osub[a], self.oresp[a], self.oearn[a] = new_occ, new_sub, new_resp, new_earn

    def move(self, a, bt, ft, dest_r, dest_b, batter_spot, batter_sub, batter_earned, play_earned, rbi_mask,
             runs_count=None):
        """dest_r (n, 3): destination of the runners on 1st, 2nd, 3rd (-1 none, 0 out, 1-3 base, 4 scores); dest_b:
        the batter's (-1 stays at the plate, 0 out, 1-4). Fixes collisions (a runner can't pass or share a base
        with the runner ahead), applies outs, runs, RBI, earned runs and the new bases."""
        n = len(a)
        dest_r = dest_r.copy()
        occ = self.occ[a]
        has = occ >= 0
        dest_r[~has] = -1
        # blocking: a runner can't pass, or stop on the same base as, the runner ahead of him
        limit = np.full(n, 4)
        for b in (2, 1, 0):
            d = dest_r[:, b]
            on = d >= 1
            d = np.where(on & (d < 4) & (d >= limit), np.maximum(limit - 1, b + 1), d)
            dest_r[:, b] = d
            limit = np.where(on & (d < 4), d, limit)
        db = dest_b.copy()
        # forcing: a base taken by someone behind pushes the runner there up a base
        claimed = np.where((db >= 1) & (db <= 3), db, 0)
        for b in (0, 1, 2):
            d = dest_r[:, b]
            on = (d >= 1) & (d <= 3)
            d = np.where(on & (d <= claimed), np.minimum(claimed + 1, 4), d)
            dest_r[:, b] = d
            claimed = np.where((d >= 1) & (d <= 3), np.maximum(claimed, d), claimed)
        outs_add = (dest_r == 0).sum(1) + (db == 0)
        cnt = np.ones(n, bool) if runs_count is None else runs_count
        scorers = (dest_r == 4) & cnt[:, None]
        new_outs = np.minimum(self.outs[a] + outs_add, 3)
        self.add_outs(a, ft, (new_outs - self.outs[a]).astype(np.int8))
        nr = self.score_runs(a, bt, ft, scorers, play_earned, rbi_mask)
        # batter scoring (home run)
        hb = (db == 4) & cnt
        if hb.any():
            aa, bb_, ff = a[hb], bt[hb], ft[hb]
            st = ~batter_sub[hb]
            self.bat[aa[st], bb_[st], batter_spot[hb][st], BI["R"]] += 1
            k = self.cur[aa, ff]
            self.pit[aa, ff, np.minimum(k, 1), PI["R"]] += 1
            erb = (self.outs[aa] + self.err_outs[aa] < 3) & (self.rng.random(len(aa)) >= OTHER_UNEARNED)
            self.pit[aa[erb], ff[erb], np.minimum(k, 1)[erb], PI["ER"]] += 1
            self.add_runs(aa, bb_, ff, np.ones(len(aa)))
            if rbi_mask is not None:
                rb = rbi_mask[hb] & st
                self.bat[aa[rb], bb_[rb], batter_spot[hb][rb], BI["RBI"]] += 1
        # new bases
        new_occ = np.full((n, 3), -1, np.int8)
        new_sub = np.zeros((n, 3), bool)
        new_resp = np.zeros((n, 3), np.int8)
        new_earn = np.ones((n, 3), bool)
        for b in range(3):
            for t in (1, 2, 3):
                m = dest_r[:, b] == t
                new_occ[m, t - 1] = occ[m, b]
                new_sub[m, t - 1] = self.osub[a[m], b]
                new_resp[m, t - 1] = self.oresp[a[m], b]
                new_earn[m, t - 1] = self.oearn[a[m], b]
        for t in (1, 2, 3):
            m = db == t
            new_occ[m, t - 1] = batter_spot[m]
            new_sub[m, t - 1] = batter_sub[m]
            new_resp[m, t - 1] = self.cur[a[m], ft[m]]
            new_earn[m, t - 1] = batter_earned[m]
        over = new_outs >= 3
        new_occ[over] = -1
        self.set_bases(a, new_occ, new_sub, new_resp, new_earn)
        return nr

    # ------------------------------------------------------------------ 1. pitching change
    def pitching(self, a):
        A, M, g = self.A, self.M, self.gl[a]
        bt, ft = self.team(a)
        spot = self.slot[a, bt]
        k = self.cur[a, ft]
        pull = np.zeros(len(a), bool)
        # starter
        s = (k == 0) & (self.sp["bf"][a, ft] >= 1) & ~self.sp_out[a, ft]
        if s.any():
            aa, b_, f_, gg, sp_ = a[s], bt[s], ft[s], g[s], spot[s]
            eff, hand, same = self.side_vs(aa, b_, sp_, np.zeros(len(aa), int), f_)
            d = {"pitches_so_far": self.sp["pc"][aa, f_], "start_inning": self.start_inn[aa, f_].astype(float),
                 "bf": self.sp["bf"][aa, f_], "tto": self.sp["bf"][aa, f_] // 9 + 1, "inning": self.inning[aa].astype(float),
                 "runs_so_far": self.sp["runs"][aa, f_], "runs_inning": self.runs_inn[aa],
                 "runners_on": (self.occ[aa] >= 0).sum(1).astype(float), "pitches_inning": self.sp["pc_inn"][aa, f_],
                 "reached_last3": self.sp_last3[aa, f_].sum(1).astype(float),
                 "margin": (self.score[aa, f_] - self.score[aa, b_]).astype(float), "hits_so_far": self.sp["h"][aa, f_],
                 "bb_so_far": self.sp["bb"][aa, f_], "hr_so_far": self.sp["hr"][aa, f_], "k_so_far": self.sp["k"][aa, f_],
                 "q": self.q_of(aa, b_, sp_), "q_next": self.q_of(aa, b_, (sp_ + 1) % 9), "same_side": same.astype(float)}
            pre = A["sp_pregame"][gg, f_]
            for j, c in enumerate(M.meta["pregame"]):
                d[c] = pre[:, j]
            Xe, _ = EX.design(d, M.ex_fams)
            z = EX.logit(M.ex_w, Xe)
            late = self.sp["pc"][aa, f_] >= M.ex_tr["late_pitches"]
            z = z + M.ex_tr["strength"] * A["sp_miss"][gg] * late
            if "exit_post" in A:                     # postseason term (level, x pitches over 75 / 10), per game
                z = z + A["exit_post"][gg, 0] + A["exit_post"][gg, 1] * EX.hinge(self.sp["pc"][aa, f_].astype(float), 75, 10)
            p_ = self.rng.random(len(aa)) < expit(z)
            p_ &= ~((self.sp["bf"][aa, f_] < 3) & ~self.start_inn[aa, f_])
            pull[s] = p_
        # reliever
        r = (k > 0) & (self.rp["bf"][a, ft] >= 1)
        if r.any():
            aa, b_, f_, gg, sp_, kk = a[r], bt[r], ft[r], g[r], spot[r], k[r]
            cx = A["cand_x"][gg, f_, kk]
            cc = M.meta["cand_cols"]
            eff, hand, same = self.side_vs(aa, b_, sp_, kk, f_)
            _, _, same_n = self.side_vs(aa, b_, (sp_ + 1) % 9, kk, f_)
            stt = self.start_inn[aa, f_]
            d = {"bf_out": self.rp["bf"][aa, f_], "start_inn": stt.astype(float), "inning": self.inning[aa].astype(float),
                 "margin": (self.score[aa, f_] - self.score[aa, b_]).astype(float), "pitches_out": self.rp["pc"][aa, f_],
                 "lefty": (hand == 0).astype(float),
                 "innings_started": self.inning[aa] - self.rp["entry"][aa, f_] + self.rp["clean"][aa, f_],
                 "outs_pre": self.outs[aa].astype(float), "runs_out": self.rp["runs"][aa, f_], "runs_inning": self.runs_inn[aa],
                 "runners": (self.occ[aa] >= 0).sum(1).astype(float), "outs_out": self.rp["outs"][aa, f_],
                 "same_side": same.astype(float), "same_side_next": same_n.astype(float),
                 "team_pen_1d": A["team_pen"][gg, f_, 0], "team_pen_3d": A["team_pen"][gg, f_, 1],
                 "app": self.napp[aa, f_].astype(float)}
            for c in ("outs_365", "multi_365", "start_share", "finished_365", "save_365", "close_late_365", "p1", "p2",
                      "k_rate", "woba_rate"):
                d[c] = cx[:, cc.index(c)]
            Xe, _ = BP.exit_design(d)
            z = EX.logit(M.ew, Xe)
            free = ~(~stt & (self.rp["bf"][aa, f_] < 3))
            st_, md_ = M.rel_tr["strength"]
            z = z + st_ * A["rel_miss"][gg, 0] * (free & stt) + md_ * A["rel_miss"][gg, 1] * (free & ~stt)
            p_ = (self.rng.random(len(aa)) < expit(z)) & free
            pull[r] = p_
        # someone left to bring in
        left = (A["avail"][g, ft] & ~self.used[a, ft]).any(1)
        pull &= left
        if pull.any():
            self.choose(a[pull], bt[pull], ft[pull])

    def choose(self, a, bt, ft):
        A, M, g = self.A, self.M, self.gl[a]
        K = self.K
        n = len(a)
        spot = self.slot[a, bt]
        avail = A["avail"][g, ft] & ~self.used[a, ft]
        avail[:, 0] = False
        rows, kk = np.nonzero(np.ones((n, K), bool))
        lg_, lb, lf, la = g[rows], bt[rows], ft[rows], a[rows]
        cc = M.meta["cand_cols"]
        cx = A["cand_x"][lg_, lf, kk]
        d = {c: cx[:, j] for j, c in enumerate(cc)}
        hand_k = A["pit_hand"][lg_, lf, kk]
        d["usage_share"] = np.maximum(A["usage"][lg_, lf, kk], 1e-9)
        d["lefty"] = (hand_k == 0).astype(float)
        sides = np.stack([A["batter_side"][g, bt, (spot + j) % 9] for j in range(3)], 1)[rows]
        d.update({"inning": self.inning[la].astype(float), "margin": (self.score[la, lf] - self.score[la, lb]).astype(float),
                  "prev_starter": self.first_change[la, lf].astype(float), "start_inn": self.start_inn[la, lf].astype(float),
                  "runners": (self.occ[la] >= 0).sum(1).astype(float),
                  "same_side3": (sides == hand_k[:, None]).sum(1).astype(float), "left_due3": (sides == 0).sum(1).astype(float)})
        Xc, _ = BP.choice_design(d)
        zc = (Xc @ M.cw).reshape(n, K)
        if "choice_post" in A:                   # postseason: managers lean on their top relievers (data/model/postseason.json)
            zc = zc + A["choice_post"][g, ft]
        zc = np.where(avail, zc, -np.inf)
        zc -= zc.max(1, keepdims=True)
        pc = np.exp(zc)
        pc /= pc.sum(1, keepdims=True)
        pick = _draw(self.rng, pc)
        self.sp_out[a[self.cur[a, ft] == 0], ft[self.cur[a, ft] == 0]] = True
        self.cur[a, ft] = pick
        self.used[a, ft, pick] = True
        self.napp[a, ft] += 1
        self.first_change[a, ft] = False
        for k_ in ("bf", "pc", "outs", "runs"):
            self.rp[k_][a, ft] = 0
        self.rp["entry"][a, ft] = self.inning[a]
        self.rp["clean"][a, ft] = self.start_inn[a, ft].astype(np.float32)

    # ------------------------------------------------------------------ 2. substitution
    def substitution(self, a):
        A, M, g = self.A, self.M, self.gl[a]
        bt, ft = self.team(a)
        spot = self.slot[a, bt]
        chk = ~self.repl[a, bt, spot] & (self.npa[a, bt, spot] >= 1)
        if not chk.any():
            return
        aa, b_, f_, gg, sp_ = a[chk], bt[chk], ft[chk], g[chk], spot[chk]
        k = self.cur[aa, f_]
        eff, hand, same = self.side_vs(aa, b_, sp_, k, f_)
        sbc = M.meta["sb_cols"]
        x = A["sb_x"][gg, b_, sp_]
        occ = self.occ[aa] >= 0
        bench = A["bench"][gg, b_]
        d = {"t": self.npa[aa, b_, sp_] + 1.0, "inning": self.inning[aa].astype(float),
             "margin": (self.score[aa, b_] - self.score[aa, f_]).astype(float), "outs_pre": self.outs[aa].astype(float),
             "runners": occ.sum(1).astype(float), "risp": (occ[:, 1] | occ[:, 2]).astype(float),
             "innings_since": (self.inning[aa] - self.last_inn[aa, b_, sp_]).astype(float),
             "prev_reached": self.prev_reached[aa, b_, sp_].astype(float), "same_side": same.astype(float),
             "pitcher_new": (k != self.last_pit[aa, b_, sp_]).astype(float), "reliever": (k > 0).astype(float),
             "bench_best": bench[:, 1], "bench_left": np.maximum(bench[:, 0] - self.subs_used[aa, b_], 0),
             "subs_used": self.subs_used[aa, b_].astype(float), "team_removed_rate": A["team_removed"][gg, b_],
             "position": np.array(SUB_POS)[A["sb_pos"][gg, b_, sp_]],
             "il_return": np.zeros(len(aa)), "age": np.full(len(aa), 28.0), "days_since_game": np.ones(len(aa)),
             "games_so_far": np.full(len(aa), 50.0)}
        for j, c in enumerate(sbc):
            d[c] = x[:, j]
        base, fam = SB.blocks(d)
        X, _ = SB.design(base, fam, M.sb_fams)
        p = expit(EX.logit(M.sb_w, X))
        out = self.rng.random(len(aa)) < p
        self.repl[aa[out], b_[out], sp_[out]] = True
        self.subs_used[aa[out], b_[out]] += 1

    # ------------------------------------------------------------------ 3. steal
    def runner_attr(self, a, bt, base):
        """steal and baserunning attributes of the runner on a base (league defaults for substitutes)"""
        A, g = self.A, self.gl[a]
        sp_ = np.maximum(self.occ[a, base], 0)
        sub = self.osub[a, base]
        st = A["st_run"][g, bt, sp_]
        dflt = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 27.0, 1.0])
        st = np.where(sub[:, None], dflt, st)
        return st, np.where(sub, 27.0, A["br_speed"][g, bt, sp_]), np.where(sub, 1.0, A["br_speed_missing"][g, bt, sp_]), \
            np.where(sub, 1.0, A["xbt"][g, bt, sp_])

    def steal(self, a):
        A, M, g = self.A, self.M, self.gl[a]
        bt, ft = self.team(a)
        b1, b2, b3 = (self.occ[a, j] >= 0 for j in range(3))
        t2 = b1 & ~b2
        t3 = b2 & ~b3 & ~t2
        opp = (t2 | t3) & (self.outs[a] < 3)
        if not opp.any():
            return
        aa, b_, f_, gg = a[opp], bt[opp], ft[opp], g[opp]
        base = np.where(t2[opp], 0, 1)
        target = np.where(base == 0, "2nd", "3rd")
        k = self.cur[aa, f_]
        spot = self.slot[aa, b_]
        eff, hand, same = self.side_vs(aa, b_, spot, k, f_)
        st, _, _, _ = self.runner_attr(aa, b_, base)
        rc = self.M.meta["st_run_cols"]
        sub_b = self.repl[aa, b_, spot]
        bk = np.where(sub_b[:, None], np.array([0.22, 0.09, 0.31]), A["st_bat"][gg, b_, spot])
        cat = A["st_cat"][gg, f_]
        d = {"target": target, "era2023": A["era2023"][gg], "pit_hand": np.where(hand == 0, "L", "R"),
             "bat_hand": np.where(eff == 0, "L", "R"), "inning": self.inning[aa].astype(float),
             "margin": (self.score[aa, b_] - self.score[aa, f_]).astype(float), "outs": self.outs[aa].astype(float),
             "pitcher_att_ratio": A["st_pit"][gg, f_, k], "starter": (k == 0).astype(float),
             "pop": cat[:, 0], "pop_missing": cat[:, 1], "catcher_succ_ratio": cat[:, 2], "arm": np.full(len(aa), 80.0),
             "catcher_att_ratio": np.ones(len(aa)), "pitcher_succ_ratio": np.ones(len(aa)),
             "bat_k": bk[:, 0], "bat_bb": bk[:, 1], "bat_woba": bk[:, 2],
             "other_on": np.where(base == 0, self.occ[aa, 2] >= 0, self.occ[aa, 0] >= 0).astype(float),
             "team_att_ratio": A["st_team"][gg, b_]}
        for j, c in enumerate(rc):
            d[c] = st[:, j]
        Xa, _ = ST.design(d, "attempt", M.st_fams["attempt"])
        za = EX.logit(M.st_w["attempt"], Xa)
        if "st_level" in A and M.st_tracker:                       # step 18: the league's attempt level so far this season
            za = za + M.st_tracker["strength"] * A["st_level"][gg]
        pa_ = expit(za)
        go = self.rng.random(len(aa)) < pa_
        if not go.any():
            return
        Xs, _ = ST.design(d, "success", M.st_fams["success"])
        ps = expit(EX.logit(M.st_w["success"], Xs))
        safe = self.rng.random(len(aa)) < ps
        aa, b_, f_, base, safe = aa[go], b_[go], f_[go], base[go], safe[go]
        n = len(aa)
        dest = np.full((n, 3), -1)
        for j in range(3):
            dest[:, j] = np.where(self.occ[aa, j] >= 0, j + 1, -1)
        dest[np.arange(n), base] = np.where(safe, base + 2, 0)
        sp_r = self.occ[aa, base]
        st_r = ~self.osub[aa, base]
        self.bat[aa[safe & st_r], b_[safe & st_r], sp_r[safe & st_r], BI["SB"]] += 1
        self.bat[aa[~safe & st_r], b_[~safe & st_r], sp_r[~safe & st_r], BI["CS"]] += 1
        self.move(aa, b_, f_, dest, np.full(n, -1), np.zeros(n, np.int8), np.zeros(n, bool), np.ones(n, bool),
                  np.ones(n, bool), None)

    # ------------------------------------------------------------------ 4. between-pitch events
    def events(self, a):
        A, M, g = self.A, self.M, self.gl[a]
        bt, ft = self.team(a)
        occ = self.occ[a] >= 0
        live = occ.any(1) & (self.outs[a] < 3)
        if not live.any():
            return
        k = self.cur[a, ft]
        d = {"outs_pre": self.outs[a].astype(float), "base1_pre": occ[:, 0].astype(int), "base2_pre": occ[:, 1].astype(int),
             "base3_pre": occ[:, 2].astype(int), "inning": self.inning[a].astype(float),
             "bat_score_pre": self.score[a, bt].astype(float), "fld_score_pre": self.score[a, ft].astype(float)}
        draws = {}
        for j, e in enumerate(M.ev_names):
            if e.startswith("Dropped"):
                continue
            spec = M.ev[e]
            unit = EV.unit_mask(d, spec["unit"]) & live
            if not unit.any():
                continue
            ratios = {}
            for f in spec["fams"]:
                ratios[f] = A["ev_pit"][g, ft, k, j] if f == "Pitcher" else A["ev_cat"][g, ft, j]
            X, _ = EV.design(d, e, A["ev_lr"][g, j], ratios, spec["fams"])
            p = expit(EX.logit(spec["w"], X))
            draws[e] = unit & (self.rng.random(len(a)) < p)
        for e in EV_ORDER:
            if e not in draws:
                continue
            m = draws[e] & (self.outs[a] < 3) & (self.occ[a] >= 0).any(1)
            if not m.any():
                continue
            aa, b_, f_ = a[m], bt[m], ft[m]
            n = len(aa)
            mv = M.ev[e]["moves"]
            occ_m = self.occ[aa] >= 0
            dest = np.where(occ_m, np.arange(1, 4)[None, :], -1)
            if e == "Pickoff, runner out":
                w = np.where(occ_m, mv[:, 0][None, :] + 1e-6, 0)
                who = _draw(self.rng, w / w.sum(1, keepdims=True))
                dest[np.arange(n), who] = 0
            else:
                for b in range(3):
                    db = _draw(self.rng, np.repeat(mv[b][None, :], n, 0))
                    dest[:, b] = np.where(occ_m[:, b], np.maximum(db, np.where(db == 0, 0, b + 1)), -1)
            earned = np.full(n, e not in UNEARNED_EV)
            self.move(aa, b_, f_, dest, np.full(n, -1), np.zeros(n, np.int8), np.zeros(n, bool), np.ones(n, bool), earned, None)

    # ------------------------------------------------------------------ 5. plate appearance
    def pa(self, a):
        A, M, g = self.A, self.M, self.gl[a]
        bt, ft = self.team(a)
        n = len(a)
        spot = self.slot[a, bt]
        sub = self.repl[a, bt, spot]
        k = self.cur[a, ft]
        occ = self.occ[a] >= 0
        b1, b2, b3 = occ[:, 0], occ[:, 1], occ[:, 2]
        risp = b2 | b3
        outs = self.outs[a]
        mb = (self.score[a, bt] - self.score[a, ft]).astype(int)
        inn = self.inning[a]
        tto = np.where(k == 0, self.sp["bf"][a, ft] // 9 + 1, 0)
        pcg = self.cur_pc(a, ft)
        qn = self.q_of(a, bt, (spot + 1) % 9)
        cols = [b1 & ~risp, risp & ~b1, risp & b1, outs == 1, outs == 2,
                mb <= -5, (mb >= -4) & (mb <= -2), mb == -1, mb == 1, (mb >= 2) & (mb <= 4), mb >= 5,
                (inn >= 2) & (inn <= 3), (inn >= 4) & (inn <= 6), (inn >= 7) & (inn <= 8), inn >= 9,
                (k == 0) & (tto == 2), (k == 0) & (tto == 3), (k == 0) & (tto >= 4),
                (pcg > 25) & (pcg <= 50), (pcg > 50) & (pcg <= 75), (pcg > 75) & (pcg <= 100), pcg > 100,
                k == 0, qn, qn * (~b1 & risp), qn * (outs == 2), qn * (b1 | b2 | b3)]
        Fm = np.column_stack([np.asarray(c, np.float64) for c in cols])
        lp = np.where(sub[:, None], self.LCs[g, bt, spot, k], self.LC[g, bt, spot, k]) + Fm @ M.W_ing
        if self.shock_g is not None:
            lp = lp + self.shock_g[a] + self.shock_t[a, bt] + np.where((k == 0)[:, None], self.shock_p[a, ft], 0)
        if "team_tilt" in A:                     # optional per-game, per-batting-team log-odds shift (market blend test)
            lp = lp + A["team_tilt"][g, bt]
        if "team_pa" in A:                       # team-wide terms: defense, schedule, conversion (model/team_terms.py)
            lp = lp + A["team_pa"][g, bt]
        lp -= lp.max(1, keepdims=True)
        p = np.exp(lp)
        p /= p.sum(1, keepdims=True)
        o = _draw(self.rng, p)
        # pitches
        x = np.column_stack([np.where(sub, 0.0, A["btend"][g, bt, spot]), A["ptend"][g, ft, k]])
        npit = np.full(n, 4.0)
        u = self.rng.random(n)
        for c, prm in M.pitch_params.items():
            m = o == c
            if m.any():
                pp = _probs_pitches(prm, x[m], M.kmax - 1)
                npit[m] = 1 + (u[m][:, None] > np.cumsum(pp, 1)).sum(1)
        top = npit >= M.kmax
        if top.any():
            npit[top] += self.rng.geometric(0.55, top.sum()) - 1
        # pitcher and batter lines
        slot_ = np.minimum(k, 1)
        P_ = self.pit
        P_[a, ft, slot_, PI["BF"]] += 1
        np.add.at(self.bf_k, (a, ft, k), 1)
        P_[a, ft, slot_, PI["PITCHES"]] += npit.astype(np.int16)
        P_[a, ft, slot_, PI["H"]] += HIT[o]
        P_[a, ft, slot_, PI["HR"]] += o == HR_
        P_[a, ft, slot_, PI["BB"]] += o == BB_
        P_[a, ft, slot_, PI["HBP"]] += o == HBP_
        P_[a, ft, slot_, PI["SO"]] += o == SO_
        st = ~sub
        Bt = self.bat
        aa, bb_, ss = a[st], bt[st], spot[st]
        oo = o[st]
        Bt[aa, bb_, ss, BI["PA"]] += 1
        if self.dists:
            self.convolve(aa, bb_, ss, p[st])
        for c, nm in ((S1, "1B"), (S2, "2B"), (S3, "3B"), (HR_, "HR"), (BB_, "BB"), (HBP_, "HBP"), (SO_, "SO")):
            Bt[aa, bb_, ss, BI[nm]] += oo == c
        # pitcher state
        s0 = k == 0
        for key, v in (("bf", 1.0), ("pc", npit), ("pc_inn", npit), ("h", HIT[o]), ("bb", (o == BB_) | (o == HBP_)),
                       ("hr", o == HR_), ("k", o == SO_)):
            self.sp[key][a[s0], ft[s0]] += np.asarray(v if np.ndim(v) else np.full(n, v), np.float32)[s0]
        self.sp_last3[a[s0], ft[s0]] = np.column_stack([self.sp_last3[a[s0], ft[s0]][:, 1:], REACH[o][s0]])
        r0 = ~s0
        self.rp["bf"][a[r0], ft[r0]] += 1
        self.rp["pc"][a[r0], ft[r0]] += npit[r0]
        # turn bookkeeping
        self.npa[a, bt, spot] += 1
        self.last_inn[a, bt, spot] = inn
        self.prev_reached[a, bt, spot] = REACH[o]
        self.last_pit[a, bt, spot] = k
        self.start_inn[a, ft] = False
        self.resolve(a, bt, ft, o, spot, sub)
        self.slot[a, bt] = (spot + 1) % 9

    def convolve(self, aa, bb_, ss, ps):
        """add one plate appearance's outcome chances to the starting batters' count-stat distributions"""
        for nm, cl in CD_CLASSES.items():
            q = ps[:, [IX[c] for c in cl]].sum(1)[:, None]
            d = self.cd[nm][aa, bb_, ss]
            new = d * (1 - q)
            new[:, 1:] += d[:, :-1] * q
            new[:, -1] += d[:, -1] * q[:, 0]
            self.cd[nm][aa, bb_, ss] = new
        # RBI: the chance of 0-4 runs batted in on this plate appearance, from the outcome chances and the step 11
        # baserunning chances for the runners on (as tools/step11_value.py), convolved the same way
        D = self.rbi_pa_dist(aa, bb_, 1 - bb_, ps)
        d = self.cd["RBI"][aa, bb_, ss]
        new = np.zeros_like(d)
        for v in range(D.shape[1]):
            sh = np.zeros_like(d)
            if v:
                sh[:, v:] = d[:, :-v]
                sh[:, -1] += d[:, -v:].sum(1)
            else:
                sh = d
            new += sh * D[:, [v]]
        self.cd["RBI"][aa, bb_, ss] = new
        d = self.cd["TB"][aa, bb_, ss]
        new = d * (1 - ps[:, [S1, S2, S3, HR_]].sum(1))[:, None]
        for v, c in ((1, S1), (2, S2), (3, S3), (4, HR_)):
            sh = np.zeros_like(d)
            sh[:, v:] = d[:, :-v]
            sh[:, -1] += d[:, -v:].sum(1)
            new += sh * ps[:, [c]]
        self.cd["TB"][aa, bb_, ss] = new

    # ------------------------------------------------------------------ runners on the plate appearance
    def br_inputs(self, a, bt, ft, lead_base):
        A, g = self.A, self.gl[a]
        n = len(a)
        occ = self.occ[a] >= 0
        spot = self.slot[a, bt]
        sub = self.repl[a, bt, spot]
        tend = np.where(sub[:, None], np.array([0.43, 0.44, 300.0, 0.234, 0.379]), A["tend"][g, bt, spot])
        d = {"outs_pre": self.outs[a].astype(float), "base1_pre": occ[:, 0].astype(int), "base2_pre": occ[:, 1].astype(int),
             "base3_pre": occ[:, 2].astype(int), "inning": self.inning[a].astype(float),
             "margin": (self.score[a, bt] - self.score[a, ft]).astype(float),
             "bat_speed": np.where(sub, 27.0, A["br_speed"][g, bt, spot]),
             "bat_gb": tend[:, 0], "bat_gb_middle": tend[:, 1], "bat_fly_dist": tend[:, 2], "bat_to_rf": tend[:, 3],
             "bat_to_lf": tend[:, 4], "fld_dp": A["fld_dp"][g, ft], "team_xbt": np.ones(n), "park_xbt": np.ones(n),
             "arm_toward": np.full(n, 87.5), "arm_mean": np.full(n, 87.5)}
        for b, nm in ((0, "r1"), (1, "r2"), (2, "r3")):
            _, spd, miss, xbt = self.runner_attr(a, bt, np.full(n, b))
            d[f"{nm}_speed"], d[f"{nm}_speed_missing"], d[f"{nm}_xbt"] = spd, miss, xbt
        return d

    def sit_prob(self, a, bt, ft, sit, m, j):
        """chance of outcome j of a step 11 situation for lanes a[m] (0 elsewhere)"""
        out = np.zeros(len(a))
        if m.any():
            aa = a[m]
            d = self.br_inputs(aa, bt[m], ft[m], None)
            k = list(BR.SITUATIONS).index(sit)
            K = len(BR.SITUATIONS[sit][1])
            off = self.A["br_off"][self.gl[aa], k, :K]
            out[m] = BR.predict(d, sit, self.M.br_W[sit], self.M.br_fams, off)[:, j]
        return out

    def rbi_pa_dist(self, a, bt, ft, p):
        """(n, 5) chance of 0..4 RBI on the plate appearance"""
        n = len(a)
        occ = self.occ[a] >= 0
        b1, b2, b3 = occ[:, 0], occ[:, 1], occ[:, 2]
        lo2 = self.outs[a] < 2
        nr = b1.astype(int) + b2 + b3

        def det(k):
            x = np.zeros((n, 5))
            x[np.arange(n), np.minimum(k, 4)] = 1
            return x

        def badd(x, q):
            y = x * (1 - q)[:, None]
            y[:, 1:] += x[:, :-1] * q[:, None]
            y[:, -1] += x[:, -1] * q
            return y

        D = det(np.zeros(n, int)) * (p[:, SO_] + p[:, ROE_])[:, None]
        D += det((b1 & b2 & b3).astype(int)) * (p[:, BB_] + p[:, HBP_])[:, None]
        D += det(nr + 1) * p[:, [HR_]] + det(nr) * p[:, [S3]]
        if not occ.any():
            return D + det(np.zeros(n, int)) * (p[:, S1] + p[:, S2] + p[:, AO_] + p[:, GO_])[:, None]
        p11 = self.sit_prob(a, bt, ft, "1B from 1st", b1, 2)
        p12 = self.sit_prob(a, bt, ft, "1B from 2nd", b2, 1)
        p21 = self.sit_prob(a, bt, ft, "2B from 1st", b1, 1)
        pao3 = self.sit_prob(a, bt, ft, "AO from 3rd", b3 & lo2, 1)
        pdp = self.sit_prob(a, bt, ft, "GO with 1st", b1 & lo2, 1)
        pgo3 = self.sit_prob(a, bt, ft, "GO from 3rd", b3 & lo2, 1) * np.where(b1, 1 - pdp, 1)
        D += badd(det((b2.astype(int) + b3)), p21 * b1) * p[:, [S2]]
        D += badd(badd(det(b3.astype(int)), p12 * b2), p11 * b1) * p[:, [S1]]
        D += badd(det(np.zeros(n, int)), pao3 * b3) * p[:, [AO_]]
        D += badd(det(np.zeros(n, int)), pgo3 * b3) * p[:, [GO_]]
        return D

    def br_draw(self, a, bt, ft, sit, m):
        """draw the step 11 situation outcome for lanes a[m]"""
        out = np.full(len(a), -1)
        if not m.any():
            return out
        aa = a[m]
        d = self.br_inputs(aa, bt[m], ft[m], None)
        j = list(BR.SITUATIONS).index(sit)
        K = len(BR.SITUATIONS[sit][1])
        off = self.A["br_off"][self.gl[aa], j, :K]
        P = BR.predict(d, sit, self.M.br_W[sit], self.M.br_fams, off)
        out[m] = _draw(self.rng, P)
        return out

    def resolve(self, a, bt, ft, o, spot, sub):
        n = len(a)
        rng = self.rng
        occ = self.occ[a] >= 0
        b1, b2, b3 = occ[:, 0], occ[:, 1], occ[:, 2]
        outs = self.outs[a]
        stay = np.where(occ, np.arange(1, 4)[None, :], -1)
        dest = stay.copy()
        db = np.full(n, -1)
        rbi = np.ones(n, bool)
        earned_play = np.ones(n, bool)
        batter_earned = np.ones(n, bool)
        runs_count = np.ones(n, bool)
        u = rng.random((n, 4))
        # strikeouts (dropped third strike: batter reaches, forced runners move up)
        so = o == SO_
        dts = so & (~b1 | (outs == 2))
        if dts.any():
            j = self.M.ev_names.index("Dropped third strike, batter safe")
            spec = self.M.ev["Dropped third strike, batter safe"]
            dd = {"outs_pre": outs[dts].astype(float)}
            X, _ = EV.design(dd, "Dropped third strike, batter safe", self.A["ev_lr"][self.gl[a[dts]], j], {}, [])
            p = expit(EX.logit(spec["w"], X))
            safe = np.zeros(n, bool)
            safe[dts] = u[dts, 0] < p
        else:
            safe = np.zeros(n, bool)
        db = np.where(so, 0, db)
        walk = (o == BB_) | (o == HBP_) | safe
        if walk.any():
            f1 = walk & b1
            f2 = f1 & b2
            dest[:, 2] = np.where(walk & b3, np.where(f2, 4, 3), dest[:, 2])
            dest[:, 1] = np.where(walk & b2, np.where(f1, 3, 2), dest[:, 1])
            dest[:, 0] = np.where(walk & b1, 2, dest[:, 0])
            db = np.where(walk, 1, db)
            rbi &= ~safe
        # home runs and triples
        hr, t3 = o == HR_, o == S3
        dest = np.where((hr | t3)[:, None] & occ, 4, dest)
        db = np.where(hr, 4, np.where(t3, 3, db))
        # doubles: runners on 2nd and 3rd score; runner on 1st by the step 11 model
        d2 = o == S2
        dest[:, 1] = np.where(d2 & b2, 4, dest[:, 1])
        dest[:, 2] = np.where(d2 & b3, 4, dest[:, 2])
        y = self.br_draw(a, bt, ft, "2B from 1st", d2 & b1)                   # to 3rd / scores / out
        dest[:, 0] = np.where(d2 & b1, np.choose(np.maximum(y, 0), [3, 4, 0]), dest[:, 0])
        db = np.where(d2, 2, db)
        # singles: runner on 3rd scores; 2nd and 1st by the step 11 models
        s1 = o == S1
        dest[:, 2] = np.where(s1 & b3, 4, dest[:, 2])
        y2 = self.br_draw(a, bt, ft, "1B from 2nd", s1 & b2)                  # held / scores / out
        held = np.where((u[:, 1] < HELD_AT_3RD) | b1, 3, 2)
        dest[:, 1] = np.where(s1 & b2, np.choose(np.maximum(y2, 0), [0, 4, 0]) + np.where(y2 == 0, held, 0), dest[:, 1])
        y1 = self.br_draw(a, bt, ft, "1B from 1st", s1 & b1)                  # to 2nd / to 3rd / scores / out
        r1d = np.choose(np.maximum(y1, 0), [2, 3, 4, 0])
        blocked = (dest[:, 1] == 3) & (r1d >= 3)
        dest[:, 0] = np.where(s1 & b1, np.where(blocked, 2, r1d), dest[:, 0])
        db = np.where(s1, 1, db)
        # the batter takes an extra base when it's free
        taken2 = (dest == 2).any(1)
        taken3 = (dest == 3).any(1)
        e1, e2 = BATTER_EXTRA["1B"]
        db = np.where(s1 & ~taken2 & (u[:, 3] < e1), 2, db)
        db = np.where(s1 & ~taken2 & ~taken3 & (u[:, 3] >= e1) & (u[:, 3] < e1 + e2), 3, db)
        db = np.where(d2 & ~taken3 & (u[:, 2] < BATTER_EXTRA["2B"]), 3, db)
        # reached on error: league shares; runs on the play unearned, no RBI
        roe = o == ROE_
        if roe.any():
            M = self.M
            for b in range(3):
                m = roe & occ[:, b]
                if m.any():
                    dest[m, b] = np.maximum(_draw(rng, np.repeat(M.roe_run[b][None, :], m.sum(), 0)), 0)
                    dest[m, b] = np.where(dest[m, b] == 0, 0, np.maximum(dest[m, b], b + 1))
            pb = M.roe_bat[1:] / M.roe_bat[1:].sum()
            db = np.where(roe, 1 + _draw(rng, np.repeat(pb[None, :], n, 0)), db)
            rbi &= ~roe
            earned_play &= ~roe
            batter_earned &= ~roe
            self.err_outs[a[roe]] += 1
        # ground outs
        go = o == GO_
        g2 = go & (outs == 2)
        db = np.where(g2, 0, db)
        g1 = go & (outs < 2) & b1
        if g1.any():
            y = self.br_draw(a, bt, ft, "GO with 1st", g1)                    # batter out / double play / runner forced
            yy = np.maximum(y, 0)
            loaded = b1 & b2 & b3
            db = np.where(g1, np.where(yy == 0, np.where(u[:, 2] < GO1_BATTER_OUT, 0, 1), np.where(yy == 1, 0, 1)), db)
            dest[:, 0] = np.where(g1, np.where(yy == 0, 2, 0), dest[:, 0])
            for c in (0, 1, 2):
                m = g1 & (yy == c)
                if not m.any():
                    continue
                m2 = m & b2
                if m2.any():
                    dest[m2, 1] = _draw(rng, np.repeat(np.array(GO1_R2[c])[None, :], m2.sum(), 0))
                m3 = m & b3
                if m3.any():
                    tab = np.where(loaded[m3][:, None], np.array(GO1_R3_LOADED[c])[None, :], np.array(GO1_R3[c])[None, :])
                    dest[m3, 2] = _draw(rng, tab)
            rbi &= ~(g1 & (yy == 1))
            tot_outs = outs + (dest[:, 0] == 0) + (db == 0) + (dest[:, 1] == 0) + (dest[:, 2] == 0)
            runs_count &= ~(g1 & (tot_outs >= 3))
        g0 = go & (outs < 2) & ~b1
        if g0.any():
            y3 = self.br_draw(a, bt, ft, "GO from 3rd", g0 & b3)                # holds / scores / out
            dest[:, 2] = np.where(g0 & b3, np.choose(np.maximum(y3, 0), [3, 4, 0]), dest[:, 2])
            y2 = self.br_draw(a, bt, ft, "GO from 2nd", g0 & b2 & ~b3)          # holds / to 3rd / out
            dest[:, 1] = np.where(g0 & b2 & ~b3, np.choose(np.maximum(y2, 0), [2, 3, 0]), dest[:, 1])
            free3 = g0 & b2 & b3 & (dest[:, 2] != 3)
            dest[:, 1] = np.where(free3, np.where(u[:, 3] < R2_TO_3RD_WHEN_FREE, 3, 2), dest[:, 1])
            fc = g0 & ((b3 & (dest[:, 2] == 0)) | (b2 & ~b3 & (dest[:, 1] == 0)))
            db = np.where(g0, np.where(fc & (u[:, 2] < GO_FC_BATTER_SAFE), 1, 0), db)
            tot_outs = outs + (db == 0) + (dest[:, 1] == 0) + (dest[:, 2] == 0)
            runs_count &= ~(g0 & (tot_outs >= 3) & (db == 0))
        # air outs
        ao = o == AO_
        db = np.where(ao, 0, db)
        a1 = ao & (outs < 2)
        if a1.any():
            y3 = self.br_draw(a, bt, ft, "AO from 3rd", a1 & b3)                # holds / scores / out
            dest[:, 2] = np.where(a1 & b3, np.choose(np.maximum(y3, 0), [3, 4, 0]), dest[:, 2])
            open3 = a1 & b2 & (~b3 | (dest[:, 2] != 3))
            y2 = self.br_draw(a, bt, ft, "AO from 2nd", open3)                  # holds / to 3rd / out
            dest[:, 1] = np.where(open3, np.choose(np.maximum(y2, 0), [2, 3, 0]), dest[:, 1])
            r1m = a1 & b1
            if r1m.any():
                d1 = _draw(rng, np.repeat(np.array(AO_R1)[None, :], r1m.sum(), 0))
                dest[r1m, 0] = np.choose(d1, [0, 1, 2, 3, 4])
        bsp = spot.astype(np.int8)
        self.move(a, bt, ft, dest, db, bsp, sub, batter_earned, earned_play, rbi, runs_count)
        # sacrifice-fly style RBI on air outs and ground outs are in move(); nothing else here

    # ------------------------------------------------------------------ inning and game flow
    def end_half(self, e):
        bt, ft = self.team(e)
        sched = self.sched[e]
        inn = self.inning[e]
        home_ahead = self.score[e, 1] > self.score[e, 0]
        over = ((bt == 0) & (inn >= sched) & home_ahead) | ((bt == 1) & (inn >= sched) & (self.score[e, 1] != self.score[e, 0]))
        over |= inn >= 20
        self.alive[e[over]] = False
        c = e[~over]
        if not len(c):
            return
        top_next = self.half[c] == 1
        self.inning[c] += top_next.astype(np.int16)
        self.half[c] = 1 - self.half[c]
        self.outs[c] = 0
        self.occ[c] = -1
        self.runs_inn[c] = 0
        self.err_outs[c] = 0
        nb, nf = self.team(c)
        self.start_inn[c, nf] = True
        self.sp["pc_inn"][c, nf] = 0
        # extra innings: a runner on 2nd (the batter before this inning's leadoff man), unearned
        ex = (self.inning[c] > self.sched[c]) & self.ghost[c]
        if ex.any():
            cc = c[ex]
            b_ = nb[ex]
            prev = (self.slot[cc, b_] - 1) % 9
            self.occ[cc, 1] = prev
            self.osub[cc, 1] = self.repl[cc, b_, prev]
            self.oresp[cc, 1] = self.cur[cc, nf[ex]]
            self.oearn[cc, 1] = False

    def walkoff(self, a):
        bt, ft = self.team(a)
        w = (bt == 1) & (self.inning[a] >= self.sched[a]) & (self.score[a, 1] > self.score[a, 0])
        self.alive[a[w]] = False

    def step(self):
        a = np.where(self.alive)[0]
        if not len(a):
            return False
        self.pitching(a)
        self.substitution(a)
        self.steal(a)
        a2 = a[self.outs[a] < 3]
        self.events(a2)
        a3 = a2[self.outs[a2] < 3]
        self.walkoff(a2)
        a3 = a3[self.alive[a3]]
        if len(a3):
            self.pa(a3)
            self.walkoff(a3)
        e = a[self.alive[a] & (self.outs[a] >= 3)]
        if len(e):
            self.end_half(e)
        return True

    def run(self, max_steps=400):
        for _ in range(max_steps):
            if not self.step():
                break
        return self.results()

    def results(self):
        S, G = self.S, self.G
        win = np.zeros((self.L, 2), bool)
        win[:, 1] = self.score[:, 1] > self.score[:, 0]
        win[:, 0] = self.score[:, 0] > self.score[:, 1]
        sp_outs = self.pit[:, :, 0, PI["OUTS"]]
        min_outs = np.where(self.sched == 7, 12, 15)[:, None]
        sp_w = win & (self.por == 0) & (sp_outs >= min_outs)
        cg = ~self.sp_out & (self.napp == 0)
        res = {"bat": self.bat.reshape(G, S, 2, 9, -1), "pit": self.pit.reshape(G, S, 2, 2, -1),
               "runs": self.score.reshape(G, S, 2), "f5": self.inn_runs[:, :, :5].sum(-1).reshape(G, S, 2),
               "inn1": self.inn_runs[:, :, 0].reshape(G, S, 2), "inn_runs": self.inn_runs.reshape(G, S, 2, 10),
               "win": win.reshape(G, S, 2), "sp_w": sp_w.reshape(G, S, 2), "cg": cg.reshape(G, S, 2),
               "innings": self.inning.reshape(G, S), "unfinished": self.alive.reshape(G, S),
               "bf_k": self.bf_k.reshape(G, S, 2, -1),
               "bat_dist": {nm: v.reshape(G, S, 2, 9, -1).mean(1) for nm, v in self.cd.items()} if self.dists else {}}
        return res


def load_inputs(path="out/step13/inputs.npz"):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def take(A, gsel):
    return {k: v[gsel] for k, v in A.items()}


def simulate(A, gsel, n_sims, M, rng, chunk_lanes=150_000, **kw):
    """simulate games gsel (indices into A) n_sims times each, in chunks; -> list of per-chunk results"""
    gsel = np.asarray(gsel)
    per = max(1, chunk_lanes // n_sims)
    out = []
    for c0 in range(0, len(gsel), per):
        idx = gsel[c0:c0 + per]
        g = Game(take(A, idx), M, n_sims, rng, **kw)
        out.append((idx, g.run()))
    return out
