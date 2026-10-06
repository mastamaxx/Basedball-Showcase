# How the model is built

Basedball is a set of models wired into one game simulator. Most of them replaced a fixed rule from the 2020 version, and each went in only after it beat that rule (or league averages) on seasons it wasn't trained on. All of them are fitted on 2021–2024 play-by-play; the numbers below are from the 2025 test season unless noted.

## The pieces

| Piece | What it decides | How | What it added |
|---|---|---|---|
| Player rates | Each player's talent level going into the day | Steamer and ZiPS projections averaged, blended with five recency windows (30 days to three years), and batted-ball results pulled toward expected stats from exit velocity and launch angle | +0.18% plate-appearance log loss over the previous rates |
| Plate appearance | The outcome of each PA (strikeout, walk, hit by type, error, ground out, air out) | Multinomial logistic regression on 17 factors: runners, outs, score, inning, home or away, times through the order and pitch count, pitch traits, platoon splits, last 30 days, temperature, wind, starter vs reliever, next batter up and more, plus an in-season league tracker | 0.78% lower log loss than a player-average baseline (0.98% in the first half of 2026) |
| Starter exit | The chance the starter is pulled before each batter | Logistic model over 634,000 decisions: pitch count, trouble this inning, his usual leash, bullpen workload, due-up batters, month, playoff race and more, plus a tracker for the league's drift toward earlier hooks | 41% lower log loss than the old batters-faced target rule |
| Pitches per PA | Pitch counts | Ordered logit given the PA outcome and both players' habits | Per-start error of about 8 pitches, close to pure chance |
| Reliever choice | Who comes in from the bullpen | Conditional logit over the relievers actually available that day: rest, handedness vs the due-up hitters, role | Picks the actual reliever 41% of the time, vs 15% by recent usage |
| Reliever exit | When a reliever is pulled | Logistic per batter, with in-season trackers | 19.5% lower log loss than league pull rates |
| Substitutions | Whether a starter's spot has been pinch-hit or subbed out | Logistic per turn: the pitcher he'd face, his usual removal rate, score, bench strength | 14% lower log loss than league rates; starters' PAs per game 4.03 predicted vs 4.03 actual (4.18 assuming every turn) |
| Steals | Attempt, then success | Two models: runner history and speed, pitcher, catcher pop time, score and outs | Stolen-base props 37.6% lower log loss than the old rule |
| Baserunning | Where runners end up on hits, outs and double plays, in 8 situations | Multinomial logistic: runner speed, the batter's batted-ball tendencies, score and inning, extra-base history | RBI props 1.8% lower log loss than the old fixed rules |
| Other events | Wild pitches, passed balls, balks, pickoffs, defensive indifference | Rates by pitcher and catcher against the league | 2025 wild pitches 1.665% predicted vs 1.668% actual per chance |
| Game-to-game swings | A pitcher's good or bad day, a hot lineup | Random shifts per game, sized on 2021–2024 results | Spread of starters' strikeouts and hits allowed within 1–2% of real games |
| Team terms | In-season defense, schedule fatigue, run conversion | One shift per batting team and game | Moneyline log loss +0.19% (2025) |
| Postseason | Quicker hooks, more walks, fewer hits in play | A starter-exit term and a rates tilt fitted on 2021–2024 postseasons | Starter outs props +1.2% (2021–2024 postseasons) |

## Does it look like real baseball?

Simulated vs actual, 2025:

| | Simulated | Actual |
|---|---|---|
| Runs per team-game | 4.43 | 4.45 |
| Starter outs per start | 15.55 | 15.56 |

League hits, home runs, walks and strikeouts all land within 1% of the real 2025 totals, and starters' batters faced, pitches and strikeouts within 1% too.

Win probabilities are calibrated: in 2025, games the model gave 36.6% went 37.5%, 45.6% went 47.1%, 54.3% went 52.9% and 63.4% went 62.5%.

## The daily run

- **Nightly:** pull the day's play-by-play, box scores, rosters, transactions and Statcast data; rebuild the point-in-time features; apply the saved models.
- **Game day:** check lineups 2 hours, 1 hour, 40 minutes and 20 minutes before the first game, then 40 minutes before each later start. Games with posted lineups are built and simulated; at the 20-minute check the rest run on projected lineups and get rebuilt once theirs post.
- **Outputs:** a workbook with every player's prop probabilities and win chances, game-by-game simulated box scores, prices against DraftKings' lines, and DraftKings classic lineups from a DFS optimizer that works off the simulated games.
- **Speed:** 10,000 simulations of a 15-game slate in about two minutes.
