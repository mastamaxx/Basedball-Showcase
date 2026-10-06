# Code sample

A few files copied from the private Basedball repo, unchanged. They're here to read, not to run: they import other modules and load fitted weights and data that aren't in this repo.

- `basedball/sim/engine.py`: the game simulator. Both teams, batter by batter, vectorized over games and simulations with NumPy.
- `basedball/model/exit.py`: the starter-exit model, the chance the manager pulls the starter before each batter.
- `basedball/model/mnl.py`: multinomial logistic regression with an offset, the base of the plate-appearance model.
- `basedball/eval/grade.py`: how every model is graded (log loss, Brier score at prop lines, calibration).
- `basedball/eval/actuals.py`: actual results from the play-by-play, for grading.
