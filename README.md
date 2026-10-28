# F1 Elo Tracker

A Streamlit viewer for comparing Formula 1 driver rating histories from the
repository's static CSV data.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app lets you search by driver name or reference, select up to ten drivers,
view each driver's latest rating, and compare rating histories through a
selected race.

## Rating model

The rating update preserves the project's existing teammate finishing-margin
formula as 80% of each update. A centered, field-wide finishing-position
adjustment contributes the remaining 20%. Ratings start at 1000 on each
driver's first observed start. Race results use `positionOrder`; entries with
zero completed laps are treated as non-starters for rating updates.

The app reads `data/DriversMOD.csv`, `data/RacesMOD.csv`, and
`data/results.csv`. The loader resolves these paths from the repository,
independent of the directory from which the app is launched.

## Refreshing race data

Race results can be refreshed from the community-maintained
[Jolpica-F1 Ergast-compatible API](https://github.com/jolpica/jolpica-f1). The
sync command fetches recent seasons, including a previous season for result
corrections, merges by season/round and driver reference, and preserves
existing driver and race IDs. It does not run when the app opens.

Preview a sync without changing the CSV files:

```bash
python -m src.data_sync --dry-run
```

Apply the fetched results:

```bash
python -m src.data_sync
```

The command is suitable for a weekly scheduler, but the scheduler itself is
not configured. It must run somewhere that can reach the API and write to the
repository's `data/` directory.

## Model evaluation

Run a walk-forward backtest:

```bash
python -m src.evaluation
```

For each race, the backtest ranks drivers using ratings from before that race,
then compares the predicted order with the finish. It reports Elo position
error, a starting-grid baseline, per-race rank correlation, and teammate
head-to-head accuracy. The current race's result is never used to make its own
prediction.

## Tests

```bash
python -m unittest discover -s tests -v
```
