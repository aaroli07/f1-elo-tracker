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

The data is static and read from `data/DriversMOD.csv`, `data/RacesMOD.csv`,
and `data/results.csv`. The loader resolves these paths from the repository,
independent of the directory from which the app is launched.

## Tests

```bash
python -m unittest discover -s tests -v
```
