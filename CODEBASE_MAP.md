# Codebase map

## Purpose and shape

This repository contains a Python implementation of an Elo-style comparison for Formula 1 drivers, plus a Streamlit viewer. The intended flow is local CSV data → `load_all()` → `build_ratings()` → per-driver rating history → a plotted comparison. There is also a Jupyter notebook that follows the same computation and plots selected drivers.

## Main flow

1. `app/streamlit_app.py` loads driver, race, and result tables, calls `src.pipeline.build_ratings`, and presents selected drivers' rating histories.
2. `src/data_load.py::load_all` reads the checked-in files from a path anchored to the repository, independent of the current working directory.
3. `src/pipeline.py::build_ratings` validates and orders race results, starts a driver at 1000 on their first observed start, carries ratings forward, and compares same-constructor starters.
4. `src.elo.py::calculate_new_elo` preserves the existing teammate finishing-margin update; the pipeline blends 80% of this component with a centered, normalized 20% field-position adjustment.
4. `notebooks/f1_elo.ipynb` uses the same loader, pipeline, and Elo functions, then plots chosen driver IDs with Matplotlib.

## Responsibilities and data

| Path | Role and relationships |
| --- | --- |
| `app/streamlit_app.py` | Cached Streamlit UI; searches full names/references, shows latest ratings for selected drivers, and plots their histories with Matplotlib. |
| `src/data_load.py::load_all` | Reads `DriversMOD.csv`, `RacesMOD.csv`, and `results.csv` from repository `data/` or an optional supplied directory. |
| `src/pipeline.py::build_ratings` | Produces a driver-by-race NumPy rating history, a `driverId`→row index dictionary, and race IDs ordered by date. Uses `positionOrder` and completed laps; resolves duplicate driver entries by most laps; averages pairwise updates for teams with more than two starters. |
| `src/elo.py` | Preserves the teammate update rule (`k=5`, `psf=500`) and defines blend weights and the scale for the overall position component. |
| `notebooks/f1_elo.ipynb` | Exploratory entry point with the same imports and computation, followed by a static driver comparison plot. |
| `data/DriversMOD.csv` | Driver identifiers and display/reference fields, including `driverId` and `driverRef`. |
| `data/RacesMOD.csv` | Race metadata, including `raceId` and `date`, used to sequence results. |
| `data/results.csv` | Result rows, including `raceId`, `driverId`, `constructorId`, and `position`, which the rating pipeline consumes. |
| `requirements.txt` | Runtime dependencies: pandas, NumPy, Streamlit, and Matplotlib. |
| `tests/test_pipeline.py` | Standard-library unittest coverage for the loader and core pipeline behavior. |

The data files are local inputs; no database, API, or other external service is configured. CSV headers were reviewed to confirm the columns used by the source. Individual rows were not exhaustively audited.

## Build, tests, and deployment

The repository has no `pyproject.toml`, setup configuration, CI workflows, or deployment/runtime manifest. `requirements.txt` is the only dependency manifest. The README documents local setup, the Streamlit command, the rating model, and the unittest command.

## Known integration gaps and constraints

- The current CSV set has no status-name lookup. The pipeline treats `laps > 0` as evidence a driver started; this excludes DNS/no-lap entries, but also excludes a driver who started and retired before completing a lap.
- `positionOrder` includes numeric classifications for retirements, so a driver with completed laps remains eligible even when `position` is a nonnumeric marker such as `R`.
- Historic race entries can include more than two same-constructor starters. The pipeline compares every pair and averages each driver's pairwise deltas to avoid scaling updates with team size.
- Some historic result rows list the same driver more than once in one race. The pipeline keeps the row with the most completed laps, then uses the best numeric order as a tie-breaker; this is a deterministic data-cleaning assumption.
- Pre-debut matrix values are `NaN`. Consumers should handle this as “no rating yet,” rather than plotting 1000 before the driver started.
- The overall-position adjustment is centered on each race's mean finishing place, normalized by the observed finishing-place span, and blended at 20%; this deliberately allows some car-performance signal into ratings.

These are observations from the checked-in source and filenames, not a claim that any local deployment or untracked data has the same layout.

## Orientation coverage

Reviewed all maintained Python source files, the README, requirements, notebook code cells, directory inventory, CSV headers, and representative result rows. The results file has 26,519 rows across 1,113 races; it was scanned for status and team-entry patterns to choose conservative pipeline handling, but all results were not manually audited. Generated environments, hidden runtime state, and external deployment settings are outside this repository review.
