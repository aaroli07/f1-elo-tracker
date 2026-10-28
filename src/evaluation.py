"""Leakage-safe, race-by-race evaluation of the Elo ratings."""

from itertools import combinations

import numpy as np
import pandas as pd

from src.pipeline import INITIAL_RATING, build_ratings


def _race_starters(results):
    starters = results.copy()
    starters["_laps"] = pd.to_numeric(starters["laps"], errors="coerce")
    starters["_position"] = pd.to_numeric(
        starters["positionOrder"], errors="coerce"
    )
    starters = starters.sort_values(
        ["raceId", "driverId", "_laps", "_position"],
        ascending=[True, True, False, True],
        na_position="last",
        kind="stable",
    ).drop_duplicates(["raceId", "driverId"], keep="first")
    return starters.loc[starters["_laps"].ge(1) & starters["_position"].notna()].copy()


def evaluate_ratings(data_drivers, data_races, data_results):
    """Evaluate pre-race Elo rankings against race results and grid order.

    Each race is predicted using the rating snapshot after the previous race.
    The current race's result is used only for evaluation and subsequent ratings.

    Returns a pair of (per-driver predictions, aggregate metrics).
    """
    history, driver_index, race_ids = build_ratings(
        data_drivers, data_races, data_results
    )
    race_order = {race_id: index for index, race_id in enumerate(race_ids)}
    starters = _race_starters(data_results)
    starters["_grid"] = pd.to_numeric(starters["grid"], errors="coerce")
    starters = starters.loc[starters["raceId"].isin(race_order)].copy()

    prediction_rows = []
    teammate_correct = 0.0
    teammate_pairs = 0
    race_spearman = []

    for race_id, race in starters.groupby("raceId", sort=False):
        index = race_order[race_id]
        prior = history[:, index - 1] if index > 0 else np.full(history.shape[0], np.nan)
        ratings = [
            float(prior[driver_index[driver_id]])
            if np.isfinite(prior[driver_index[driver_id]])
            else INITIAL_RATING
            for driver_id in race["driverId"]
        ]
        race["_elo_before"] = ratings
        race["_elo_rank"] = race["_elo_before"].rank(
            ascending=False, method="average"
        )

        valid_grid = race["_grid"].where(race["_grid"].gt(0))
        missing_grid_rank = (valid_grid.max() if valid_grid.notna().any() else 0) + 1
        race["_grid_rank"] = valid_grid.fillna(missing_grid_rank).rank(
            ascending=True, method="average"
        )

        actual = race["_position"]
        # Spearman is Pearson correlation over ranks; this avoids requiring
        # SciPy just to calculate the evaluation metric.
        correlation = race["_elo_rank"].corr(actual.rank(method="average"))
        if pd.notna(correlation):
            race_spearman.append(float(correlation))

        team_key = race["constructorId"].replace(r"\N", pd.NA)
        for _, team in race.loc[team_key.notna()].assign(_team=team_key).groupby(
            "_team", sort=False
        ):
            rows = list(team[["_elo_before", "_position"]].itertuples(index=False, name=None))
            for (rating_a, place_a), (rating_b, place_b) in combinations(rows, 2):
                predicted = (
                    1.0
                    if rating_a > rating_b
                    else 0.0
                    if rating_a < rating_b
                    else 0.5
                )
                actual_outcome = (
                    1.0
                    if place_a < place_b
                    else 0.0
                    if place_a > place_b
                    else 0.5
                )
                teammate_correct += (
                    1.0 if predicted == actual_outcome else 0.5 if predicted == 0.5 else 0.0
                )
                teammate_pairs += 1

        for (
            driver_id,
            elo_before,
            elo_rank,
            grid_position,
            grid_rank,
            actual_position,
        ) in race[
            [
                "driverId",
                "_elo_before",
                "_elo_rank",
                "_grid",
                "_grid_rank",
                "_position",
            ]
        ].itertuples(index=False, name=None):
            prediction_rows.append(
                {
                    "raceId": race_id,
                    "driverId": driver_id,
                    "ratingBeforeRace": elo_before,
                    "predictedPosition": elo_rank,
                    "gridPosition": grid_position,
                    "gridPredictedPosition": grid_rank,
                    "actualPosition": actual_position,
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        summary = {
            "races": 0,
            "driver_predictions": 0,
            "elo_position_mae": np.nan,
            "grid_position_mae": np.nan,
            "mean_race_spearman": np.nan,
            "teammate_pairwise_accuracy": np.nan,
            "teammate_pairs": 0,
        }
    else:
        summary = {
            "races": int(predictions["raceId"].nunique()),
            "driver_predictions": len(predictions),
            "elo_position_mae": float(
                (predictions["predictedPosition"] - predictions["actualPosition"])
                .abs()
                .mean()
            ),
            "grid_position_mae": float(
                (predictions["gridPredictedPosition"] - predictions["actualPosition"])
                .abs()
                .mean()
            ),
            "mean_race_spearman": (
                float(np.mean(race_spearman)) if race_spearman else np.nan
            ),
            "teammate_pairwise_accuracy": (
                teammate_correct / teammate_pairs if teammate_pairs else np.nan
            ),
            "teammate_pairs": teammate_pairs,
        }
    return predictions, summary


def main():
    from src.data_load import load_all

    drivers, races, results = load_all()
    _, summary = evaluate_ratings(drivers, races, results)
    print("Walk-forward Elo backtest (ratings known before each race)")
    for metric, value in summary.items():
        if isinstance(value, float) and np.isfinite(value):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")


if __name__ == "__main__":
    main()
