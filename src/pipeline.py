from itertools import combinations

import numpy as np
import pandas as pd

from src.elo import (
    OVERALL_POSITION_SCALE,
    OVERALL_POSITION_WEIGHT,
    TEAMMATE_WEIGHT,
    calculate_new_elo,
)


INITIAL_RATING = 1000.0
STARTING_LAPS_MINIMUM = 1


def _numeric_column(frame, column):
    return pd.to_numeric(frame[column], errors="coerce")


def build_ratings(data_drivers, data_races, data_results):
    """Build race-by-race ratings from result rows.

    A driver starts at ``INITIAL_RATING`` on their first race with at least one
    completed lap. Each race blends the existing teammate margin update (80%)
    with a centered, field-wide finishing-position adjustment (20%).

    Returns a driver-by-race rating matrix, a driverId-to-row mapping, and race
    IDs ordered by race date. Ratings before a driver's first start are NaN.
    """
    del data_drivers  # Driver rows are not needed; IDs come from race results.

    required_result_columns = {
        "raceId", "driverId", "constructorId", "positionOrder", "laps"
    }
    missing_result_columns = required_result_columns - set(data_results.columns)
    if missing_result_columns:
        raise ValueError(
            "Results data is missing required columns: "
            + ", ".join(sorted(missing_result_columns))
        )
    if not {"raceId", "date"}.issubset(data_races.columns):
        raise ValueError("Race data must contain raceId and date columns")
    if data_races["raceId"].duplicated().any():
        raise ValueError("Race data contains duplicate raceId values")
    if data_results[["raceId", "driverId"]].isna().any().any():
        raise ValueError("Results data contains missing raceId or driverId values")

    races = data_races[["raceId", "date"]].copy()
    races["date"] = pd.to_datetime(races["date"], errors="coerce")
    if races["date"].isna().any():
        raise ValueError("Race data contains missing or invalid dates")

    merged = data_results.merge(
        races,
        how="left",
        on="raceId",
        validate="many_to_one",
        indicator=True,
    )
    unmatched_races = merged.loc[merged["_merge"] != "both", "raceId"].unique()
    if len(unmatched_races):
        raise ValueError(
            "Results reference raceId values missing from race data: "
            + ", ".join(map(str, unmatched_races[:5]))
        )
    merged = merged.drop(columns="_merge").sort_values(
        ["date", "raceId"], kind="stable"
    )
    merged["_laps"] = _numeric_column(merged, "laps")
    merged["_position"] = _numeric_column(merged, "positionOrder")
    # Some historic races list one driver under multiple entries. Keep the
    # entry in which they completed the most laps; use numeric classification
    # as a stable tie-breaker before treating the driver as a single entrant.
    merged = merged.sort_values(
        ["raceId", "driverId", "_laps", "_position"],
        ascending=[True, True, False, True],
        na_position="last",
        kind="stable",
    ).drop_duplicates(["raceId", "driverId"], keep="first")
    merged = merged.sort_values(["date", "raceId"], kind="stable")
    sorted_race_ids = merged["raceId"].drop_duplicates().tolist()

    driver_ids = np.sort(merged["driverId"].unique())
    driver_index_map = {driver_id: index for index, driver_id in enumerate(driver_ids)}
    ratings = np.full((len(driver_ids), len(sorted_race_ids)), np.nan, dtype=float)

    for race_index, race_id in enumerate(sorted_race_ids):
        race = merged.loc[merged["raceId"] == race_id].copy()
        if race_index:
            ratings[:, race_index] = ratings[:, race_index - 1]

        race["_constructor"] = race["constructorId"].replace(r"\N", pd.NA)
        starters = race.loc[
            race["_laps"].ge(STARTING_LAPS_MINIMUM)
            & race["_position"].notna()
        ].copy()
        if starters.empty:
            continue

        starter_ids = starters["driverId"].unique()
        current = {}
        for driver_id in starter_ids:
            row_index = driver_index_map[driver_id]
            value = ratings[row_index, race_index]
            if np.isnan(value):
                value = INITIAL_RATING
                ratings[row_index, race_index] = value
            current[driver_id] = value

        # Compute all teammate updates from pre-race ratings so a team's result
        # does not depend on row or pair iteration order. For historic teams
        # with more than two starters, average each driver's pairwise deltas.
        teammate_delta_totals = {driver_id: 0.0 for driver_id in starter_ids}
        teammate_comparison_counts = {driver_id: 0 for driver_id in starter_ids}
        for _, team in starters.loc[starters["_constructor"].notna()].groupby(
            "_constructor", sort=False
        ):
            team_rows = list(team[["driverId", "_position"]].itertuples(index=False, name=None))
            for (driver_a, place_a), (driver_b, place_b) in combinations(team_rows, 2):
                new_a, new_b = calculate_new_elo(
                    current[driver_a], current[driver_b], int(place_a), int(place_b)
                )
                teammate_delta_totals[driver_a] += new_a - current[driver_a]
                teammate_delta_totals[driver_b] += new_b - current[driver_b]
                teammate_comparison_counts[driver_a] += 1
                teammate_comparison_counts[driver_b] += 1

        # Center the overall result adjustment around this race's mean place,
        # then normalize by field size so years with different grids are
        # comparable. This component sums to zero across the race field.
        positions = starters.set_index("driverId")["_position"]
        field_size = len(positions)
        mean_position = positions.mean()
        position_denominator = max(field_size - 1, 1)

        for driver_id in starter_ids:
            count = teammate_comparison_counts[driver_id]
            teammate_delta = (
                teammate_delta_totals[driver_id] / count if count else 0.0
            )
            place = float(positions.loc[driver_id])
            overall_delta = (
                OVERALL_POSITION_SCALE
                * (mean_position - place)
                / position_denominator
            )
            row_index = driver_index_map[driver_id]
            ratings[row_index, race_index] += (
                TEAMMATE_WEIGHT * teammate_delta
                + OVERALL_POSITION_WEIGHT * overall_delta
            )

    return ratings, driver_index_map, sorted_race_ids
