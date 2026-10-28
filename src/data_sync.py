"""Fetch and merge recent finalized F1 race results from Jolpica-F1."""

import argparse
import json
import os
import tempfile
import time
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from src.data_load import DATA_DIR, load_all
from src.pipeline import build_ratings


API_BASE = "https://api.jolpi.ca/ergast/f1"
API_LIMIT = 100
USER_AGENT = "f1-elo-rankings/0.1"
REQUEST_TIMEOUT = 20


def fetch_api_json(url, timeout=REQUEST_TIMEOUT):
    """Fetch one JSON page with the identifying User-Agent Jolpica requests."""
    request = Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError) as error:
            if attempt == 2:
                raise RuntimeError(f"Jolpica request failed: {url}: {error}") from error
            time.sleep(2**attempt)


def fetch_season_results(season, fetch_json=fetch_api_json):
    """Fetch and combine all result pages for one season.

    Jolpica paginates result rows, not race objects. A race can therefore be
    repeated across pages with only part of its Results list on each page.
    """
    offset = 0
    total = None
    race_by_key = {}

    while total is None or offset < total:
        query = urlencode({"limit": API_LIMIT, "offset": offset})
        url = f"{API_BASE}/{season}/results.json?{query}"
        payload = fetch_json(url)
        mr_data = payload.get("MRData", {})
        race_table = mr_data.get("RaceTable", {})
        total = int(mr_data.get("total", 0))
        page_limit = int(mr_data.get("limit", API_LIMIT))
        page_races = race_table.get("Races", [])

        for race in page_races:
            key = (int(race["season"]), int(race["round"]))
            if key not in race_by_key:
                race_by_key[key] = {**race, "Results": []}
            existing_results = {
                result.get("Driver", {}).get("driverId")
                for result in race_by_key[key]["Results"]
            }
            for result in race.get("Results", []):
                driver_ref = result.get("Driver", {}).get("driverId")
                if driver_ref not in existing_results:
                    race_by_key[key]["Results"].append(result)
                    existing_results.add(driver_ref)

        if total == 0:
            break
        if page_limit <= 0 or not page_races:
            raise ValueError(
                f"Jolpica pagination stopped before completing season {season}"
            )
        offset += page_limit

    return [race_by_key[key] for key in sorted(race_by_key)]


def _driver_row(driver, driver_id, columns):
    row = {column: r"\N" for column in columns}
    row.update(
        {
            "driverId": driver_id,
            "driverRef": driver.get("driverId", r"\N"),
            "number": driver.get("permanentNumber", r"\N"),
            "code": driver.get("code", r"\N"),
            "forename": driver.get("givenName", r"\N"),
            "surname": driver.get("familyName", r"\N"),
            "dob": driver.get("dateOfBirth", r"\N"),
            "nationality": driver.get("nationality", r"\N"),
            "url": driver.get("url", r"\N"),
        }
    )
    if "Unnamed: 0" in row:
        row["Unnamed: 0"] = driver_id - 1
    return row


def _race_values(race):
    circuit = race.get("Circuit", {})
    return {
        "year": int(race["season"]),
        "round": int(race["round"]),
        "circuitId": f"jolpica:{circuit.get('circuitId', 'unknown')}",
        "name": race.get("raceName", r"\N"),
        "date": race.get("date", r"\N"),
        "time": race.get("time", r"\N"),
        "url": race.get("url", r"\N"),
    }


def _result_row(result, race_id, driver_id, result_id, position_order, columns):
    row = {column: r"\N" for column in columns}
    constructor = result.get("Constructor", {})
    race_time = result.get("Time", {}) or {}
    fastest_lap = result.get("FastestLap", {}) or {}
    fastest_lap_time = fastest_lap.get("Time", {}) or {}
    average_speed = fastest_lap.get("AverageSpeed", {}) or {}
    row.update(
        {
            "resultId": result_id,
            "raceId": race_id,
            "driverId": driver_id,
            "constructorId": f"jolpica:{constructor.get('constructorId', 'unknown')}",
            "number": result.get("number", r"\N"),
            "grid": result.get("grid", r"\N"),
            "position": result.get("position", r"\N"),
            "positionText": result.get("positionText", r"\N"),
            "positionOrder": position_order,
            "points": result.get("points", r"\N"),
            "laps": result.get("laps", r"\N"),
            "time": race_time.get("time", r"\N"),
            "milliseconds": race_time.get("millis", r"\N"),
            "fastestLap": fastest_lap.get("lap", r"\N"),
            "rank": fastest_lap.get("rank", r"\N"),
            "fastestLapTime": fastest_lap_time.get("time", r"\N"),
            "fastestLapSpeed": average_speed.get("speed", r"\N"),
        }
    )
    return row


def merge_recent_results(drivers, races, results, api_races):
    """Merge API race results into the repository's existing CSV table shapes."""
    drivers = drivers.copy()
    races = races.copy()
    results = results.copy()

    driver_ids = {
        str(row.driverRef): int(row.driverId)
        for row in drivers.itertuples(index=False)
        if pd.notna(row.driverRef)
    }
    next_driver_id = int(pd.to_numeric(drivers["driverId"]).max()) + 1
    new_driver_rows = []

    # Add new IDs in API chronology, making the mapping stable on later syncs.
    for race in api_races:
        for result in race.get("Results", []):
            driver = result.get("Driver", {})
            driver_ref = driver.get("driverId")
            if driver_ref and driver_ref not in driver_ids:
                driver_ids[driver_ref] = next_driver_id
                new_driver_rows.append(
                    _driver_row(driver, next_driver_id, drivers.columns)
                )
                next_driver_id += 1
    if new_driver_rows:
        drivers = pd.concat([drivers, pd.DataFrame(new_driver_rows)], ignore_index=True)

    race_ids = {
        (int(row.year), int(row.round)): int(row.raceId)
        for row in races.itertuples(index=False)
    }
    if "circuitId" in races.columns:
        races["circuitId"] = races["circuitId"].astype("object")
    next_race_id = int(pd.to_numeric(races["raceId"]).max()) + 1
    new_race_rows = []
    api_race_ids = {}
    for race in sorted(api_races, key=lambda item: (int(item["season"]), int(item["round"]))):
        race_key = (int(race["season"]), int(race["round"]))
        if race_key not in race_ids:
            row = {column: r"\N" for column in races.columns}
            row.update(
                {
                    column: value
                    for column, value in _race_values(race).items()
                    if column in row
                }
            )
            row["raceId"] = next_race_id
            race_ids[race_key] = next_race_id
            new_race_rows.append(row)
            next_race_id += 1
        else:
            row_index = races.index[
                (pd.to_numeric(races["year"]) == race_key[0])
                & (pd.to_numeric(races["round"]) == race_key[1])
            ][0]
            for column, value in _race_values(race).items():
                if column in races.columns:
                    races.at[row_index, column] = value
        if race.get("Results"):
            api_race_ids[race_key] = race_ids[race_key]
    if new_race_rows:
        races = pd.concat([races, pd.DataFrame(new_race_rows)], ignore_index=True)

    race_id_set = set(api_race_ids.values())
    existing_result_ids = {
        (int(row.raceId), int(row.driverId)): int(row.resultId)
        for row in results.loc[results["raceId"].isin(race_id_set)].itertuples(index=False)
    }
    max_result_id = pd.to_numeric(results["resultId"], errors="coerce").max()
    results = results.loc[~results["raceId"].isin(race_id_set)].copy()
    next_result_id = int(max_result_id) + 1 if pd.notna(max_result_id) else 1
    new_result_rows = []

    for race in sorted(api_races, key=lambda item: (int(item["season"]), int(item["round"]))):
        race_key = (int(race["season"]), int(race["round"]))
        race_id = api_race_ids.get(race_key)
        if race_id is None:
            continue
        for position_order, result in enumerate(race.get("Results", []), start=1):
            driver_ref = result.get("Driver", {}).get("driverId")
            if not driver_ref or driver_ref not in driver_ids:
                raise ValueError(f"Jolpica result has an unmapped driver: {driver_ref!r}")
            driver_id = driver_ids[driver_ref]
            result_id = existing_result_ids.get((race_id, driver_id))
            if result_id is None:
                result_id = next_result_id
                next_result_id += 1
            new_result_rows.append(
                _result_row(
                    result,
                    race_id,
                    driver_id,
                    result_id,
                    position_order,
                    results.columns,
                )
            )

    if new_result_rows:
        results = pd.concat([results, pd.DataFrame(new_result_rows)], ignore_index=True)
    results = results.sort_values("resultId", kind="stable").reset_index(drop=True)
    drivers = drivers.sort_values("driverId", kind="stable").reset_index(drop=True)
    races = races.sort_values(["year", "round"], kind="stable").reset_index(drop=True)
    return drivers, races, results


def _write_tables(data_dir, drivers, races, results):
    """Write all tables through temporary files so failed fetches never truncate data."""
    tables = {
        "DriversMOD.csv": drivers,
        "RacesMOD.csv": races,
        "results.csv": results,
    }
    temporary_paths = []
    try:
        for filename, frame in tables.items():
            handle = tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", prefix=f".{filename}.",
                dir=data_dir, delete=False, newline="",
            )
            temporary_path = Path(handle.name)
            temporary_paths.append((temporary_path, data_dir / filename))
            with handle:
                frame.to_csv(handle, index=False)
        for temporary_path, destination in temporary_paths:
            os.replace(temporary_path, destination)
    finally:
        for temporary_path, _ in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def sync_data(data_dir=DATA_DIR, current_year=None, fetch_json=fetch_api_json, write=True):
    """Refresh the latest seasons and append/replace finalized race results."""
    data_dir = Path(data_dir)
    drivers, races, results = load_all(data_dir)
    latest_local_year = int(pd.to_numeric(races["year"]).max())
    current_year = int(current_year or date.today().year)
    first_season = max(1950, latest_local_year - 1)

    api_races = []
    for season in range(first_season, current_year + 1):
        api_races.extend(fetch_season_results(season, fetch_json=fetch_json))
    if not api_races:
        raise ValueError("Jolpica returned no finalized race results to sync")

    updated = merge_recent_results(drivers, races, results, api_races)
    # Run the production pipeline before writing, so incomplete or malformed
    # API data cannot replace a dataset the app can currently load.
    build_ratings(*updated)
    if write:
        _write_tables(data_dir, *updated)

    return {
        "seasons_fetched": (first_season, current_year),
        "races_received": sum(bool(race.get("Results")) for race in api_races),
        "drivers": len(updated[0]),
        "races": len(updated[1]),
        "results": len(updated[2]),
        "written": write,
    }


def main():
    parser = argparse.ArgumentParser(description="Sync F1 results from Jolpica-F1")
    parser.add_argument(
        "--dry-run", action="store_true", help="fetch and validate without writing CSVs"
    )
    args = parser.parse_args()
    summary = sync_data(write=not args.dry_run)
    print(
        f"Fetched seasons {summary['seasons_fetched'][0]}-"
        f"{summary['seasons_fetched'][1]}: {summary['races_received']} races; "
        f"dataset now has {summary['drivers']} drivers, {summary['races']} races, "
        f"and {summary['results']} results."
    )
    if args.dry_run:
        print("Dry run: CSV files were not changed.")


if __name__ == "__main__":
    main()
