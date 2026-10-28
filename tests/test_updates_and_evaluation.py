import unittest
from urllib.parse import parse_qs, urlparse

import pandas as pd
from pandas.testing import assert_frame_equal

from src.data_sync import fetch_season_results, merge_recent_results
from src.evaluation import evaluate_ratings


def api_result(driver_ref, position, constructor_ref="team"):
    return {
        "Driver": {
            "driverId": driver_ref,
            "givenName": driver_ref.title(),
            "familyName": "Driver",
            "nationality": "Test",
        },
        "Constructor": {"constructorId": constructor_ref},
        "position": str(position),
        "positionText": str(position),
        "grid": str(position),
        "laps": "10",
        "points": "0",
    }


class DataSyncTests(unittest.TestCase):
    def test_pagination_combines_a_race_split_across_result_pages(self):
        pages = {
            0: {
                "MRData": {
                    "total": "3",
                    "limit": "2",
                    "RaceTable": {
                        "Races": [
                            {
                                "season": "2024",
                                "round": "1",
                                "raceName": "Test Grand Prix",
                                "Results": [api_result("one", 1), api_result("two", 2)],
                            }
                        ],
                    },
                }
            },
            2: {
                "MRData": {
                    "total": "3",
                    "limit": "2",
                    "RaceTable": {
                        "Races": [
                            {
                                "season": "2024",
                                "round": "1",
                                "raceName": "Test Grand Prix",
                                "Results": [api_result("three", 3)],
                            }
                        ],
                    },
                }
            },
        }
        offsets = []

        def fake_fetch(url):
            offset = int(parse_qs(urlparse(url).query)["offset"][0])
            offsets.append(offset)
            return pages[offset]

        races = fetch_season_results(2024, fetch_json=fake_fetch)

        self.assertEqual(offsets, [0, 2])
        self.assertEqual(len(races), 1)
        self.assertEqual(
            [result["Driver"]["driverId"] for result in races[0]["Results"]],
            ["one", "two", "three"],
        )

    def test_merge_is_idempotent_and_adds_new_drivers_and_races(self):
        drivers = pd.DataFrame(
            [{"driverId": 1, "driverRef": "one", "forename": "One", "surname": "Driver"}]
        )
        races = pd.DataFrame(
            [
                {
                    "raceId": 10,
                    "year": 2024,
                    "round": 1,
                    "date": "2024-03-01",
                    "name": "Old Name",
                    "circuitId": 1,
                }
            ]
        )
        result_columns = [
            "resultId", "raceId", "driverId", "constructorId", "number", "grid",
            "position", "positionText", "positionOrder", "points", "laps", "time",
            "milliseconds", "fastestLap", "rank", "fastestLapTime",
            "fastestLapSpeed", "statusId",
        ]
        results = pd.DataFrame(
            [[1, 10, 1, 7, "1", 1, "1", "1", 1, 0, 10, r"\N", r"\N", r"\N", r"\N", r"\N", r"\N", r"\N"]],
            columns=result_columns,
        )
        api_races = [
            {
                "season": "2024",
                "round": "1",
                "raceName": "Corrected Name",
                "date": "2024-03-02",
                "Circuit": {"circuitId": "test"},
                "Results": [api_result("one", 1), api_result("two", 2)],
            },
            {
                "season": "2024",
                "round": "2",
                "raceName": "New Grand Prix",
                "date": "2024-03-15",
                "Circuit": {"circuitId": "new"},
                "Results": [api_result("two", 1), api_result("one", 2)],
            },
        ]

        first = merge_recent_results(drivers, races, results, api_races)
        second = merge_recent_results(*first, api_races)

        for first_frame, second_frame in zip(first, second):
            assert_frame_equal(first_frame, second_frame)
        self.assertEqual(len(first[0]), 2)
        self.assertEqual(len(first[1]), 2)
        self.assertEqual(len(first[2]), 4)
        self.assertEqual(first[1].loc[first[1]["round"] == 1, "name"].iloc[0], "Corrected Name")


class EvaluationTests(unittest.TestCase):
    def test_predictions_for_a_race_use_only_prior_race_ratings(self):
        drivers = pd.DataFrame({"driverId": [1, 2]})
        races = pd.DataFrame(
            {"raceId": [1, 2], "date": ["2024-01-01", "2024-01-08"]}
        )
        results = pd.DataFrame(
            [
                {"raceId": 1, "driverId": 1, "constructorId": 7, "positionOrder": 1, "laps": 10, "grid": 1},
                {"raceId": 1, "driverId": 2, "constructorId": 7, "positionOrder": 2, "laps": 10, "grid": 2},
                {"raceId": 2, "driverId": 1, "constructorId": 7, "positionOrder": 2, "laps": 10, "grid": 1},
                {"raceId": 2, "driverId": 2, "constructorId": 7, "positionOrder": 1, "laps": 10, "grid": 2},
            ]
        )

        predictions, summary = evaluate_ratings(drivers, races, results)
        race_two = predictions.loc[predictions["raceId"] == 2].set_index("driverId")

        self.assertEqual(race_two.loc[1, "predictedPosition"], 1.0)
        self.assertEqual(race_two.loc[1, "actualPosition"], 2)
        self.assertEqual(race_two.loc[1, "ratingBeforeRace"], 1014.0)
        self.assertEqual(summary["races"], 2)
        self.assertEqual(summary["teammate_pairs"], 2)


if __name__ == "__main__":
    unittest.main()
