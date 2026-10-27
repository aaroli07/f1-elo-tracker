import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_load import load_all
from src.elo import calculate_new_elo
from src.pipeline import build_ratings


class EloTests(unittest.TestCase):
    def test_existing_teammate_formula_is_preserved(self):
        self.assertEqual(calculate_new_elo(1000, 1000, 1, 5), (1020, 980))


class PipelineTests(unittest.TestCase):
    def test_orders_races_uses_position_order_and_starts_history_at_debut(self):
        drivers = pd.DataFrame({"driverId": [1, 2, 3]})
        races = pd.DataFrame(
            {"raceId": [2, 1], "date": ["2020-02-01", "2020-01-01"]}
        )
        results = pd.DataFrame(
            [
                # Earlier race: the finishing text is nonnumeric, but the
                # official ordering and completed laps still define the result.
                {"raceId": 1, "driverId": 1, "constructorId": 7, "position": "1", "positionOrder": 1, "laps": 10},
                {"raceId": 1, "driverId": 2, "constructorId": 7, "position": "R", "positionOrder": 2, "laps": 8},
                # Driver 3 debuts in the later race.
                {"raceId": 2, "driverId": 1, "constructorId": 7, "position": "2", "positionOrder": 2, "laps": 10},
                {"raceId": 2, "driverId": 2, "constructorId": 7, "position": "1", "positionOrder": 1, "laps": 10},
                {"raceId": 2, "driverId": 3, "constructorId": 8, "position": "3", "positionOrder": 3, "laps": 10},
            ]
        )

        history, index, race_ids = build_ratings(drivers, races, results)

        self.assertEqual(race_ids, [1, 2])
        self.assertTrue(np.isnan(history[index[3], 0]))
        self.assertEqual(history[index[1], 0], 1014.0)
        self.assertEqual(history[index[2], 0], 986.0)
        self.assertTrue(np.isfinite(history[index[3], 1]))

    def test_nonstarters_do_not_receive_a_rating_or_update(self):
        drivers = pd.DataFrame({"driverId": [1, 2]})
        races = pd.DataFrame({"raceId": [1], "date": ["2020-01-01"]})
        results = pd.DataFrame(
            [
                {"raceId": 1, "driverId": 1, "constructorId": 7, "positionOrder": 1, "laps": 10},
                {"raceId": 1, "driverId": 2, "constructorId": 7, "positionOrder": 2, "laps": 0},
            ]
        )

        history, index, _ = build_ratings(drivers, races, results)

        self.assertEqual(history[index[1], 0], 1000.0)
        self.assertTrue(np.isnan(history[index[2], 0]))

    def test_rejects_results_for_unknown_races(self):
        drivers = pd.DataFrame({"driverId": [1]})
        races = pd.DataFrame({"raceId": [1], "date": ["2020-01-01"]})
        results = pd.DataFrame(
            [{"raceId": 2, "driverId": 1, "constructorId": 7, "positionOrder": 1, "laps": 10}]
        )

        with self.assertRaisesRegex(ValueError, "missing from race data"):
            build_ratings(drivers, races, results)

    def test_uses_single_entry_with_most_laps_for_historic_duplicate_rows(self):
        drivers = pd.DataFrame({"driverId": [1, 2]})
        races = pd.DataFrame({"raceId": [1], "date": ["2020-01-01"]})
        results = pd.DataFrame(
            [
                {"raceId": 1, "driverId": 1, "constructorId": 7, "positionOrder": 1, "laps": 3},
                {"raceId": 1, "driverId": 1, "constructorId": 8, "positionOrder": 2, "laps": 10},
                {"raceId": 1, "driverId": 2, "constructorId": 8, "positionOrder": 4, "laps": 10},
            ]
        )

        history, index, _ = build_ratings(drivers, races, results)

        self.assertEqual(history[index[1], 0], 1028.0)
        self.assertEqual(history[index[2], 0], 972.0)

    def test_load_all_uses_expected_filenames_and_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            (data_dir / "DriversMOD.csv").write_text("driverId,driverRef\n1,one\n")
            (data_dir / "RacesMOD.csv").write_text("raceId,date\n1,2020-01-01\n")
            (data_dir / "results.csv").write_text(
                "raceId,driverId,constructorId,positionOrder,laps\n1,1,7,1,10\n"
            )

            drivers, races, results = load_all(data_dir)

        self.assertEqual(drivers.loc[0, "driverRef"], "one")
        self.assertEqual(races.loc[0, "raceId"], 1)
        self.assertEqual(results.loc[0, "positionOrder"], 1)


if __name__ == "__main__":
    unittest.main()
