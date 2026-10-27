import unittest
from pathlib import Path
import sys

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.data_load import load_all


class StreamlitAppTests(unittest.TestCase):
    def test_app_renders_driver_search_and_multiple_selected_ratings(self):
        app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
        project_root = app_path.parent.parent.resolve()
        original_path = sys.path[:]
        sys.path[:] = [
            entry
            for entry in sys.path
            if Path(entry or Path.cwd()).resolve() != project_root
        ]
        try:
            app = AppTest.from_file(str(app_path)).run(timeout=90)
        finally:
            sys.path[:] = original_path

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.multiselect), 1)
        self.assertEqual(len(app.slider), 1)

        driver_picker = app.multiselect[0]
        self.assertGreaterEqual(len(driver_picker.options), 2)
        driver_picker.set_value(driver_picker.options[:2]).run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 2)

        drivers, races, results = load_all()
        ordered_races = races.assign(
            date=pd.to_datetime(races["date"], errors="coerce")
        ).sort_values(["date", "raceId"], kind="stable")
        race_number_by_id = {
            race_id: index + 1
            for index, race_id in enumerate(ordered_races["raceId"].drop_duplicates())
        }
        completed_laps = pd.to_numeric(results["laps"], errors="coerce")
        starts = results.loc[completed_laps.ge(1), ["driverId", "raceId"]]
        driver_race_numbers = {
            driver_id: sorted(
                {
                    race_number_by_id[race_id]
                    for race_id in group["raceId"]
                    if race_id in race_number_by_id
                }
            )
            for driver_id, group in starts.groupby("driverId", sort=False)
        }
        driver_labels = {
            f"{row.forename} {row.surname} ({row.driverRef})": row.driverId
            for row in drivers.itertuples(index=False)
        }
        eligible_labels = [
            label
            for label in driver_picker.options
            if label in driver_labels
            and driver_labels[label] in driver_race_numbers
            and driver_race_numbers[driver_labels[label]][0]
            < driver_race_numbers[driver_labels[label]][-1]
        ]
        later_driver_label = max(
            eligible_labels,
            key=lambda label: driver_race_numbers[driver_labels[label]][0],
        )
        later_driver_id = driver_labels[later_driver_label]
        expected_start = driver_race_numbers[later_driver_id][0]
        expected_end = driver_race_numbers[later_driver_id][-1]

        app.multiselect[0].set_value([later_driver_label]).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.slider[0].min, expected_start)
        self.assertEqual(app.slider[0].max, expected_end)
        self.assertEqual(app.slider[0].value, expected_end)


if __name__ == "__main__":
    unittest.main()
