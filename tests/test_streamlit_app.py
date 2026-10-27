import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class StreamlitAppTests(unittest.TestCase):
    def test_app_renders_driver_search_and_multiple_selected_ratings(self):
        app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
        app = AppTest.from_file(str(app_path)).run(timeout=90)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.multiselect), 1)
        self.assertEqual(len(app.slider), 1)

        driver_picker = app.multiselect[0]
        self.assertGreaterEqual(len(driver_picker.options), 2)
        driver_picker.set_value(driver_picker.options[:2]).run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 2)


if __name__ == "__main__":
    unittest.main()
