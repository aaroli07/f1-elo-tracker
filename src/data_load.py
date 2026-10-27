from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_all(data_dir=DATA_DIR):
    """Load the driver, race, and result tables from the repository data folder."""
    data_dir = Path(data_dir)
    drivers = pd.read_csv(data_dir / "DriversMOD.csv")
    races = pd.read_csv(data_dir / "RacesMOD.csv")
    results = pd.read_csv(data_dir / "results.csv")
    return drivers, races, results
