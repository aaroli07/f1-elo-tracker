import pandas as pd

def load_all():
    drivers = pd.read_csv(f"../data/DriversMOD.csv")
    races = pd.read_csv(f"../data/RacesMOD.csv")
    results = pd.read_csv(f"../data/ResultsMOD.csv")
    return drivers, races, results