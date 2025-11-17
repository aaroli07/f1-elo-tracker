import streamlit as st
import matplotlib.pyplot as plt
from src.data_io import load_all
from src.pipeline import build_ratings

# Load data
data_drivers, data_races, data_results = load_all()
Ratings_History, driver_index_map, sorted_race_ids = build_ratings(
    data_drivers, data_races, data_results
)

st.title("F1 Elo Rating Viewer with Slider Animation")

# Select multiple drivers (comma-separated or one at a time)
drivers_input = st.text_input("Enter driverRef names (comma-separated, max 10):")
driver_names = [d.strip() for d in drivers_input.split(",") if d.strip()][:10]

# Slider for race number
max_races = Ratings_History.shape[1]
race_frame = st.slider("Select Race Number", 1, max_races, 1)

if driver_names:
    fig, ax = plt.subplots()
    any_valid = False

    for name in driver_names:
        try:
            driver_id = data_drivers[data_drivers['driverRef'] == name]['driverId'].values[0]
            idx = driver_index_map[driver_id]
            ax.plot(range(race_frame), Ratings_History[idx, :race_frame], label=name)
            any_valid = True
        except IndexError:
            st.warning(f"Driver '{name}' not found")

    if any_valid:
        ax.set_xlabel("Race Number")
        ax.set_ylabel("Elo Rating")
        ax.set_title(f"Elo Ratings up to Race {race_frame}")
        ax.legend()
        st.pyplot(fig)
