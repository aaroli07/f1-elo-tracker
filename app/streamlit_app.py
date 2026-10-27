import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Streamlit may put only the app directory on sys.path when launched from WSL
# or by an IDE. Add the repository root so the sibling src package is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_load import DATA_DIR, load_all
from src.pipeline import build_ratings


@st.cache_data(show_spinner="Loading race history...")
def load_rating_snapshot(data_fingerprint):
    del data_fingerprint  # Included in the cache key to refresh when CSVs change.
    drivers, races, results = load_all()
    history, index_map, race_ids = build_ratings(drivers, races, results)
    return drivers, races, history, index_map, race_ids


data_fingerprint = tuple(
    (path.name, path.stat().st_size, path.stat().st_mtime_ns)
    for path in (
        DATA_DIR / "DriversMOD.csv",
        DATA_DIR / "RacesMOD.csv",
        DATA_DIR / "results.csv",
    )
)
data_drivers, data_races, ratings_history, driver_index_map, sorted_race_ids = (
    load_rating_snapshot(data_fingerprint)
)

st.title("Formula 1 Driver Elo")
st.caption(
    "Ratings compare teammates, with a smaller adjustment for overall race "
    "position."
)

if ratings_history.shape[1] == 0:
    st.warning("No race results are available to rate.")
    st.stop()

drivers_with_ratings = {}
for row in data_drivers.itertuples(index=False):
    driver_id = row.driverId
    if driver_id not in driver_index_map:
        continue
    history = ratings_history[driver_index_map[driver_id]]
    if not np.isfinite(history).any():
        continue
    full_name = f"{row.forename} {row.surname}".strip()
    driver_ref = row.driverRef
    drivers_with_ratings[driver_id] = f"{full_name} ({driver_ref})"

selected_driver_ids = st.multiselect(
    "Search for one or more drivers",
    options=list(drivers_with_ratings),
    max_selections=10,
    format_func=lambda driver_id: drivers_with_ratings[driver_id],
    help="Search by driver name or reference, then select multiple drivers to compare.",
)

race_count = st.slider(
    "Show history through race number",
    min_value=1,
    max_value=ratings_history.shape[1],
    value=ratings_history.shape[1],
)

if selected_driver_ids:
    metric_columns = st.columns(len(selected_driver_ids))
    for column, driver_id in zip(metric_columns, selected_driver_ids):
        driver_history = ratings_history[driver_index_map[driver_id]]
        available = np.flatnonzero(np.isfinite(driver_history))
        latest_index = available[-1]
        column.metric(
            drivers_with_ratings[driver_id],
            f"{driver_history[latest_index]:.1f}",
            help=f"Latest rating after race {latest_index + 1}.",
        )

    fig, ax = plt.subplots()
    x_values = np.arange(1, race_count + 1)
    for driver_id in selected_driver_ids:
        driver_history = ratings_history[driver_index_map[driver_id]][:race_count]
        ax.plot(
            x_values,
            driver_history,
            label=drivers_with_ratings[driver_id],
        )

    ax.set_xlabel("Race number")
    ax.set_ylabel("Elo rating")
    ax.set_title(f"Driver ratings through race {race_count}")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig)
else:
    st.info("Select one or more drivers to see their latest rating and history.")
