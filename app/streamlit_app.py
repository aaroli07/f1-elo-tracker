import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
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
    race_number_by_id = {race_id: index + 1 for index, race_id in enumerate(race_ids)}
    completed_laps = pd.to_numeric(results["laps"], errors="coerce")
    starts = results.loc[completed_laps.ge(1), ["driverId", "raceId"]]
    driver_race_numbers = {
        driver_id: sorted(
            {
                race_number_by_id[race_id]
                for race_id in race_group["raceId"]
                if race_id in race_number_by_id
            }
        )
        for driver_id, race_group in starts.groupby("driverId", sort=False)
    }
    return drivers, races, history, index_map, race_ids, driver_race_numbers


data_fingerprint = tuple(
    (path.name, path.stat().st_size, path.stat().st_mtime_ns)
    for path in (
        DATA_DIR / "DriversMOD.csv",
        DATA_DIR / "RacesMOD.csv",
        DATA_DIR / "results.csv",
    )
)
(
    data_drivers,
    data_races,
    ratings_history,
    driver_index_map,
    sorted_race_ids,
    driver_race_numbers,
) = load_rating_snapshot(data_fingerprint)

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

if selected_driver_ids:
    history_start = min(
        driver_race_numbers[driver_id][0] for driver_id in selected_driver_ids
    )
    history_end = max(
        driver_race_numbers[driver_id][-1] for driver_id in selected_driver_ids
    )
else:
    history_start = 1
    history_end = ratings_history.shape[1]

if history_start == history_end:
    race_count = history_end
    st.caption(f"History is available through race {race_count}.")
else:
    if "race_count" not in st.session_state:
        st.session_state["race_count"] = history_end
    else:
        st.session_state["race_count"] = min(
            max(st.session_state["race_count"], history_start), history_end
        )
    race_count = st.slider(
        "Show history through race number",
        min_value=history_start,
        max_value=history_end,
        key="race_count",
    )

if selected_driver_ids:
    metric_columns = st.columns(len(selected_driver_ids))
    for column, driver_id in zip(metric_columns, selected_driver_ids):
        driver_history = ratings_history[driver_index_map[driver_id]]
        latest_race = driver_race_numbers[driver_id][-1]
        column.metric(
            drivers_with_ratings[driver_id],
            f"{driver_history[latest_race - 1]:.1f}",
            help=f"Latest rating after their last start, race {latest_race}.",
        )

    chart_rows = []
    for driver_id in selected_driver_ids:
        driver_history = ratings_history[driver_index_map[driver_id]][:race_count]
        first_race = driver_race_numbers[driver_id][0]
        last_race = min(race_count, driver_race_numbers[driver_id][-1])
        for race_number in range(first_race, last_race + 1):
            rating = driver_history[race_number - 1]
            if np.isfinite(rating):
                chart_rows.append(
                    {
                        "Race number": race_number,
                        "Elo rating": rating,
                        "Driver": drivers_with_ratings[driver_id],
                    }
                )

    chart_data = pd.DataFrame(chart_rows)
    rating_chart = (
        alt.Chart(chart_data)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X(
                "Race number:Q",
                title="Race number",
                scale=alt.Scale(domain=[history_start, race_count]),
                axis=alt.Axis(format="d", tickMinStep=1),
            ),
            y=alt.Y("Elo rating:Q", title="Elo rating", scale=alt.Scale(zero=False)),
            color=alt.Color("Driver:N", title=None, legend=alt.Legend(orient="bottom")),
            tooltip=[
                alt.Tooltip("Driver:N", title="Driver"),
                alt.Tooltip("Race number:Q", title="Race", format="d"),
                alt.Tooltip("Elo rating:Q", title="Elo rating", format=".1f"),
            ],
        )
        .properties(
            height=440,
            title=f"Elo ratings through race {race_count}",
        )
        .interactive()
        .configure_view(stroke=None)
        .configure_axis(gridColor="#E6EAF0", domain=False, tickColor="#CCD2DA")
        .configure_legend(labelLimit=240, symbolStrokeWidth=3)
    )
    st.altair_chart(
        rating_chart,
        width="stretch",
        alt="Selected drivers' Elo ratings over their race histories",
    )
else:
    st.info("Select one or more drivers to see their latest rating and history.")
