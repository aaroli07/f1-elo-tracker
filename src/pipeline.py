

import numpy as np
import pandas as pd
from src.elo import calculate_new_elo  # import your Elo math functions

def build_ratings(data_drivers, data_races, data_results):
    """
    Compute Elo ratings for all drivers over all races.
    
    Args:
        data_drivers (pd.DataFrame): Drivers data.
        data_races (pd.DataFrame): Races data.
        data_results (pd.DataFrame): Results data.
        
    Returns:
        Ratings_History (np.ndarray): Elo ratings matrix [drivers x races].
        driver_index_map (dict): Mapping driverId -> row index in Ratings_History.
        sorted_race_ids (np.ndarray): Array of race IDs sorted by date.
    """
    
    # --- Extract arrays ---
    results_raceId = np.array(data_results['raceId'])
    results_driverId = np.array(data_results['driverId'])
    results_constructorId = np.array(data_results['constructorId'])
    results_positionOrder = np.array(data_results['position'])
    
    race_raceId = np.array(data_races['raceId'])
    race_date = np.array(data_races['date'])
    
    # --- Sort races by date ---
    sorted_race_ids = sorted(race_raceId, key=lambda rid: race_date[np.where(race_raceId == rid)[0][0]])
    
    # --- Merge results with race dates ---
    merged_results = pd.merge(
        data_results,
        data_races[['raceId','date']],
        how='left',
        on='raceId'
    )
    merged_sorted = merged_results.sort_values(by='date')
    
    # Arrays from merged results
    results_raceId = np.array(merged_sorted['raceId'])
    results_driverId = np.array(merged_sorted['driverId'])
    results_constructorId = np.array(merged_sorted['constructorId'])
    results_positionOrder = np.array(merged_sorted['position'])
    
    # --- Driver indexing ---
    unique_driver_ids = np.unique(results_driverId)
    driver_index_map = {driver_id: idx for idx, driver_id in enumerate(unique_driver_ids)}
    
    n_drivers = len(unique_driver_ids)
    n_races = len(np.unique(results_raceId))
    
    # --- Initialize Ratings_History ---
    Ratings_History = np.zeros((n_drivers, n_races))
    Ratings_History[:,0] = 1000  # initial Elo
    
    # --- Main loop ---
    number_of_iterations = 0
    for race_id in sorted_race_ids:
        race_indices = np.where(results_raceId == race_id)[0]
        race_drivers = results_driverId[race_indices]
        race_constructors = results_constructorId[race_indices]
        race_positions = results_positionOrder[race_indices]
        
        # Carry over previous ratings
        if number_of_iterations > 0:
            Ratings_History[:, number_of_iterations] = Ratings_History[:, number_of_iterations-1]
        
        # Loop over constructors (teams)
        for constructor in np.unique(race_constructors):
            team_indices = race_indices[race_constructors == constructor]
            if len(team_indices) != 2:
                continue
            
            # Driver IDs and positions
            d1_idx, d2_idx = team_indices
            driver1_id = results_driverId[d1_idx]
            driver2_id = results_driverId[d2_idx]
            
            try:
                pos1 = int(results_positionOrder[d1_idx])
                pos2 = int(results_positionOrder[d2_idx])
            except ValueError:
                continue
            if pos1 == r"\N" or pos2 == r"\N":
                continue
            
            # Current Elo
            elo1 = Ratings_History[driver_index_map[driver1_id], number_of_iterations]
            elo2 = Ratings_History[driver_index_map[driver2_id], number_of_iterations]
            
            # Update Elo
            new_elo1, new_elo2 = calculate_new_elo(elo1, elo2, pos1, pos2)
            Ratings_History[driver_index_map[driver1_id], number_of_iterations] = new_elo1
            Ratings_History[driver_index_map[driver2_id], number_of_iterations] = new_elo2
        
        number_of_iterations += 1
    
    return Ratings_History, driver_index_map, sorted_race_ids
