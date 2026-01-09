
import os
import logging
from datetime import datetime

import scripts.config as CONF

from scripts.lod2.compute_features_functions import *
from scripts.lod2.compute_features_wall_join import find_walls_to_join, join_walls


# ------
# logger
# ------
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
logging.basicConfig(
    level=logging.DEBUG,                                   # INFO/ DEBUG
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(f"{CONF.log_file_dir}/{timestamp}_compute_features.log"),    # log to file
        logging.StreamHandler()                             # log to console
    ]
)
logger = logging.getLogger(__name__)


# ------
# params
# ------
min_wall_area_thresh = 4
filter_walls = True
save_steps = True


def main():
    leverkusen_gdf = read_original_lod_data(filepath=os.path.join(CONF.dataframes_dir, 'df1_parts.pkl'))
    df2_walls_gs_gdf = read_data(df1_filename="df1_parts.pkl", df2_filename="df2_walls_00.pkl")

    # 1. compute area
    df2_walls_gs_gdf = calculate_wall_area(df2_walls_gs_gdf, min_wall_area_thresh=min_wall_area_thresh, filter_walls=filter_walls)
    
    # 2. check if wall is vertical 
    df2_walls_gs_gdf = calc_normal(df2_walls_gs_gdf)
    df2_walls_gs_gdf = check_verticality_of_walls(df2_walls_gs_gdf, filter_walls)

    # 3. find walls to join
    to_join = find_walls_to_join(df2_walls_gs_gdf)
    
    if save_steps:
        to_join.to_pickle(os.path.join(CONF.dataframes_dir, "to_join.pkl"))

    # 4. join walls
    df2_walls_gs_df = join_walls(df2_walls_gs_gdf, to_join)
    df2_walls_gs_gdf = make_geodataframe(df2_walls_gs_df, "ground_surface")
    if save_steps:
        df2_walls_gs_gdf.to_pickle(os.path.join(CONF.dataframes_dir, "df2_joined_01_after_wall_merging.pkl"))
        
    # 4. find ground bound, wall height, orientation
    df2_walls_gs_gdf = find_ground_bound(df2_walls_gs_gdf, filter_walls)
    df2_walls_gs_gdf = calculate_wall_height_and_width(df2_walls_gs_gdf)
    df2_walls_gs_gdf = get_wall_orientation(df2_walls_gs_gdf)
    df2_walls_gs_gdf = calculate_outside_factors(df2_walls_gs_gdf, leverkusen_gdf)
    logger.info(f"Done and saving now, number of walls {len(df2_walls_gs_gdf)}")

    df2_walls_gs_gdf.to_pickle(os.path.join(CONF.dataframes_dir, "df2_walls_02_all_lod_factors.pkl"))
    




if __name__ == "__main__":
    main()