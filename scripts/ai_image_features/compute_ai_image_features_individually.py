import os
import pandas as pd
from shapely.geometry import Polygon

import logging
from datetime import datetime

import scripts.config as CONF
from scripts.debug_helpers import print_surfaces_at_coordinates, print_surface_and_detections
from scripts.ai_image_features.compute_ai_image_features_functions import *
from scripts.ai_image_features.clean_house_nohouse import run_house_nohouse_inference

# ------
# logger
# ------
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
logging.basicConfig(
    level=logging.DEBUG,                                   # INFO/ DEBUG
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(f"{CONF.log_file_dir}/{timestamp}_calculate_ai_image_features_INDIVIDUALLY.log"),    # log to file
        logging.StreamHandler()                             # log to console
    ]
)
logger = logging.getLogger(__name__)


# iterate df5, read projection pkl file and calculate image features: wwr, swa
def calculate_image_features(df5: pd.DataFrame, detections_folder, prediction_confidence_threshold, image_folder = "", show_plot=False)-> pd.DataFrame:
    logger.info(f"Calculating image features (WWR, SWA) for given dataframe and detections folder {detections_folder}")
    df5 = df5.copy()
    wwr_results = []
    swa_results = []

    #only look at entries where is_house == True
    df5 = df5[df5['is_house'] == True]
    for _, row in df5.iterrows():
        wall_image_id = row['ID']
        wall_area = row['area_m2']
        building_id = row['building_id']
        file_name = f"{building_id}_{wall_image_id}_rectified"
        rectified_outside_surface_in_image = row['rectified_outside_surface_in_image'] #transformed_outside_surface_in_image
        wall_area_image = Polygon(rectified_outside_surface_in_image).area
          
        detections_pkl = os.path.join(detections_folder, file_name + ".pkl")
        # check if detection folder / file exists:
        if not os.path.isdir(detections_folder) or not os.path.isfile(detections_pkl) or os.path.getsize(detections_pkl) == 0:
            logger.debug(f"For: building id {building_id}, Wall id {wall_image_id}: No (valid) detection file for Building ID. Skipping.")
            wwr_results.append(None)
            swa_results.append(None)
            continue

        detections_df = pd.read_pickle(detections_pkl)
        detections_df, error_message = preprocess_detections(detections_df, wall_area_image, prediction_confidence_threshold)
        if detections_df is None:
            logger.debug(f"For: building id {building_id}, Wall id {wall_image_id}: {error_message}")
            wwr_results.append(None)
            swa_results.append(None)
            continue
        #corrected_outside_surface = remove_roof_from_surface_area(outside_surface_image, detections_df, roof_label="roof") #use roof detections to correct outside surface area in oblique image #FIXME: not working correctly
        class_names_to_process = ['window', 'door', 'balcony', 'fenster', 'haustuer', 'shop'] #FIXME: HACKY!!!!
        # only take those classes from  detections df
        detections = detections_df[detections_df['class_name'].isin(class_names_to_process)]['polygon'].tolist() 
        
        # optional for debugging: draw surface and detections on wall
        if show_plot:
            class_names = detections_df['class_name'].tolist()
            confidences = detections_df['confidence'].tolist()
            img_path = os.path.join(image_folder, file_name +".jpg")
            print_surface_and_detections(img_path, rectified_outside_surface_in_image, detections, class_names, confidences)
            print("detections:", list(zip(detections_df['class_name'], detections_df['confidence'])))
                
        wwr = calculate_wwr(detections, rectified_outside_surface_in_image)
        wwr_results.append(wwr)
        swa = calculate_swa(wwr, wall_area)
        swa_results.append(swa)

    df5['wwr'] = wwr_results
    df5['swa'] = swa_results
    return df5
      


def main(run_oblique: bool = True, run_streetview: bool = False):
    CONFIDENCE_THRESHOLD = 0.7
    # ----------
    # obliques
    # ----------
    if run_oblique:
        #df5_projections = pd.read_pickle(CONF.df5_proj_fp)
        #df5_projections = run_house_nohouse_inference(df5_projections, CONF.output_dst_oblique_dir, CONF.oblique_house_nohouse_model, CONFIDENCE_THRESHOLD)

        #df5_projections.to_pickle(CONF.dataframes_dir + "df5_oblique_afterhouse.pkl")
        df5_projections = pd.read_pickle(CONF.dataframes_dir + "df5_oblique_afterhouse.pkl")

        df5_projections_with_features = calculate_image_features(
            df5_projections,
            CONF.output_oblique_detections_dir,
            CONFIDENCE_THRESHOLD
        )
        df5_projections_with_features.to_pickle(CONF.dataframes_dir + "df5_oblique_with_image_features.pkl")

        # df5_projections_with_features = pd.read_pickle(CONF.dataframes_dir + "df5_oblique_with_image_features.pkl")
        df5_oblique_averaged = calculate_averages_per_wall(df5_projections_with_features)
        df5_oblique_averaged.to_pickle(CONF.dataframes_dir + "df5_oblique_averaged_per_wall.pkl")

    # ----------
    # street views
    # ----------
    if run_streetview:
        #df5_streetviews = pd.read_pickle(CONF.df5_streetview_fp)
        #df5_streetviews = run_house_nohouse_inference(df5_streetviews, CONF.output_dest_streetview_dir, CONF.streetview_house_no_house_model, CONFIDENCE_THRESHOLD)

        #df5_streetviews.to_pickle(CONF.dataframes_dir + "df5_streetview_afterhouse.pkl")
        df5_streetviews = pd.read_pickle(CONF.dataframes_dir + "df5_streetview_afterhouse.pkl") 

        df5_streetviews_with_features = calculate_image_features(
            df5_streetviews,
            CONF.output_streetview_detections_dir,
            CONFIDENCE_THRESHOLD
        )
        df5_streetviews_with_features.to_pickle(CONF.dataframes_dir + "df5_streetview_with_image_features.pkl") 

        # df5_streetviews_with_features = pd.read_pickle(CONF.dataframes_dir + "df5_streetview_with_image_features.pkl")
        df5_streetview_averaged = calculate_averages_per_wall(df5_streetviews_with_features)
        df5_streetview_averaged.to_pickle(CONF.dataframes_dir + "df5_streetview_averaged_per_wall.pkl")




if __name__ == "__main__":
    main(run_oblique=True, run_streetview=True)