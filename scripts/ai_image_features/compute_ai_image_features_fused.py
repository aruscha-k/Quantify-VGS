import pandas as pd
from shapely.geometry import Polygon
from collections import defaultdict

import logging
from datetime import datetime

import scripts.config as CONF
#from scripts.debug_helpers import print_surfaces_at_coordinates, print_surface_and_detections
from scripts.ai_image_features.compute_ai_image_features_fused_coordinate_conversion import *
from scripts.ai_image_features.compute_ai_image_features_fused_functions import *
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
        logging.FileHandler(f"{CONF.log_file_dir}/{timestamp}_calculate_ai_image_features_FUSED.log"),    # log to file
        logging.StreamHandler()                             # log to console
    ]
)
logger = logging.getLogger(__name__)


def run_fusion(df5, det_pool, wall_homographies, common_wall_ids, eps, prediction_confidence_threshold, show_plot = False, img_file_dir = ""):
    """Find all available wall ids from Street view and oblique together
       for all images available for the wall ID, reproject all detections to real world coordinates to create a detection pool for both image types of the wall ID 

    Args:
        df5_sv (_type_): _description_
        df2_walls_gs_gdf (_type_): _description_
        eps_sv (_type_): _description_
        eps_oblique (_type_): _description_
        show_plot (bool, optional): _description_. Defaults to True.
    """
    records_for_new_df5 = []
    print(f"Running Fusion. Total unique wall IDs to process: {len(common_wall_ids)}")

    for wall_id in common_wall_ids:
        df5_wall_id = df5[df5['wall_id'] == wall_id]

        # --- Determine which perspectives contributed ---
        sources = set(src for _, _, src in det_pool[wall_id])  # <-- here
        if 'sv' in sources and 'oblique' in sources:
            fusion_source = 'both'
        elif 'sv' in sources:
            fusion_source = 'sv'
        else:
            fusion_source = 'oblique'

        print("Available sources for wall ID:", fusion_source)
        # --- Loop through each image for the wall ID ---
        for _, row in df5_wall_id.iterrows():
            file_name = f"{row['building_id']}_{row['ID']}_rectified"
            rectified_outside_surface_in_image = row['rectified_outside_surface_in_image'] #transformed_outside_surface_in_image
            wall_area_image = Polygon(rectified_outside_surface_in_image).area
            wall_area = row['area_m2']
            building_id = row['building_id']
            wall_image_id = row['ID']
            detections_for_image = []
            if wall_homographies[wall_image_id] == {}:
                print(f"Skipping wall ID {wall_id}, image ID {wall_image_id} due to missing homography.")
                continue
            H_rect_to_wall2D, origin, x_axis, y_axis = wall_homographies[wall_image_id]['H_rect_to_wall2D'], wall_homographies[wall_image_id]['origin'], wall_homographies[wall_image_id]['x_axis'], wall_homographies[wall_image_id]['y_axis']
            
            # Apply homography to detections and recalculate px coords for each image respectively
            all_detections_px_dict = defaultdict(list) # format of detections is now dict with class names as keys and list of detections as values; Before unsorted list of tuples (class name, list of detections)
            for classname, class_dets, _ in det_pool[wall_id]:
                pixel_dets = calculate_pixel_coordinates(class_dets, H_rect_to_wall2D, origin, x_axis, y_axis)
                all_detections_px_dict[classname].append(pixel_dets) 

            # Process each class separately
            for classname, class_dets in all_detections_px_dict.items():
                print(f"Processing class: {classname} with {len(class_dets)} detections")
                
                num_clusters, clusters, labels, centroids = cluster_detections(class_dets, eps=eps)
                #if show_plot:
                    # draw_clusters(centroids, labels, is_3D=False)
                    # image_file = os.path.join(img_file_dir, f"{file_name}.jpg")
                    # draw_boxes_on_image(image_file, class_dets, "All re-projected detections")

                fused_boxes = fuse_cluster_bboxes_median(clusters, filter=True)
                if len(fused_boxes) == 0:
                    print("No fused boxes created, skipping alignment.")
                    continue
                aligned_boxes = align_boxes_grid(fused_boxes, eps=eps)

                for box in aligned_boxes:
                    fused_detection = {
                        'Image': file_name + ".jpg",
                        'class_name': classname,
                        'polygon': box,
                        'confidence': 1.0  # Placeholder confidence
                    }
                    detections_for_image.append(fused_detection)
        
                # Draw boxes per class
                #if show_plot:
                    #draw_clusters(centroids, labels, is_3D=False)
                    #image_file = os.path.join(img_file_dir, f"{file_name}.jpg")
                    # draw_boxes_on_image(image_file, class_dets, "All re-projected detections")
                    #draw_boxes_on_image(image_file, fused_boxes, "Fused boxes")
                    #draw_boxes_on_image(image_file, aligned_boxes, "Aligned fused boxes")

            detections_df = pd.DataFrame.from_records(detections_for_image)
            detections_df, error_message = preprocess_detections(detections_df,  wall_area_image, prediction_confidence_threshold)
            if detections_df is None:
                #logger.debug(f"For: building id {building_id}, Wall id {wall_image_id}: {error_message}")
                continue

            class_names_to_process = ['window', 'door', 'balcony']
            # only take those classes from  detections df
            detections = detections_df[detections_df['class_name'].isin(class_names_to_process)]['polygon'].tolist() 
            wwr = calculate_wwr(detections, rectified_outside_surface_in_image)
            swa = calculate_swa(wwr, wall_area)

            if 'recording_id' in row:
                record = {
                    'ID': wall_image_id,
                    'wall_id': wall_id,
                    'building_id': building_id,
                    'recording_id': row['recording_id'],
                    'wwr': wwr,
                    'swa': swa,
                    'area_m2': wall_area,
                    'fusion_source': fusion_source
                }
            else:
                record = {
                    'ID': wall_image_id,
                    'wall_id': wall_id,
                    'building_id': building_id,
                    'FILENAME': row['FILENAME'],
                    'wwr': wwr,
                    'swa': swa,
                    'area_m2': wall_area,
                    'fusion_source': fusion_source
                }                
            records_for_new_df5.append(record)
    new_df5 = pd.DataFrame.from_records(records_for_new_df5)
    return new_df5
      


def main(run_oblique: bool = True, run_streetview: bool = False):
    CONFIDENCE_THRESHOLD = 0.7

    df5_oblique = pd.read_pickle(CONF.df5_proj_fp)
    df5_streetview = pd.read_pickle(CONF.df5_streetview_fp)
    df2_walls_gs_gdf = pd.read_pickle(CONF.dataframes_dir + "/df2_walls_02_all_lod_factors.pkl") #needed for wall normal

    common_wall_ids = get_all_wall_ids(df5_streetview, df5_oblique)
    det_pool, wall_homographies = create_detection_pool_and_homographies(common_wall_ids, df2_walls_gs_gdf, df5_streetview, df5_oblique)
    # ----------
    # obliques
    # ----------
    if run_oblique:
        
        df5_oblique = run_house_nohouse_inference(df5_oblique, CONF.output_dst_oblique_dir, CONF.oblique_house_nohouse_model, CONFIDENCE_THRESHOLD)
        df5_oblique.to_pickle(CONF.dataframes_dir + "df5_oblique_afterhouse.pkl")
        # df5_projections = pd.read_pickle(CONF.dataframes_dir + "df5_oblique_afterhouse.pkl")

        df5_oblique_fused = run_fusion(df5_oblique, det_pool, wall_homographies, common_wall_ids, eps=45, prediction_confidence_threshold = 0.75, show_plot = False, img_file_dir=CONF.output_dst_oblique_dir)
        df5_oblique_fused.to_pickle(CONF.dataframes_dir + "df5_oblique_fused.pkl")

        # df5_projections_with_features = pd.read_pickle(CONF.dataframes_dir + "df5_oblique_fused.pkl")
        df5_oblique_averaged = calculate_averages_per_wall(df5_oblique_fused)
        df5_oblique_averaged.to_pickle(CONF.dataframes_dir + "df5_oblique_fused_averaged_per_wall.pkl")

    # ----------
    # street views
    # ----------
    if run_streetview:
        df5_streetviews = run_house_nohouse_inference(df5_streetviews, CONF.output_dest_streetview_dir, CONF.streetview_house_no_house_model, CONFIDENCE_THRESHOLD)
        df5_streetviews.to_pickle(CONF.dataframes_dir + "df5_streetview_afterhouse.pkl")
        # df5_streetviews = pd.read_pickle(CONF.dataframes_dir + "df5_streetview_afterhouse.pkl") 

        df5_streetview_fused = run_fusion(df5_streetview, det_pool, wall_homographies, common_wall_ids, eps=30, prediction_confidence_threshold = 0.7, show_plot = False, img_file_dir=CONF.output_dest_streetview_dir)
        df5_streetview_fused.to_pickle(CONF.dataframes_dir + "df5_streetview_fused.pkl") 

        # df5_streetviews_with_features = pd.read_pickle(CONF.dataframes_dir + "df5_streetview_fused.pkl")
        df5_streetview_averaged = calculate_averages_per_wall(df5_streetview_fused)
        df5_streetview_averaged.to_pickle(CONF.dataframes_dir + "df5_streetview_fused_averaged_per_wall.pkl")




if __name__ == "__main__":
    main(run_oblique=True, run_streetview=True)