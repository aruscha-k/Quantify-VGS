# config file to configure paths, outcommented paths are not included in paper version

import os

def validate_path(path: str) -> str:
    if not os.path.exists(path):
        print("CHECK PATH:", path)
        return ""
    return path

def validate_or_create(path: str) -> str:
    if not os.path.exists(path):
        print("CREATING:", path)
        os.makedirs(path)
    return path


BASE__FOLDER = validate_path("") #BASE FOLDER containing scripts/, daten/, imgs/, models/, dataframes/, log/


#------------
# logging & processing
#------------
log_file_dir = validate_or_create(os.path.join(BASE__FOLDER, "log/"))
#cut_out_walls_pkl_file = os.path.join(log_file_dir, "cut_out_walls.pkl") #might only be created during and not exists from start
#cyclomedia_camera_cache_pkl_file = os.path.join(log_file_dir, "cyclomedia_camera_cache.pkl") #might only be created during and not exists from start
#cyclomedia_extracted_images_pkl_file = os.path.join(log_file_dir, "cyclomedia_extracted_images.pkl") #created during

#------------
# input data
#------------
data_dir = validate_path(os.path.join(BASE__FOLDER, "daten/"))

# oblique metadata files
# photos_df_pkl_file = validate_path(os.path.join(data_dir, "oblique/Metadata/Image_Orientation_XYOPK/Koeln_Leverkusen_UM.prj_photos_df.pkl"))
# camera_df_pkl_file = validate_path(os.path.join(data_dir, "oblique/Metadata/Image_Orientation_XYOPK/Koeln_Leverkusen_UM.prj_camera_df.pkl"))
# oblique_footprint_nord_fp =  validate_path(os.path.join(data_dir, "oblique/Metadata/ObliqueFootprints_SHP/Leverkusen_Footprints_NORD.shp"))
# oblique_footprint_sued_fp = validate_path(os.path.join(data_dir, "oblique/Metadata/ObliqueFootprints_SHP/Leverkusen_Footprints_SUED.shp"))
# oblique_footprint_west_fp = validate_path(os.path.join(data_dir, "oblique/Metadata/ObliqueFootprints_SHP/Leverkusen_Footprints_WEST.shp"))
# oblique_footprint_ost_fp = validate_path(os.path.join(data_dir, "oblique/Metadata/ObliqueFootprints_SHP/Leverkusen_Footprints_OST.shp"))

# images
# base_src_obliques_folder_path = validate_path(os.path.join(data_dir, "oblique/Oblique-Bilder/"))

# lod data
lod2_data_dir = validate_path(os.path.join(data_dir, "lod2_data/"))


#------------
# output data
#------------
image_data_dir = validate_path(os.path.join(BASE__FOLDER, "imgs/"))
# out oblique cutouts
output_dst_oblique_dir = os.path.join(image_data_dir, "oblique-walls/") # created during
# output_oblique_detections_dir = os.path.join(image_data_dir, "detections_yolo11_train2/") # downloaded from cluster

# street view
image_data_dir_surface = validate_path(os.path.join(image_data_dir,'buildings_from_surface/'))
output_dest_streetview_dir = os.path.join(image_data_dir, "streetviews") #created during
# output_streetview_detections_dir = os.path.join(image_data_dir, "streetview_detections_detectron/") # downloaded from cluster

# ---------------
# dataframes
# ---------------
dataframes_dir = validate_or_create(os.path.join(BASE__FOLDER, "dataframes/"))

df1_fp = validate_path(os.path.join(dataframes_dir, "df1_parts.pkl"))
df2_fp = validate_path(os.path.join(dataframes_dir, "df2_walls_02_all_lod_factors.pkl"))
# df4_proj_fp = os.path.join(dataframes_dir,"df4_projections.pkl") #created during
# df5_proj_fp = os.path.join(dataframes_dir,"df5_cut_rectified.pkl") #created during

# df4_streetview_fp = os.path.join(dataframes_dir, "df4_streetview.pkl")
# df5_streetview_fp = os.path.join(dataframes_dir, "df5_streetview.pkl")

# ---------------
# models
# ---------------
# model_dir = validate_path(os.path.join(BASE__FOLDER, "models/"))
# streetview_house_no_house_model = os.path.join(model_dir, "streetview_House_or_not_5epochs.keras")
# oblique_house_nohouse_model = os.path.join(model_dir, "oblique_house_or_not_ALL_4epochs_2ft_acc0.926.keras")