# Maximizing Data Coverage: Fusing Street View and Oblique Imagery to Quantify Vertical Greenery Potential

This is the workflow to preprocess wall surfaces from LoD2 data (merging, filtering) and derive factors from LoD2 data. 
The workflow to extract images cannot be shared (see paper for explaination).
The methods to run the fusion and calculate the potential index is shared in theory. The data is described below.

# Data needed:

## LoD2 data:

Openly available: https://www.opengeodata.nrw.de/produkte/geobasis/3dg/lod2_gml/lod2_gml/ 

LoD2 data may differ in property names and available attributes. Please adjust code, if different LoD2 data is used. A sample of the used data in this work can be found in the directory "daten". 

## Oblique and street view 

Due to licensing restrictions, the image data cannot be shared. See below how you can implement these steps on your own.

# Method

## 1. LoD2 data
 
1. Run script _01_read_lod_data.py_ to read GML data -> creates df1_parts.pkl df2_walls_00.pkl, df3_roofs.pkl
2. Run script _02_compute_features_refactored.py_ to compute first factors that can be derived from LoD2 and filter walls -> creates df2_walls_02_all_lod_factors.pkl

## 2. Image extraction and AI computation

Adjust here that you have 1-3 images for each wall for each source with the respective detections:
For all available images with successfull a dataframe is needed containing the columns:

Streetview: ['wall_id', 'building_id', 'wall_height', 'wall_width', 'area_m2',
       'surface_coordinates', 'outside_bounding_box', 'recording_IDs'] where _outside_bounding_box_ is a rectangle encompassing the surface if it is not rectangular, and _recording_IDs_ are the IDs of the photo

Oblique: ['ID', 'wall_id', 'building_id', 'wall_height', 'wall_width', 'area_m2',
       'wall_camera_angle_deg', 'surface_coordinates', 'outside_bounding_box',
       'projected_wall_surface', 'projected_ground_surface',
       'projected_outside_bbox', 'ORI', 'FILENAME', 'camera_id', 'img_width',
       'img_height', 'eligible_for_photo',
       'rectified_outside_surface_in_image', 'cut_out_success']

where _projected_wall_surface_ & _projected_ground_surface_ are the 2D px values for the respective 3D LoD2 values. They are needed to check if a wall is within image bounds.
_projected_outside_bbox_ is the 2D px value for the outside_bounding_box. 

Train object detection networks that return bounding boxes for each facade element in the format of $[(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1,y1)]$ 
Save detections image wise as pd.DataFrame as pkl in the following format:
For each detection create a row with these properties:
columns = ['image', 'x1', 'y1', 'x2', 'y2', 'polygon', 'confidence', 'class_id', 'class_name'] 
where _'image'_ is the image filename, the other properties are for each detection in the image seperately.

## 3. Factor Calculation & Image fusion 

Run _compute_ai_image_features_individually.py_ to compute features without fusion approach and _compute_ai_image_features_fused.py_ to compute features with fusion approach.

The resulting df contains the columns which can be used for factor calculation:
['ID', 'wall_id', 'building_id', 'FILENAME', 'wwr', 'swa', 'area_m2',
       'fusion_source']