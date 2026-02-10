# Maximizing Data Coverage: Fusing Street View and Oblique Imagery to Quantify Vertical Greenery Potential

This is the workflow to preprocess wall surfaces from LoD2 data (merging, filtering) and derive factors from LoD2 data. 
The workflow to extract images cannot be shared (see paper for explaination). If you have images of (the same) walls from street view and oblique perspectives, it is described how object detection results must be saved to use the fusion and index computation methods.

The data is described below.

# Data needed:

## LoD2 data:

Openly available: https://www.opengeodata.nrw.de/produkte/geobasis/3dg/lod2_gml/lod2_gml/ 

LoD2 data may differ in property names and available attributes. Please adjust code, if different LoD2 data is used. A sample of the used data in this work can be found in the directory "daten". 

## Oblique and street view 

Due to licensing restrictions, the image data cannot be shared. See below how you can implement these steps on your own.

# Methods

## 1. LoD2 data
 
1. Run script _01_read_lod_data.py_ to read GML data -> creates df1_parts.pkl df2_walls_00.pkl, df3_roofs.pkl
2. Run script _02_compute_features_refactored.py_ to compute first factors that can be derived from LoD2 and filter walls -> creates df2_walls_02_all_lod_factors.pkl

## 2. Image extraction

This part of the data/code cannot be shared. To be able to run computation of wwr and swa (in ai_image_features) and calculate potential index, the steps are explained in theory if you have image data of your own.
You need to have 1-3 images for each wall *for each source* with the obligatory detections of classes: window, door, roof. Optional classes: balcony, vegetation.


While extracting images, a dataframe called df4 is created to save if an image was available for a wall (sv)/ or if the cut out was successfull (oblique) with the following properties. An ID is created representing the filename of the photo which consists of the wall ID and the recording ID (sv) / oblique file ID (oblique)

Streetview: ['wall_id', 'building_id', 'wall_height', 'wall_width', 'area_m2',
       'surface_coordinates', 'outside_bounding_box', 'recording_IDs']

where
       - _outside_bounding_box_ is a rectangle encompassing the surface if it is not rectangular (calculated in compute_features_refactored.py), 
       - _recording_IDs_ are the IDs of the recording points of the photo (are not needed anymore for calculations but code will crash if there is non so just assign any string value)
       - 'wall_id', 'building_id', 'wall_height', 'wall_width', 'area_m2', 'surface_coordinates' come from the LoD data

Oblique: ['ID' ({oblique_file_id}_{wall_id}), 'wall_id', 'building_id', 'wall_height', 'wall_width', 'area_m2',
       'wall_camera_angle_deg', 'surface_coordinates', 'outside_bounding_box',
       'projected_wall_surface', 'projected_ground_surface',
       'projected_outside_bbox', 'ORI', 'FILENAME', 'camera_id', 'img_width',
       'img_height', 'eligible_for_photo',
       'rectified_outside_surface_in_image', 'cut_out_success']

where 
       - _projected_wall_surface_ & _projected_ground_surface_ are the 2D px values in the image for the respective 3D LoD2 values of wall surface and ground surface. They are needed to check if a wall is within image bounds. They are calculated to cut out the respective wall from the oblique image.
       - _projected_outside_bbox_ is the 2D px value for the outside_bounding_box. It is also calculated when extracting the wall from the oblique image.

From all values in df4 a new dataframe is created (df5) that contains only rows, for which an image was downloaded. These dataframes must be created manually to be able to further use the code. Their paths must be specified in _config.py_.

## 3. Object detection 

Train object detection networks that return bounding boxes for each facade element. The detections of all images must be saved individually (one df per image, saved as pkl-file) in a dataframe containing the properties (Adjust the bounding box to be able to save the following properties):

$columns = ['image', 'x1', 'y1', 'x2', 'y2', 'polygon', 'confidence', 'class\_id', 'class\_name']$

where
       - _'image'_ is the image filename, 
       - _''x1', 'y1', 'x2', 'y2'_ are the lower left and upper right corner of the detection bbox
       - _polygon_ is the closed bounding box as $[(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1,y1)]$ 
       - _'confidence'_ of the detection (from network) 
       - _'class_id'_ id of the class (int)
       - _'class_name'_ name of the class (str)

Save detections image-wise as pd.DataFrame as pkl.


## 4. Image fusion, calculation of wwr and swa & Index Calculation
Run _compute_ai_image_features_individually.py_ to compute features without fusion approach and _compute_ai_image_features_fused.py_ to compute features with fusion approach.

Within this code, a check runs if an image shows a valid house/wall or not (code for this is in clean_house_nohouse.py). The model cannot be shared due to licensing restrictions. You can disable this check by commenting it out, which will influence the results. To train a model for this clean up step, divide images into "good and bad" and train a simple model for few epochs. 

The resulting df contains the following columns which can be used for factor calculation:
$columns = ['ID', 'wall_id', 'building\_id', 'FILENAME', 'wwr', 'swa', 'area_m2', 'fusion\_source']$

For index calculation run _calculate_potential_index.ipynb_