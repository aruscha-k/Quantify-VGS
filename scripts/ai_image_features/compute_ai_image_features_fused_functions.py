import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
import os
import pandas as pd

import scripts.config as CONF
from scripts.ai_image_features.compute_ai_image_features_fused_functions import calculate_real_world_coordinates

def get_normalized_normal_vector_3D(wall_id, df2_walls_gs_gdf):
    """from existing df2 for walls get the normal vector

    Args:
        wall_id (str): 
        df2_walls_gs_gdf (pd.geodataframe): df2_walls with lod factors

    Returns:
        np.array: normal vector 3D normalized
    """
    normal_vector_3D = df2_walls_gs_gdf.loc[df2_walls_gs_gdf['wall_id'] == wall_id, 'normal_vector_3d'].values[0]
    normal_vector_3D = np.array(normal_vector_3D)
    normal_vector_3D /= np.linalg.norm(normal_vector_3D)
    return normal_vector_3D



def normalize_class_names(detections):
    """Because oblique and streetview CNNs have different class names, match class names first
    Args:
        detections (list (tuples)): class name, list of detections
    Returns:
        (list (tuples)): class name, list of detections
    """
    CLASS_NAME_MAPPING = {
        'fenster': 'window',
        'window': 'window',
        'door': 'door',
        'haustuer': 'door'
    }
    normalized_dets = []
    for classname, class_dets in detections:
        canonical_classname = CLASS_NAME_MAPPING.get(classname.lower(), classname.lower())
        normalized_dets.append((canonical_classname, class_dets))
    return normalized_dets


def find_detections(wall_id, df5, normalized_normal_3d, detections_dir, src):
    """takes all walls in the df5 with the wall id, 
        finds the detection pkl files, computes real world coordinates of the detections and saves homographies
    Args:
        wall_id (str): 
        df5 (pd.Dataframe): 
        normalized_normal_3d (np.array): normal vector 3D normalized of wall
        detections_dir (str): _description_
        src (str): source type (sv or oblique)

    Returns:
        detection_pool: dict (key wall ID) of list of detections as tuples (class_name, list of real world points)
        homographies: dict of homographies for each image ID
    """
    found_detections = list()
    homographies = dict()
    wall_sub_df5 = df5[df5['wall_id'] == wall_id]
    
    for idx, row in wall_sub_df5.iterrows():
        try:
            image_wall_id = row['ID']
            file_name = f"{row['building_id']}_{row['ID']}_rectified"
            # detections = pd.read_pickle(os.path.join(detections_dir, file_name + ".pkl")) #FIXME!!!!!! detections from single files are often corrupted use all detections.pkl instead more efficiently
            detections = pd.read_pickle(os.path.join(detections_dir, "all_detections.pkl"))
            detections = detections.loc[detections['image']== file_name+".jpg"]
            detections_with_world_coords, H_rect_to_wall2D, origin, x_axis, y_axis = calculate_real_world_coordinates(detections, row['rectified_outside_surface_in_image'], row['surface_coordinates'], normalized_normal_3d)
            homographies[image_wall_id] = {'H_rect_to_wall2D':H_rect_to_wall2D,
                                    'origin': origin, 
                                    'x_axis': x_axis, 
                                    'y_axis': y_axis }
            dets = detections_with_world_coords['real_world_detections'].tolist() #real_world_detections is column with tuples (class name, detections)
            norm_dets = normalize_class_names(dets)
            norm_dets_with_src = [(classname, poly, src) for classname, poly in norm_dets]
            found_detections.extend(norm_dets_with_src)

        except Exception as e:
            print(f"Error processing wall ID {file_name}: {e}")
            homographies[image_wall_id] = {}
            continue
    return found_detections, homographies


def create_detection_pool_and_homographies(common_wall_ids, df2_walls_gs_gdf, df5_sv, df5_oblique):
    """For a list of wall IDs create detection pool and wall homographies
    Args:
        common_wall_ids (list): list of string wall IDs
        df2_walls_gs_gdf (pd.Geodataframe): 
        df5_sv (pd.DataFrame): _description_
        df5_oblique (pd.DataFrame): _description_
    Returns:
        list(): _description_
    """
    print("Creating detection pool and homographies for all wall IDs.")
    all_homographies = dict()
    all_detection_pool = dict()

    for wall_id in common_wall_ids:        
        normalized_normal_3d = get_normalized_normal_vector_3D(wall_id, df2_walls_gs_gdf)

        for df, detections_dir, src in [(df5_sv, CONF.output_streetview_detections_dir, "sv"), (df5_oblique, CONF.output_oblique_detections_dir, "oblique")]:
            if df.empty:
                continue
            df_detections, df_homographies = find_detections(wall_id, df, normalized_normal_3d, detections_dir, src)
            all_homographies.update(df_homographies)
            if wall_id in all_detection_pool.keys():
                all_detection_pool[wall_id].extend(df_detections)
            else:
                all_detection_pool[wall_id] = df_detections
    return all_detection_pool, all_homographies


def get_all_wall_ids(df5_sv, df5_oblique):
    """Get all unique wall IDs from both DataFrames

    Args:
        df5_sv (pd.DataFrame): DataFrame for street view.
        df5_oblique (pd.DataFrame): DataFrame for oblique images.
    Returns:
        set: Set of common wall IDs.
    """
    print("Finding common wall IDs between Street View and Oblique datasets.")
    df5_sv_ids = set(df5_sv['wall_id'].unique())
    df5_oblique_ids = set(df5_oblique['wall_id'].unique())
    #common_wall_ids = df5_sv_ids.intersection(df5_oblique_ids)
    common_wall_ids = df5_sv_ids.union(df5_oblique_ids)
    return common_wall_ids


def cluster_detections(det_pool, eps, min_samples=1):
    """creates centroids for the detections in det_pool and clusters them using DBSCAN

    Args:
        det_pool (list):  polygon list
        eps (int): param for DBSCAN
        min_samples (int, optional): min number of samples Defaults to 1.
    """
    def polygon_centroid(polygon_3D):
        pts = np.stack(polygon_3D)
        return pts.mean(axis=0)

    # Convert all detection polygons
    centroids = [polygon_centroid(poly) for poly in det_pool]
    centroids = np.stack(centroids)  # shape: (num_detections, 3)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centroids)  #eps → maximum distance to consider points as neighbors; in the same units as data, so the scale matters a lot. (px vs metric coords!!)
    labels = clustering.labels_  # array of cluster labels
    num_clusters = len(set(labels))

    clusters = defaultdict(list)
    for poly, label in zip(det_pool, labels):
        if label != -1:
            clusters[label].append(poly)
            
    return num_clusters, clusters, labels, centroids


def fuse_cluster_bboxes_median(clusters, filter=True):

    """
    Fuse clusters using the median of the coordinates (x1, y1, x2, y2).
    This preserves the structural integrity of the box better than point-cloud percentiles.
    """
    fused_boxes = []

    for cluster_id, cluster_polies in clusters.items():
        if filter and len(cluster_polies) <= 1:
            continue
        
        # Convert list of polygons to array of shape (N, 4) -> [x_min, y_min, x_max, y_max]
        # Assuming input poly is [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)]
        coords = []
        for poly in cluster_polies:
            pts = np.array(poly[:-1])
            x_min, x_max = np.min(pts[:, 0]), np.max(pts[:, 0])
            y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
            coords.append([x_min, y_min, x_max, y_max])
        
        coords = np.array(coords)

        # Calculate median of each coordinate component
        # usage of median ignores outliers (e.g. one box that is way too large)
        fused_coords = np.median(coords, axis=0) 
        
        x_min, y_min, x_max, y_max = fused_coords

        # Reconstruct polygon structure
        fused_box = [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
            (x_min, y_min)
        ]
        fused_boxes.append(fused_box)

    return fused_boxes



def align_boxes_grid(boxes, eps=0.1):
    """
    Align x and y coordinates of boxes that are roughly in the same column/row.

    Parameters:
    - boxes: list of boxes, each box is [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)]
    - eps: clustering tolerance (in same units as your coordinates)

    Returns:
    - aligned_boxes: list of boxes with x/y values snapped to median of clusters
    """
    # Collect all x and y values
    xs = []
    ys = []
    for box in boxes:
        box_arr = np.array(box[:-1])
        xs.extend(box_arr[:,0])
        ys.extend(box_arr[:,1])
    xs = np.array(xs).reshape(-1,1)
    ys = np.array(ys).reshape(-1,1)

    # Cluster x-values (columns)
    x_clustering = DBSCAN(eps=eps, min_samples=1).fit(xs)
    x_labels = x_clustering.labels_
    x_medians = {}
    for label in set(x_labels):
        x_medians[label] = np.median(xs[x_labels==label])

    # Cluster y-values (rows)
    y_clustering = DBSCAN(eps=eps, min_samples=1).fit(ys)
    y_labels = y_clustering.labels_
    y_medians = {}
    for label in set(y_labels):
        y_medians[label] = np.median(ys[y_labels==label])

    # Replace coordinates by cluster median
    aligned_boxes = []
    for box in boxes:
        box_arr = np.array(box[:-1])

        snapped_x = []
        snapped_y = []
        for x, y in box_arr:
            x_label = x_labels[np.argmin(np.abs(xs.flatten() - x))]
            y_label = y_labels[np.argmin(np.abs(ys.flatten() - y))]
            snapped_x.append(x_medians[x_label])
            snapped_y.append(y_medians[y_label])

        # 🔹 ONLY CHANGE: enforce rectangle format
        x_min = min(snapped_x)
        x_max = max(snapped_x)
        y_min = min(snapped_y)
        y_max = max(snapped_y)

        new_box = [
            (float(x_min), float(y_min)),
            (float(x_max), float(y_min)),
            (float(x_max), float(y_max)),
            (float(x_min), float(y_max)),
            (float(x_min), float(y_min))
        ]

        aligned_boxes.append(new_box)

    return aligned_boxes