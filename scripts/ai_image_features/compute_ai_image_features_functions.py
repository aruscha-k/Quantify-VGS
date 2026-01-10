from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import pandas as pd
import os


def preprocess_detections(detections_df, wall_area_image, prediction_confidence_threshold):  
    #if df empty or df doesnt contain any window/door detections above confidence threshold, skip
    if detections_df.empty:
        debug_message = f"No detections for Building ID. Skipping."
        return None, debug_message   

    detections_df = detections_df[detections_df['confidence'] >= prediction_confidence_threshold]   #filter by confidence
    detections_df = remove_detections_overlapping_roof(detections_df, roof_label="roof", threshold=0.5) 

    if not check_vegetation_ratio(detections_df, wall_area_image, vegetation_label="vegetation", threshold=0.5):
        debug_message = f"Vegetation ratio too high. Skipping."
        return None, debug_message
    
    return detections_df, ""


def check_vegetation_ratio(detections_df, wall_area_px, vegetation_label="vegetation", threshold=0.5):
    ''' 
    Checks if the ratio of vegetation in the image exceeds a given threshold.
    PARAMS:
    * detections_df: DataFrame containing detection results with 'class_name' and 'polygon' columns.
    * wall_area_px: Total area of the wall in pixel space.
    * vegetation_label: The label used to identify vegetation in the detections.
    * threshold: The maximum allowed ratio of vegetation area to total detected area.
    RESULTS:
    * Boolean indicating whether the vegetation ratio is below the threshold.
    '''
    veg_detections = detections_df[detections_df['class_name'] == vegetation_label].copy()
    if veg_detections.empty:
        return True,  # no vegetation detected   
    veg_detections['polygon'] = veg_detections['polygon'].apply(lambda x: Polygon(x) if not isinstance(x, Polygon) else x)
    total_veg_area = veg_detections['polygon'].apply(lambda p: p.area).sum()   # compute total vegetation area
    veg_ratio = total_veg_area / wall_area_px if wall_area_px > 0 else 0    
    return veg_ratio < threshold


def remove_detections_overlapping_roof(detections_df, roof_label="roof", threshold=0.5):
    temp_detections_df = detections_df.copy()
    # make polygon a polygon column
    temp_detections_df['polygon'] = temp_detections_df['polygon'].apply(lambda x: Polygon(x) if not isinstance(x, Polygon) else x)
    
    roof_rows = temp_detections_df[temp_detections_df['class_name'] == roof_label]
    if roof_rows.empty:
        return detections_df
    # If multiple roof detections, union them
    roof_poly = unary_union(roof_rows['polygon'].tolist())

    # --- Compute bool mask for df ---
    keep_mask = []
    for idx, row in temp_detections_df.iterrows():
        det_poly = row['polygon']
        # Keep roof as is 
        if row['class_name'] == roof_label:
            keep_mask.append(True)
            continue

        # Compute overlap
        intersection_area = det_poly.intersection(roof_poly).area
        area = det_poly.area if det_poly.area > 0 else 0
        overlap_ratio = intersection_area / area if area > 0 else 0

        # Keep only detections with < threshold overlap
        keep_mask.append(overlap_ratio < threshold)

    return detections_df[keep_mask].copy()


def remove_roof_from_surface_area(surface_coords, detections_df, roof_label="roof"):
    ''' 
    Removes roof area from the given surface coordinates based on roof detections.
    PARAMS:
    * surface_coords: List of (x, y) tuples representing the coordinates of the wall surface in pixel space.
    * detections_df: DataFrame containing detection results with 'class_name' and 'polygon' columns.
    * roof_label: The label used to identify roof in the detections.
    RESULTS:
    * Area of the wall surface excluding roof area.
    '''
    total_area_polygon = Polygon(surface_coords).buffer(0)
    roof_rows = detections_df[detections_df['class_name'] == roof_label]
    if roof_rows.empty:
        return surface_coords  # no roof detected

    roof_polygons = [Polygon(roof_rows.iloc[i]['polygon']).buffer(0) for i in range(len(roof_rows))]
    if len(roof_polygons) == 1:
        roof_poly = roof_polygons[0]
    else:
        roof_poly = MultiPolygon(roof_polygons).buffer(0) #FIXME!: check if this works correctly with multiple roof polygons

    solid_wall_polygon = total_area_polygon.difference(roof_poly)
    return solid_wall_polygon.exterior.coords


def calculate_wwr(detections, outside_surface_px):
    ''' 
    Calculates the Window-to-Wall Ratio (WWR) for a given image of a building facade.
    Checks, if detected windows are within the outside surface area
    This function reads a detections for rectified image and its corresponding outside surface coordinates, 
    and computes the WWR by comparing the area of the detected windows to the total area of the wall surface.

    PARAMS:
    * detections: List of bounding boxes representing detected windows on the wall surface. Each bounding box is defined by its corner coordinates.
    * outside_surface_px: List of (x, y) tuples representing the coordinates of the outside wall surface in pixel space.

    RESULTS:
    * Window-to-Wall Ratio (WWR) as a float, rounded to four decimal places.
    '''

    # make outside surface to polygon
    total_area_polygon = Polygon(outside_surface_px).buffer(0)

    # remove detections that are outside of the total area polygon and create multipolygon from windows
    windows_multi_polygon = []
    for box in detections:
        box_polygon = Polygon(box).buffer(0)
        if total_area_polygon.contains(box_polygon):
            windows_multi_polygon.append(box_polygon)

    windows_multi_polygon = MultiPolygon(windows_multi_polygon).buffer(0)

    # take difference of outside surface with window polygon
    solid_wall_polygon = total_area_polygon.difference(windows_multi_polygon) 

    # return Window-to-wall ratio
    return round(1 - (solid_wall_polygon.area / total_area_polygon.area), 4)


def calculate_swa(wwr, outside_area):
    '''
    Calculates the Solid Wall Area (SWA) based on the Window-to-Wall Ratio (WWR) and the outside wall area.
    PARAMS:
    * wwr: Window-to-Wall Ratio as a float.
    * outside_area: Total area of the outside wall surface as a float in meter squared.
    RESULTS:
    * Solid Wall Area (SWA) as a float.
    '''
    # solid wall area: outside area * (1-wwr)
    return (1 - wwr) * outside_area


def calculate_averages_per_wall(df5: pd.DataFrame) -> pd.DataFrame:
    """takes all wwr / swa values of the same wall (groupby wall_id) and calculates an average wwr / swa for each wall
    Args:
        df5 (pd.DataFrame): 

    Returns:
        pd.DataFrame: dataframe with building_id, wall_id, average_wwr_oblique, average_swa_oblique
    """
    agg_dict = {
    "building_id": ("building_id", "first"),
    "average_wwr": ("wwr", "mean"),
    "average_swa": ("swa", "mean")
}

    if "fusion_source" in df5.columns:
        agg_dict["fusion_source"] = ("fusion_source", lambda x: "+".join(sorted(set(x))))

    grouped = df5.groupby("wall_id").agg(**agg_dict).reset_index()
    return grouped