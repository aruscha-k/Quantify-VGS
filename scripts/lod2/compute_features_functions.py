import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.affinity import translate, scale
import os
import logging
logger = logging.getLogger(__name__)

from scripts.lod2.compute_features_wall_join import make_wall_2d, make_wall_3d, get_angle_0_to_180
import scripts.config as CONF


def make_geodataframe(df, geometry_column):
    df['ground_surface'] = df['ground_surface'].apply(lambda x: Polygon(x))
    gdf = gpd.GeoDataFrame(df, geometry=geometry_column)
    return gdf

def read_original_lod_data(filepath):
    logger.info(f"Reading original LoD2 data from {filepath}")
    original_df = pd.read_pickle(filepath)
    original_df['ground_surface'] = original_df['ground_surface'].apply(lambda x:Polygon(x))
    original_gdf = gpd.GeoDataFrame(original_df, geometry="ground_surface")
    return original_gdf


def read_data(df1_filename, df2_filename):
    logger.info(f"Reading dataframes {df1_filename} and {df2_filename} and making merged geodataframe")
    df1_parts = pd.read_pickle(os.path.join(CONF.dataframes_dir, df1_filename))
    df2_walls = pd.read_pickle(os.path.join(CONF.dataframes_dir, df2_filename))

    # change type: make lists out of strings in surface columns ONLY NEEDED WHEN SAVED AS CSV
    # df1_parts["ground_surface"] = df1_parts["ground_surface"].apply(ast.literal_eval)
    # df2_walls["surface_coordinates"] = df2_walls["surface_coordinates"].apply(ast.literal_eval)

    df2_walls_with_gs = pd.merge(
        df2_walls,
        df1_parts[["building_id", "ground_surface"]],
        on="building_id",
        how="left"
    )
    print(df2_walls_with_gs.columns)
    print(df2_walls_with_gs.iloc[[1]])

    df2_walls_gs_gdf = make_geodataframe(df2_walls_with_gs.iloc[[1]], "ground_surface")
    
    return df2_walls_gs_gdf #df1_parts, df2_walls


def calculate_wall_area(df, min_wall_area_thresh, filter_walls):
    """Calculates the area of a wall, adds column area

    Args:
        df (pd.df): needs columns surface_coordinates
        min_wall_area_thresh (float): area that a wall should at least have
        filter_walls (bool): if walls under threshold should be deleted or kept

    Returns:
        df: pd.df with new column area
    """
    logger.info(f"Calculating wall area with min_wall_area_thresh={min_wall_area_thresh}, filter_walls={filter_walls}")
    df = df.copy()
    areas = []
    # for each wall, extract coordinates and use cross product method to calculate area with 3d coordinates
    for index, row in df.iterrows():
        # get coords, but ignore the last coordinate tuple as it is the same as the first one and not needed here
        coords = np.array(row["surface_coordinates"])[:-1]
        area = 0
        # use cross product method to calculate area (the area calculation of shapely does not deal well with 3d coords)
        try:
            for i in range(1, len(coords)-1):
                # always computes the area of the triangle between coords[0], coords[i] and coords[i+1]
                v1 = coords[i] - coords[0]
                v2 = coords[i+1] - coords[0]
                cross_prod = np.cross(v1,v2)
                area += np.linalg.norm(cross_prod) / 2
            # save calculated area in according row
        except Exception:
            pass
        areas.append(area)

    df['area'] = areas
    if filter_walls:
        # delete all walls with less than min_wall_area_thresh area
        before = len(df)
        removed = df[df["area"] <= min_wall_area_thresh]["wall_id"].tolist()
        df = df[df["area"] > min_wall_area_thresh].reset_index(drop = True)
        logger.info(f"{len(df)}, after deleting walls with area below {min_wall_area_thresh}; {before - len(df)} walls removed")
        if removed:
            logger.debug("Filtered-out wall_ids:\n"+ "\n".join(map(str, removed)))
    return df


def validate_outward_orientation(wall_points: np.ndarray, wall_normal_vector: np.ndarray, groundsurface_midpoint: Point):
    """
    Validate whether wall normal points outward (away from center of building footprint).
    
    wall_points: np.ndarray of shape (n,3) for the wall vertices
    wall_normal_vector: np.ndarray (3,) wall normal vector from cross product
    groundsurface_midpoint: shapely Point (2D centroid of ground surface)
    """
    # Compute centroid of the wall in 3D
    wall_centroid = np.mean(wall_points, axis=0)

    # Convert ground surface midpoint (shapely Point) to 3D (z=0)
    ground_centroid = np.array([groundsurface_midpoint.x, groundsurface_midpoint.y, 0.0])

    # Vector from wall centroid to ground centroid
    vec_to_ground = ground_centroid - wall_centroid

    # Dot product with wall normal
    dot = np.dot(vec_to_ground, wall_normal_vector)

    return dot < 0   # True if normal points outward


def get_normal_vector(wall_points: np.ndarray, ground_surface_midpoint: Point):
    """
    Calculate the normal vector of a surface (wall) and validate its pointing towards the outside of the building by checking with the groundsurface midpoint (which lies inside the building)
    wall_points: list of [x,y,z] coords for the wall surface
    ground_surface: list of [x,y,z] coords for ground surface

    RETURNS:
    normal_vector (in 2D) and normal vector in 3D
    """ 
    
    wall = np.array(wall_points)
    if wall.shape[0] < 3:
        return 0, None   # not enough points to define a plane

    p1, p2, p3 = wall[:3, :3]  # three 3D points

    # --- STEP 2: Compute normal via cross product
    v1 = p2 - p1
    v2 = p3 - p1
    normal_vector_3d = np.cross(v1, v2)
  # Project to XY-plane (azimuth only cares about x,y)
    #normal_vector = normal_vector_3d[:2]

    if not validate_outward_orientation(wall[:3, :3], normal_vector_3d, ground_surface_midpoint):
        #print("[!] Check orientation of wall vectors - pointing inward! - flipping! building: ", b_id)
        normal_vector_3d = -normal_vector_3d  # flip if pointing inward
        #normal_vector = normal_vector_3d[:2]

    if isinstance(normal_vector_3d, np.ndarray) and normal_vector_3d.shape == (3,):
        return normal_vector_3d
    else:
        return np.array([0,0,0])

def calc_normal(df):
    logger.info("Calculating normal vectors for walls")
    df = df.copy()
    normal_vecs = []
    for row in df.itertuples(index=False):
        norm_3d = get_normal_vector(
            row.surface_coordinates, 
            row.ground_surface.centroid
        )
        normal_vecs.append(norm_3d)

    df["normal_vector_3d"] = normal_vecs
    return df


def check_verticality_of_walls(df, filter_walls:bool):
    logger.info(f"Finding ground bound walls, filter_walls={filter_walls}")
    df = df.copy()
    verticals = []
    for index, row in df.iterrows():
        vertical = False
        normal_vector = row['normal_vector_3d']
        if np.linalg.norm(normal_vector) != 0:
            normalized_normal_vector = normal_vector / np.linalg.norm(normal_vector)
        # check whether in will be considered as vertical: for 1m long normal vector, if it goes up or down more than 5cm its not vertical
        if np.abs(normalized_normal_vector[2]) <= 0.05:
            vertical = True
        verticals.append(vertical)
    
    df['vertical'] = verticals

    if filter_walls:
        before = len(df)
        removed = df.loc[~df["vertical"], "wall_id"].tolist()
        df = df[df["vertical"]].reset_index(drop = True)
        logger.info(f"{len(df)} after deleting non-vertical walls; {before - len(df)} walls removed")
        if removed:
            logger.debug("Non-vertical wall_ids removed:\n" +"\n".join(map(str, removed)))
    df = df.drop("vertical", axis = 1)
    return df


def at_least_two_on_ground(wall_coords, ground_coords):
    ''' 
    for a list of coordinates (belonging to a wall) and a minimum and maximum value (representing the ground values, extracted from ground_coords) this function checks whether to at least two coordinates of the wall are in the range from minimum to maximum value (ergo on the ground)

    PARAMS:
    * wall_coords (list of 3d tuples): wall coordinate tuples which are checked whether two are on ground
    * ground_coords (list of 3d tuples): ground surface coordinates, used to get minimum and maximum z value of ground

    RETURNS:
    * boolean: True if at least two wall coordinates are on ground, else False
    '''
    ground_min = np.min(np.array(ground_coords)[:,2]) - 0.1
    ground_max = np.max(np.array(ground_coords)[:,2]) + 0.1
    count_on_ground = 0
    for corner in wall_coords[:-1]:
        if corner[2] < ground_max and corner[2] > ground_min:
            count_on_ground += 1
            if count_on_ground == 2:
                return True
        
    return False

def find_ground_bound(df, filter_walls):
    logger.info(f"Finding ground bound walls, filter function: {filter_walls}")
    df = df.copy()
    ground_bounds = []
    # check if wall is ground-bound or not (if not, no green facades possible or at least difficult)
    for index, row in df.iterrows():
        ground_bound = False
        # try:
        ground_surface = row["ground_surface"].exterior.coords
        wall_surface = row["surface_coordinates"]
        # check if at least two of the corner of wall_surface are in the ground surface, and if yes, it is ground-based
        ground_bound = at_least_two_on_ground(wall_surface, ground_surface)
        # except Exception as e:
        #     print(e)
        ground_bounds.append(ground_bound)
    df['ground_bound'] = ground_bounds

    if filter_walls:
        removed = df.loc[~df["ground_bound"], "wall_id"].tolist()
        df = df[df["ground_bound"]].reset_index(drop = True)
        df = df.drop("ground_bound", axis = 1)
        if removed:
            logger.debug("Non-ground-bound wall_ids removed:\n" +"\n".join(map(str, removed)))
    return df


def calculate_wall_height_and_width(df):
    """Calculates the width of a wall (not thickness) using the min/max x coordinates
    Args:
        df (pd df): needs column surface coordinates as list of coords (not shapely)
    Returns:
        df: df with widths
    """
    logger.info("Calculating wall height and width")
    df = df.copy()
    heights, widths =[], []
    for _, row in df.iterrows():
        try:
            wall_surface = np.array(row["surface_coordinates"])
            height = np.max(wall_surface[:,2]) - np.min(wall_surface[:,2]) #max_z - min_z
            
            xs = wall_surface[:, 0]
            width = float(np.max(xs) - np.min(xs)) #max_x - min_x
        except Exception:
            height, width = np.nan, np.nan
        heights.append(height)
        widths.append(width)
    df['wall_height'] = heights
    df['wall_width'] = widths
    return df


def calculate_wall_orientation(wall_points:list, normal_vector_3d:np.array, show_plot:bool): #
    """
    Calculate azimuth/orientation of a wall surface (triangle, quad, or polygon).
    wall_points: list of [x,y,z] coords for the wall surface
    ground_surface: list of [x,y,z] coords for ground surface
    show_plot: bool, whether to plot
    """
    
    # # ground_surface_poly = Polygon(ground_surface)
    # # ground_surface_midpoint = ground_surface_poly.centroid
    # normal_vector_3d = get_normal_vector(wall_points, ground_surface_centroid)
    # Project to XY-plane (azimuth only cares about x,y)
    normal_vector_2d = normal_vector_3d[:2]#
    
    # Compare with north vector # Define north direction as the positive Y-axis (0, 1)
    north_vector = np.array([0, 50])
    # Calculate the angle between the normal vector and north direction (dot product)
    dot_product = np.dot(north_vector, normal_vector_2d)
     # Compute the magnitudes
    magnitude_normal = np.linalg.norm(normal_vector_2d)
    magnitude_north = np.linalg.norm(north_vector)

    # Compute the cosine of the angle - Normalize
    cos_theta = dot_product / (magnitude_normal * magnitude_north)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical safety

    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)

    # Determine CW/CCW using 2D cross product
    cross_product = north_vector[0] * normal_vector_2d[1] - north_vector[1] * normal_vector_2d[0]
    #print("angle deg:",(angle_deg))

    if cross_product > 0:
        #print('counterclockwise')
        if angle_deg < 22.5:
            orientation = "N"
        elif 22.5 <= angle_deg < 67.5:
            orientation = "NW"
        elif 67.5 <= angle_deg < 112.5:
            orientation = "W"
        elif 112.5 <= angle_deg < 157.5:
            orientation = "SW"
        elif 157.5 <= angle_deg <= 180:
            orientation = "S"

    elif cross_product < 0:
        #print('clockwise')
        if angle_deg < 22.5:
            orientation = "N"
        elif 22.5 <= angle_deg < 67.5:
            orientation = "NE"
        elif 67.5 <= angle_deg < 112.5:
            orientation = "E"
        elif 112.5 <= angle_deg < 157.5:
            orientation = "SE"
        elif 157.5 <= angle_deg <= 180:
            orientation = "S"
    else:
        # cross_product == 0 -> exactly aligned with north vector
        orientation = "N" if dot_product > 0 else "S"

    # if show_plot:
    #     print(orientation)
    #     plot_wall_orientation(wall_points, ground_surface_poly, normal_vector_2d, north_vector)

    return angle_deg, orientation


#chat gpt improvement
# # Adjust angle so that 0° = North, 90° = East
#     if cross < 0:  # clockwise
#         orientation = _angle_to_compass(angle_deg)
#     else:  # counter-clockwise
#         orientation = _angle_to_compass(-angle_deg)

# # --- 2. Compass mapping helper ---
# def _angle_to_compass(angle: float) -> str:
#     """Map a signed angle to one of the 8 cardinal directions."""
#     angle = (angle + 360) % 360
#     directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
#     idx = int((angle + 22.5) // 45) % 8
#     return directions[idx]


def get_wall_orientation(df):
    logger.info("Calculating wall orientation")
    df = df.copy()
    angles, orientations = [], []
    for row in df.itertuples(index=False):
        ang, ori = calculate_wall_orientation(
            row.surface_coordinates, 
            row.normal_vector_3d,
            show_plot=False
        )
        angles.append(ang)
        orientations.append(ori)

    df["angle_deg"] = angles
    df["orientation"] = orientations

    return df


def get_sorted_bounding_box(bounding_box, normal, ground_surface_midpt):
    '''
    Sorts the corners of a bounding box based on their positions relative to a specified normal vector: they should be ordered in a counterclockwise manner when viewed from the exterior, starting from the upper right corner

    PARAMS:
    * bounding_box (list of 3d tuples): A list containing five tuples, where each tuple represents a 3D coordinate (x, y, z) of the corners of the bounding box.
    * normal (3d tuple): A tuple representing the normal vector (a, b, c) that defines the orientation of the original wall.
    * ground_surface_midpt (Point): midpoint of the groundsurface of the building to calculate direction of normal

    RETURNS:
    * list: A reordered list of the bounding box corners in counterclockwise order if successful, or an empty list if the input bounding box is invalid or if the normal vector is not valid.
    '''
    # check: bounding box has 5 coordinates
    if not len(bounding_box) == 5:
        return []

    # get center
    center_point = np.mean(bounding_box, axis = 0)

    # get normal of bounding box without order: if it is similar to given normal, counterclockwise direction holds
    check_normal_3D = get_normal_vector(bounding_box, ground_surface_midpt)

    if np.linalg.norm(check_normal_3D) == 0:
        return []
    # if check_normal goes into same direction as original normal, direction is already counterclockwise (seen from exterior)
    # if not, change direction
    if get_angle_0_to_180(check_normal_3D, normal) > 90:
        bounding_box = bounding_box[::-1]

    #now we just need to start with the two upper ones that follow each other
    index_to_start_with = 0
    for corner_ind in range(len(bounding_box)-1):
        if (bounding_box[corner_ind][2] > center_point[2]) and (bounding_box[corner_ind+1][2] > center_point[2]):
            index_to_start_with = corner_ind
    bounding_box_new = bounding_box[index_to_start_with:-1] + bounding_box[:index_to_start_with+1]
    return bounding_box_new


# OLD replace by chat GPT function below
# def get_bounding_box_from_single_surface(wall_id, surface, normal_vector_3d, ground_surface_midpt):
#     '''
#     Computes the 3D bounding box of a single surface defined by its coordinates. It projects the surface onto a 2D plane, calculates the 2D bounding box, and then transforms this bounding box back into 3D space.

#     PARAMS:
#     * surface (list of tuple): A list of tuples representing the coordinates of the surface in 3D space.
#     * normal (tuple): A tuple representing the normal vector (a, b, c) of the plane in which the surface is located.

#     RETURNS:
#     * list: A list of tuples representing the coordinates of the 3D bounding box of the surface, ordered such that it starts with the upper right corner, then upper left, lower left, and lower right (as viewed from the exterior of the wall).
#     '''
#     # get plane equation ax+by+cz+d=0 for plane in which wall is
#     try:
#         normalized_normal = normal_vector_3d/np.linalg.norm(normal_vector_3d)
#         a,b,c = normalized_normal
#         d = - (a * surface[0][0] + b* surface[0][1] + c* surface[0][2])

#         # get two vectors, both orthonormal to the normal and to each other, to span the plane
#         v1 = np.array(surface[0]) - np.array(surface[1])
#         v1 = v1/np.linalg.norm(v1)
#         v2 = np.cross(normalized_normal, v1)                              # FIX: use unit normal here
#         #v2 = v2/np.linalg.norm(v2)                                # FIX: normalize v2

#         # make a 2d polygon out of surface with the help of v1 and v2 in the plane
#         surface_2d = Polygon(make_wall_2d(surface, v1, v2)).buffer(0)
        
#         # get 2d bounding box with shapely (3d does not work) and export coordinates
#         if surface_2d.is_empty or surface_2d.area == 0:
#             return []
#         bounding_box = Polygon(surface_2d).oriented_envelope

#         # get back to 3d again, first need origin in plane described by a,b,c,d and then return
#         distance_origin_to_plane = (a * 0 + b * 0 + c * 0 + d) / np.sqrt(a**2 + b**2 + c**2)
#         # Project the point onto the plane
#         origin_on_plane = np.array([0,0,0], dtype=float) - distance_origin_to_plane * normalized_normal
    
#         bounding_box_3d = make_wall_3d(list(bounding_box.exterior.coords), v1, v2, origin_on_plane)

#         # sort bounding box such that it starts with upper right corner, then upper left (seen from exterior of wall), then lower left, lower right
#         bounding_box_3d = get_sorted_bounding_box(bounding_box_3d, tuple(normalized_normal), ground_surface_midpt)
            
#         return bounding_box_3d
#     except Exception as e:
#         print(wall_id, e)
#         return []


def get_bounding_box_from_single_surface(wall_id, surface, normal_vector_3d, ground_surface_midpt):
    """
    Compute a stable 3D outside bounding box (full façade rectangle) for a wall.

    Parameters
    ----------
    surface : iterable of (x, y, z)
        3D coordinates of the wall polygon (can be non-rectangular, slightly non-planar).
    normal_vector_3d : array-like
        Approximate wall normal (from your LoD process), pointing outward.
    ground_surface_midpt : shapely Point
        Used only to keep orientation consistent with your existing code.

    Returns
    -------
    list[tuple]
        5 points (closed ring) of the 3D façade rectangle, ordered and oriented
        the same way as your old outside_bounding_box.
    """
    try:
        pts = np.asarray(surface, dtype=float)
        if pts.shape[0] < 3:
            return []

        # 1) Normalize normal
        n = np.asarray(normal_vector_3d, dtype=float)
        if np.allclose(n, 0):
            return []
        n = n / np.linalg.norm(n)

        # 2) Define a *stable* orthonormal basis (v1, v2) in the wall plane
        # choose a reference not parallel to n
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(ref, n)) > 0.9:   # wall normal almost vertical -> use X-axis instead
            ref = np.array([1.0, 0.0, 0.0])

        v1 = np.cross(ref, n)
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(n, v1)
        v2 = v2 / np.linalg.norm(v2)

        # 3) Use wall centroid as origin in the plane
        centroid = pts.mean(axis=0)

        # 4) Project all wall points into 2D plane coords (u, v)
        uv = []
        for p in pts:
            d = p - centroid
            u = np.dot(d, v1)
            v = np.dot(d, v2)
            uv.append((u, v))

        poly_2d = Polygon(uv).buffer(0)
        if poly_2d.is_empty or poly_2d.area == 0:
            return []

        # 5) Oriented minimum bounding rectangle in the wall plane
        rect_2d = poly_2d.minimum_rotated_rectangle
        rect_uv = list(rect_2d.exterior.coords)[:-1]  # 4 points

        # 6) Map 2D rectangle corners back to 3D
        bbox3d = []
        for u, v in rect_uv:
            p3d = centroid + u * v1 + v * v2
            bbox3d.append(tuple(p3d))
        bbox3d.append(bbox3d[0])  # close ring

        # 7) Reuse your existing sorting/orientation logic
        bbox3d_sorted = get_sorted_bounding_box(bbox3d, n, ground_surface_midpt)

        return bbox3d_sorted
    except Exception as e:
        print(wall_id, e)
        return []



def check_outside_exposure(self_building_id, wall_normal, ground_surface, original_lod_gdf, buffer_radius:int=5):
    """Checks for a wall if it is exposed to the outside using the original lod data (since in the processed data not all walls that acutally exists will be present anymore)
    Args:
        self_building_id (str): building id of the wall to check to remove from possible list
        wall_normal (np.array): normal vector
        ground_surface (list or geometry): ground surface of the building belonging to the wall
        original_lod_df (pandas gdf): gdf of the original lod data 
        buffer_radius (int, optional): radius to look for buildings in. Defaults to 5.
    Returns:
        bool: if exposed to the outside
    """
    if np.array_equal(wall_normal, np.array([0, 0, 0])):
        return False
    wall_normal_normalized = wall_normal / np.linalg.norm(wall_normal)
        
    buffer_area = Polygon([(p.x, p.y) for p in [translate(Point(x, y), xoff=wall_normal_normalized[0]*(buffer_radius-1), yoff=wall_normal_normalized[1]*(buffer_radius-1)) for (x, y, z) in ground_surface]])
    buffer_area = scale(buffer_area, xfact=0.7, yfact=0.7, origin='center')
    possible = original_lod_gdf.sindex.query(buffer_area, predicate="intersects")
    nearby_walls_df = original_lod_gdf.iloc[possible]
    nearby_walls_df = nearby_walls_df.loc[nearby_walls_df['building_id'] != self_building_id]
    #print("the following buldings are nearby", nearby_walls_df['building_id'].unique().tolist())

    return nearby_walls_df.empty # if empty no walls in front of wall -> True


def calculate_outside_factors(df, original_df):
    """Calculate the surrounding boundbox of walls that have more than four coordinate points and check wheter a wall is exposed to the outside or hid behind a wall
    Args:
        df (pandas df):
    Returns:
        pandas d: with new columns outside_bounding_box and outside (boolean)
    """
    logger.info("Calculate outside factors")
    df = df.copy()
    df['surface_coordinates'] = df['surface_coordinates'].apply(lambda x: Polygon(x))

    outside_bounding_boxes, outsides = [], []
    for row in df.itertuples(index=False):
        outside_bbox = get_bounding_box_from_single_surface(
            row.wall_id, 
            row.surface_coordinates.exterior.coords, 
            row.normal_vector_3d,
            row.ground_surface.centroid
        )
        outside = check_outside_exposure(
            row.building_id,
            row.normal_vector_3d,
            row.ground_surface.exterior.coords,
            original_df
        )

        outside_bounding_boxes.append(outside_bbox)
        outsides.append(outside)

    df["outside_bounding_box"] = outside_bounding_boxes
    df["outside"] = outsides

    return df


#-----------------------------
def validate_df2_surface_coordinates(surface_coordinates):
    """
    Validates that each element in surface_coordinates is a list of (float, float, float) tuples.
    If not, tries to coerce them into the right format.
    Args:
        surface_coordinates (pd DF column): list of surface coords
    """
    validated = []

    for idx, val in surface_coordinates.items():
        if isinstance(val, list):
            new_list = []
            for elem in val:
                # must be iterable of length 3
                if isinstance(elem, (list, tuple)) and len(elem) == 3:
                    try:
                        new_list.append(tuple(float(x) for x in elem))
                    except Exception:
                        pass
            validated.append(new_list)
        else:
            validated.append([])  # fallback: empty list

    return pd.Series(validated, index=surface_coordinates.index)