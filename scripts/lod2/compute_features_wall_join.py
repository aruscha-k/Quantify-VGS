import numpy as np
import pandas as pd
from itertools import combinations
from typing import Tuple, List
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
import shapely
from tqdm import tqdm


import scripts.config as CONF
import logging
logger = logging.getLogger(__name__)


# REFACTOR:
def get_angle_0_to_180(vec1: Tuple[float, ...], vec2: Tuple[float, ...]) -> float:
    """
    Compute the angle (0–180°) between two 2D or 3D vectors.
    """
    v1, v2 = np.asarray(vec1), np.asarray(vec2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0

    cos_theta = np.clip(np.dot(v1, v2) / denom, -1, 1)
    return np.degrees(np.arccos(cos_theta))



def check_if_same_line(corner1, corner2, corner3):
    '''
    Takes three coordinates tuples and checks whether they lie on the same 3D line and where the corner3 lies in respect to the others

    PARAMS:
    * corner1, corner2 (3d tuples): two consecutive coordinate tuples of first wall
    * corner3 (3d tuple): coordinate tuples of second wall

    RETURNS:
    * whether all three corners lie on the same 3D line (boolean)
    * scaling_factor: corner3 - corner1 / corner2 - corner1. If the factor is positive but below 1, corner3 is between both other corners on the line  corner1--corner3--corner2. If it is negative, corner3 is before corner1 corner3--corner1--corner2. Above 1: corner1--corner2--corner3
    '''
    if ((corner2[1] - corner1[1]) * (corner3[0] - corner1[0])) == ((corner3[1] - corner1[1]) * (corner2[0] - corner1[0])):
        if ((corner2[2] - corner1[2]) * (corner3[0] - corner1[0])) == ((corner3[2] - corner1[2]) * (corner2[0] - corner1[0])):
            if (corner2[0] - corner1[0]) != 0:
                scaling_factor = (corner3[0] - corner1[0])/(corner2[0] - corner1[0])
            elif (corner2[1] - corner1[1]) != 0:
                scaling_factor = (corner3[1] - corner1[1])/(corner2[1] - corner1[1])
            elif (corner2[2] - corner1[2]) != 0:
                scaling_factor = (corner3[2] - corner1[2])/(corner2[2] - corner1[2])
            else:
                "corner1 and corner2 are the same!" # should normally not happen, were removed already
                return False, 0
            return True, scaling_factor
        
    return False, 0


def common_edge(wall1_coordinates: List[Tuple[float, float, float]],
                wall2_coordinates: List[Tuple[float, float, float]]) -> bool:
    """ 
    check for two walls whether they have a common edge 
    1. check whether both walls have consecutive corners in common (mostly the case)
    2. check if two consecutive corners of each wall lie on the same 3D line and the line segments intersect each other

    PARAMS:
    * wall1_coordinates: a list of tuples with (x,y,z)-coordinates, last one = first one, for first wall
    * wall2_coordinates: a list of tuples with (x,y,z)-coordinates, last one = first one, for second wall

    RETURNS:
    boolean, whether both walls have a common edge
    """
    wall1 = np.array(wall1_coordinates)
    wall2 = np.array(wall2_coordinates)

    # check whether both walls have two consecutive corners in common 
    for corner_index in range(len(wall1[:-1,:])):
        for corner_index2 in range(1,len(wall2[:,:])):
            if np.all(wall1[corner_index] == wall2[corner_index2]):
                if np.all(wall1[corner_index + 1] == wall2[corner_index2 - 1]):
                    return True

    # else check if one edge is partly in common (two points of each wall on the same 3d line in space and with an intersection in between)
    for corner_index in range(0,len(wall1[:-1,:])):
        for corner_index2 in range(1,len(wall2)):
            first_point_on_line, scaling_factor1 = check_if_same_line(wall1[corner_index], wall1[corner_index+1], wall2[corner_index2])
            if first_point_on_line:
                second_point_on_line, scaling_factor2 = check_if_same_line(wall1[corner_index], wall1[corner_index+1], wall2[corner_index2-1])
                if second_point_on_line:
                    # check whether one of the points from wall2 is between both points of wall1 (scaling factor between 0 and 1)
                    if 0 <= scaling_factor1 <= 1 or 0 <= scaling_factor2 <= 1:
                        return True
                    # if not, the possibility is still that wall1 is completely in between the two wall2 points
                    if (scaling_factor1 < 0 and scaling_factor2 > 0) or (scaling_factor1 > 0 and scaling_factor2 < 0):
                        return True
                    
    return False


# refactor
# def find_walls_to_join_common_edge(df) -> pd.DataFrame:
#     """
#     Identify walls belonging to the same part that share an edge
#     and have nearly identical normals (angle < 1°).

#     df: pandas df or gdf, needs columns normal_vector_3d, surface_coordinates, wall_id
#     """
#     logger.info("Finding walls to join")
#     def normalize_vector(v):
#         v = np.array(v, dtype=float)
#         norm = np.linalg.norm(v)
#         return list(v / norm) if norm != 0 else [0.0, 0.0, 0.0]
#     df = df.copy()
#     df["normalized_nv3d"] = df["normal_vector_3d"].apply(normalize_vector)

#     grouped = df.groupby("building_id").agg(
#         coordinates=('surface_coordinates', list),
#         normals=('normalized_nv3d', list),
#         wall_ids=('wall_id', list)
#     ).reset_index()

#     results = []
#     new_id_counter = 0

#     for _, part in grouped.iterrows():
#         building_id = part.building_id
#         normals, coords, wall_ids = part.normals, part.coordinates, part.wall_ids
#         joins = []  # list of tuples of wall_ids to join

#         # Compare each unique pair once
#         for (i, j) in combinations(range(len(normals)), 2):
#             if get_angle_0_to_180(normals[i][:2], normals[j][:2]) < 1:
#                 if common_edge(coords[i], coords[j]):
#                     joins.append((wall_ids[i], wall_ids[j]))

#         # Merge overlapping groups efficiently
#         groups = []
#         for a, b in joins:
#             added = False
#             for g in groups:
#                 if a in g or b in g:
#                     g.update([a, b])
#                     added = True
#                     break
#             if not added:
#                 groups.append(set([a, b]))

#         for g in groups:
#             results.append({
#                 "building_id": building_id,
#                 "wall_id_list": tuple(g),
#                 "new_id": f"combined_{new_id_counter}"
#             })
#             new_id_counter += 1

#     to_join = pd.DataFrame(results)
#     logger.info(f"{len(to_join)} wall groups can be joined.")
#     return to_join



def normalize_vector(v):
    v = np.array(v, dtype=float)
    norm = np.linalg.norm(v)
    return list(v / norm) if norm != 0 else [0.0, 0.0, 0.0]


def share_vertex(coords_a, coords_b) -> bool:
    """Return True if the two coordinate lists share at least one identical vertex."""
    set_a = {tuple(pt) for pt in coords_a}
    set_b = {tuple(pt) for pt in coords_b}
    return len(set_a.intersection(set_b)) > 0
    

def find_walls_to_join_common_vertex(df) -> pd.DataFrame:
    """
    Identify walls to join based on:
      - belonging to the same building
      - similar normals (angle difference < 3 degrees)
      - at least one shared vertex

    Required columns:
      building_id
      normal_vector_3d
      surface_coordinates
      wall_id
    """
    logger.info("Finding walls to join")

    df = df.copy()
    df["normalized_nv3d"] = df["normal_vector_3d"].apply(normalize_vector)

    results = []
    new_id_counter = 0

    # Process one building at a time
    for building_id, bdf in df.groupby("building_id"):

        normals = bdf["normalized_nv3d"].tolist()
        coords = bdf["surface_coordinates"].tolist()
        wall_ids = bdf["wall_id"].tolist()

        joins = []
        wall_neighbors_map = {wid: set() for wid in wall_ids}
        
        # Compare each unique pair within this building
        for i, j in combinations(range(len(wall_ids)), 2):

            angle = get_angle_0_to_180(normals[i][:2], normals[j][:2])
            if angle < 3:
                if share_vertex(coords[i], coords[j]):
                    joins.append((wall_ids[i], wall_ids[j]))
            # if not same angle but share vertex, add to neighbor map for snapping later
            else:
                if share_vertex(coords[i], coords[j]):
                    wall_neighbors_map[wall_ids[i]].add(wall_ids[j])
                    wall_neighbors_map[wall_ids[j]].add(wall_ids[i])
        # Merge overlapping join pairs into groups
        groups = []
        for a, b in joins:
            added = False
            for g in groups:
                if a in g or b in g:
                    g.update([a, b])
                    added = True
                    break
            if not added:
                groups.append(set([a, b]))
                
        # Output groups for this building
        for g in groups:
            group_set = set(g)
            group_neighbors = set()
            
            for wid in group_set:
                group_neighbors.update(wall_neighbors_map[wid])
            # remove the wall the neighbour check is about from the map of neighbors
            group_neighbors -= group_set

            results.append({
                "building_id": building_id,
                "wall_id_list": tuple(g),
                "new_id": f"combined_{new_id_counter}",
                "neighbors": tuple(group_neighbors)
            })
            new_id_counter += 1

    to_join = pd.DataFrame(results)

    logger.info(f"{len(to_join)} wall groups can be joined.")
    return to_join


def make_wall_2d(wall, v1, v2):
    """
    Project a list of 3D points (wall) onto a 2D plane defined by v1, v2.
    """
    wall = np.asarray(wall)
    x_new = wall @ v1
    y_new = wall @ v2
    return list(map(tuple, np.column_stack((x_new, y_new))))


def make_wall_3d(wall_2d, v1, v2, origin_on_plane):
    """
    Reproject 2D wall coordinates (wall) back into 3D using v1, v2, and plane origin.
    """
    wall_2d = np.asarray(wall_2d)
    wall_3d = origin_on_plane + np.outer(wall_2d[:, 0], v1) + np.outer(wall_2d[:, 1], v2)
    return [tuple(np.round(p, 3)) for p in wall_3d]


def get_joined_surface_of_walls(list_of_surfaces, normal, tol=0.15): #as normal vector is normalized, tol can be in the same unit as the coordinates, e.g. 0.1m, which should be sufficient to catch most of the cases where walls are not exactly coplanar but close enough to be joined
    """
    Join multiple wall surfaces into a single 3D polygon if coplanar.
    Returns (success, joined_surface_3d, area).
    """
    
    a, b, c = normal
    n = np.array(normal, dtype=float)
    n /= np.linalg.norm(n)

    # --- define reference plane using centroid of all points ---
    all_points = np.vstack(list_of_surfaces)
    centroid = all_points.mean(axis=0)
    d = -np.dot(n, centroid)

    # Quick coplanarity check (use only first two corners per wall)
    for wall in list_of_surfaces:
        for p in wall[:2]:
            if abs(np.dot(n, p) + d) > tol:
                return False, [], 0.0

    # Find orthonormal basis for the plane
    v1 = np.cross(n, [0, 0, 1]) if abs(n[2]) < 0.9 else np.cross(n, [0, 1, 0])
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(n, v1)

    # Project all walls to 2D polygons
    polygons_2d = [Polygon(make_wall_2d(wall, v1, v2)).buffer(0) for wall in list_of_surfaces]

    # Union operation
    joined = shapely.union_all(polygons_2d)
    if joined.is_empty:
        return False, [], 0.0

    area = joined.area

    # Compute origin of projection plane
    origin_on_plane = -d * n  # point closest to origin on plane

    # Convert result to 3D
    if isinstance(joined, Polygon):
        coords = list(joined.exterior.coords)
    elif isinstance(joined, MultiPolygon):
        joined = shapely.union_all([poly.buffer(0.03) for poly in joined.geoms])
        if not isinstance(joined, Polygon):
            return False, [], 0.0
        coords = list(joined.exterior.coords)
    else:
        return False, [], 0.0

    joined_surface_3d = make_wall_3d(coords, v1, v2, origin_on_plane)
    return True, joined_surface_3d, area


# refactor
def remove_double_coordinates(df, col_name):
    '''
    Removes duplicate coordinates from a specified column in a DataFrame.

    PARAMS:
    df (pandas.DataFrame): df containing the coordinates row that should be checked for duplicates
    col_name (str): name of the column in the df for which the duplicate coordinates should be removed

    RETURNS:
    df (pandas.DataFrame): modified df with duplicate coordinates removed from the specified column.
    '''
    df = df.copy()
    changed = 0
    for idx, coords in df[col_name].items():
        if not coords:
            continue
        unique = list(dict.fromkeys(coords))  # preserve order, remove dups
        if unique[0] != unique[-1]:
            unique.append(unique[0])  # close loop
        if unique != coords:
            df.at[idx, col_name] = unique
            changed += 1
    logger.info(f"{changed} coordinate sets modified.")
    return df


def snap_merged_wall_to_neighbors(merged_surface_coordinates, neighbor_surfaces_coordinates, tol=0.05):
    """
    Due to high tolerance on coplanarity check, merged walls may have vertices slightly off from neighbor walls.
    To algin them after merging with their neighbours, snap vertices of merged wall to nearby neighbor walls within a tolerance.
    merged_wall_coords: list of 3D tuples
    neighbor_coords_list: list of lists of 3D tuples
    """

    merged_pts = np.array(merged_surface_coordinates)
    for i, pt in enumerate(merged_pts):
        dists = np.linalg.norm(neighbor_surfaces_coordinates - pt, axis=1)
        min_idx = np.argmin(dists)
        if dists[min_idx] < tol:
            merged_pts[i] = neighbor_surfaces_coordinates[min_idx]
    merged_pts = [tuple(p) for p in merged_pts]
    return merged_pts


# refactor
def join_walls(df, to_join):
    """
    Join walls listed in `to_join` into new combined walls.
    Returns updated df and a filtered to_join DataFrame.
    Df need columns surface coordinates, normal_vector_3d
    to_join needs columns wall_id_list, building_id, new_id
    """
    # check if there are any walls to join:
    if to_join.empty:
        logger.info("No walls to join.")
        return df
    
    df = df.copy()
    before = len(df)
    logger.info(f"Joining {before} walls...")

    # Create lookup dicts to avoid repeated DataFrame scans
    wall_map = df.set_index("wall_id").to_dict(orient="index")
    to_drop = set()
    new_walls = []
    new_df = pd.DataFrame({ #check that columns below are same!!
            "building_id": "building_id",
            "wall_id": "new_id",
            "ground_surface": [],
            "surface_coordinates": [],
            "area": 0,
            "normal_vector_3d": 0
        })
    if not set(df.columns).issubset(new_df.columns):
        print("[!] df is missing columns: orig cols:", df.columns, "should have cols:", new_df.columns)
        return None
    del new_df
    
    for _, row in tqdm(to_join.iterrows(), total=len(to_join)):
        wall_ids = row["wall_id_list"]
        building_id = row["building_id"]
        new_id = row["new_id"]
        neighbor_ids = row.get("neighbors", ())

        walls = [wall_map[w] for w in wall_ids if w in wall_map]
        if not walls:
            continue

        list_of_coords = [w["surface_coordinates"] for w in walls]
        normal_vec = walls[0]["normal_vector_3d"]
        ground_surface = walls[0].get("ground_surface", None)

        was_success, joined_surface, area = get_joined_surface_of_walls(list_of_coords, normal_vec)
        if not was_success:
            continue

        # --- Snapping edges to old neighboring walls ---
        for n_id in neighbor_ids:
            if n_id not in wall_map:
                continue
            neighbor_wall = wall_map[n_id]["surface_coordinates"]
            joined_surface = snap_merged_wall_to_neighbors(joined_surface, neighbor_wall, tol=0.2)

        new_walls.append({
            "building_id": building_id,
            "wall_id": new_id,
            "ground_surface": ground_surface,
            "surface_coordinates": joined_surface,
            "area": area,
            "normal_vector_3d": normal_vec
        })
        to_drop.update(wall_ids)

    # Build new DataFrame efficiently
    new_df = pd.DataFrame(new_walls)
    #check that new_df has same columns before concat
    
    
    new_df = remove_double_coordinates(new_df, "surface_coordinates")
    df = df[~df["wall_id"].isin(to_drop)].reset_index(drop=True)
    df = pd.concat([df, new_df], ignore_index=True)

    logger.info(f"{len(df)} walls after joining ({before - len(df)} removed)")
    if to_drop:
        logger.debug(f"Dropped wall IDs:" + "\n".join(map(str, to_drop)))
    return df
