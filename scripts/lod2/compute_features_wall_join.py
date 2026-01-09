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


# def get_angle_0_to_180(vec1, vec2):
#     '''
#     Takes two vectors (2d or 3d) and computes the angle in degrees between both. Order of the vectors does not matter

#     PARAMS:
#     * vec1, vec2 (2d or 3d tuples): vectors between which the angle should be calculated
#     '''
#     vec1 = np.array(vec1)
#     vec2 = np.array(vec2)
    
#     # Calculate the dot product
#     dot_product = np.dot(vec1, vec2)
    
#     # Calculate the magnitudes of the vectors
#     magnitude_a = np.linalg.norm(vec1)
#     magnitude_b = np.linalg.norm(vec2)
    
#     # Calculate the cosine of the angle
#     if magnitude_a * magnitude_b != 0:
#         cos_theta = np.clip(dot_product / (magnitude_a * magnitude_b), -1, 1)
#     else:  # if both magnitudes multiplied are 0, one of them needs to be 0 -> check this!
#         print("vec1:", vec1)
#         print("vec2:", vec2)
#         print("Magnitude_a:", magnitude_a)
#         print("Magnitude_b:", magnitude_b)
#         return 0
    
#     # Calculate the angle in radians
#     angle_radians = np.arccos(cos_theta)
    
#     # Convert the angle to degrees
#     angle_degrees = np.degrees(angle_radians)
    
#     return angle_degrees


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


# def check_for_id_in_to_join(wall_id, to_join, part_id):
#     ''' 
#     search through all walls of this part that are already being joined for wall_id.
#     If it is already in there, return index in to_join

#     PARAMS:
#     * wall_id (str): which wall to search for
#     * to_join (df): df with three columns: part_id, wall_id_list (walls that are already to join), new_id (for new, joined wall). Each row will be joined to one wall. Can have several rows per part_id
#     * part_id (str): which part the wall is in (significantly speeds up the process to search for it)

#     RETURNS:
#     * whether the wall_id is already in any of the to_join rows 
#     * indx_nr: index of the row it is in
#     '''
#     # filter to_join for only those rows concerning the given part_id
#     to_join_for_this_part = to_join[to_join["part_id"] == part_id]
#     # search in those for wall_id
#     for indx_nr, row in to_join_for_this_part.iterrows():
#         if wall_id in row["wall_id_list"]:
#             return True, indx_nr
        
#     return False, 0



# def find_walls_to_join(df):
#     """_summary_

#     Args:
#         df (pd or gpd): needs columns: part_id, surface_coordinates, normalized_nv3d, wall_id
#     """

#     grouped_by_part = df.groupby("part_id").agg(
#         coordinates=('surface_coordinates', list), 
#         normals = ("normalized_nv3d", list),
#         wall_ids = ("wall_id", list)        
#     ).reset_index()

#     # initialize df for which walls to join. For each pair or group of walls (all saved as list in wall_id_list) that fulfill the requirements of sharing an edge and having the same normal, save part_id and make up a new_id.
#     to_join = pd.DataFrame({"part_id": [], "wall_id_list": [], "new_id": []})

#     count = 0
#     self_made_id_nr = 0
#     for index, part in grouped_by_part.iterrows():
#         # for each part, check always two walls and whether they fulfill the requirements for a join
#         for x, normal_vec1 in enumerate(part["normals"]):
#             for y, normal_vec2 in enumerate(part["normals"]):
#                 if x < y:
#                     # test whether the normal_vecs are very similar (angle < 1°) 
#                     angle = get_angle_0_to_180(normal_vec1[0:2], normal_vec2[0:2])
#                     if (angle < 1):
#                         # test whether the walls have a common edge                          
#                         if common_edge(part["coordinates"][x], part["coordinates"][y]):
#                             count += 1
#                             # include in df
#                             # three main possibilities:
#                             # 1. both walls are not in any wall_id_list tuple -> make new row with new id and everything
#                             # 2. one of the walls already is -> add other wall_id to this wall_id_list tuple (no new row)
#                             # 3. both walls already are -> join both lists: take the row where wall_x already is and add wall_y plus the other walls in the wall_id_list of wall_y, delete the row where wall_y was present
#                             x_in_join, x_id_in_join = check_for_id_in_to_join(part["wall_ids"][x], to_join, part["part_id"])
#                             y_in_join, y_id_in_join = check_for_id_in_to_join(part["wall_ids"][y], to_join, part["part_id"])
#                             # 1. both walls are not in any wall_id_list tuple
#                             if not x_in_join and not y_in_join:
#                                 new_id = "combined_" + str(self_made_id_nr)
#                                 self_made_id_nr += 1
#                                 new_join = pd.DataFrame({
#                                     "part_id": [part["part_id"]],
#                                     "wall_id_list": [(part["wall_ids"][x], part["wall_ids"][y])], 
#                                     "new_id": [new_id]
#                                     })
#                                 to_join = pd.concat((to_join, new_join)).reset_index(drop = True)
#                             # 2. one of the walls already is
#                             elif x_in_join and not y_in_join:
#                                 wall_id_list_before = to_join.at[x_id_in_join, "wall_id_list"]
#                                 wall_id_list_after = wall_id_list_before + (part["wall_ids"][y],)
#                                 to_join.at[x_id_in_join, "wall_id_list"] = wall_id_list_after
#                             elif y_in_join and not x_in_join:
#                                 wall_id_list_before = to_join.at[y_id_in_join, "wall_id_list"]
#                                 wall_id_list_after = wall_id_list_before + (part["wall_ids"][x],)
#                                 to_join.at[y_id_in_join, "wall_id_list"] = wall_id_list_after
#                             # 3. both walls already are
#                             else:
#                                 wall_id_list_before = to_join.at[x_id_in_join, "wall_id_list"]
#                                 wall_id_list_after = wall_id_list_before + to_join.at[y_id_in_join, "wall_id_list"]
#                                 to_join.at[x_id_in_join, "wall_id_list"] = wall_id_list_after
#                                 # delete other row
#                                 to_join = to_join.drop(index=y_id_in_join)
                    
#     to_join = to_join.reset_index(drop= True)
#     print(count, "wall pairs can be joined") #162326
#     return to_join


# refactor
def find_walls_to_join(df) -> pd.DataFrame:
    """
    Identify walls belonging to the same part that share an edge
    and have nearly identical normals (angle < 1°).

    df: pandas df or gdf, needs columns normal_vector_3d, surface_coordinates, wall_id
    """
    logger.info("Finding walls to join")
    def normalize_vector(v):
        v = np.array(v, dtype=float)
        norm = np.linalg.norm(v)
        return list(v / norm) if norm != 0 else [0.0, 0.0, 0.0]
    df = df.copy()
    df["normalized_nv3d"] = df["normal_vector_3d"].apply(normalize_vector)

    grouped = df.groupby("building_id").agg(
        coordinates=('surface_coordinates', list),
        normals=('normalized_nv3d', list),
        wall_ids=('wall_id', list)
    ).reset_index()

    results = []
    new_id_counter = 0

    for _, part in grouped.iterrows():
        building_id = part.building_id
        normals, coords, wall_ids = part.normals, part.coordinates, part.wall_ids
        joins = []  # list of tuples of wall_ids to join

        # Compare each unique pair once
        for (i, j) in combinations(range(len(normals)), 2):
            if get_angle_0_to_180(normals[i][:2], normals[j][:2]) < 1:
                if common_edge(coords[i], coords[j]):
                    joins.append((wall_ids[i], wall_ids[j]))

        # Merge overlapping groups efficiently
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

        for g in groups:
            results.append({
                "building_id": building_id,
                "wall_id_list": tuple(g),
                "new_id": f"combined_{new_id_counter}"
            })
            new_id_counter += 1

    to_join = pd.DataFrame(results)
    logger.info(f"{len(to_join)} wall groups can be joined.")
    return to_join



# def make_wall_2d(wall, v1, v2):
#     '''
#     Projects a 3D wall onto a 2D plane using two vectors.

#     PARAMS:
#     wall (list of 3d tuples): coordinate tuples (x, y, z) of wall which should be projected
#     v1, v2 (3d vectors): both vectors used for projection

#     RETURNS:
#     wall_2d: A list of 2d tuples representing the projected wall
#     '''
#     wall_2d = []
#     for point in wall:
#         x_new = np.dot(v1, np.array(point))
#         y_new = np.dot(v2, np.array(point))
#         wall_2d.append((x_new, y_new))

#     return wall_2d


# def make_wall_3d(wall, v1, v2, origin_on_plane):
#     '''
#     Projects a 2D wall back to 3D plane using both vectors used for 3d-2d projection and the origin_on_plane

#     PARAMS:
#     wall (list of 2d tuples): coordinate tuples (x, y) of wall which should be projected
#     v1, v2 (3d vectors): both vectors used for projection
#     origin_on_plane (3d vector): vector pointing from the origin of the 3D space to the origin of the plane where the wall was projected onto

#     RETURNS:
#     wall_3d: A list of 3d tuples representing the projected wall
#     '''
#     wall_3d = []
#     for point in wall:
#         # for each point: sum up vector to origin on plane, both projecting vectors (3d) times the according coordinate from the 2d wall point to get back to the 3d coordinate
#         point_3d = tuple(np.round(origin_on_plane + point[0] * v1 + point[1] * v2, 3))
#         wall_3d.append(point_3d)

#     return wall_3d

# refactor
def make_wall_2d(wall, v1, v2):
    """
    Project a list of 3D points onto a 2D plane defined by v1, v2.
    """
    wall = np.asarray(wall)
    x_new = wall @ v1
    y_new = wall @ v2
    return list(map(tuple, np.column_stack((x_new, y_new))))


def make_wall_3d(wall_2d, v1, v2, origin_on_plane):
    """
    Reproject 2D wall coordinates back into 3D using v1, v2, and plane origin.
    """
    wall_2d = np.asarray(wall_2d)
    wall_3d = origin_on_plane + np.outer(wall_2d[:, 0], v1) + np.outer(wall_2d[:, 1], v2)
    return [tuple(np.round(p, 3)) for p in wall_3d]


# def get_joined_surface_of_walls(list_of_surfaces, normal):
#     '''
#     Joins multiple wall surfaces into a single surface in 3D space.

#     This function takes a list of wall surfaces and a normal vector that defines the orientation of the plane
#     in which the walls lie. It projects the walls onto a 2D plane, checks if they are coplanar, and then
#     combines them into a single polygonal surface. The resulting surface is then transformed back into 3D space.

#     PARAMS:
#     * list_of_surfaces (list of list of 3d tuple): A list containing surfaces, where each surface is represented
#         as a list of 3d tuples
#     * normal (3d vector): A tuple representing the normal vector of the plane (a, b, c) to which the walls are aligned.

#     RETURNS:
#     * bool: True if the surfaces were successfully joined, False otherwise.
#     * list: The joined surface in 3D space, or an empty list if joining failed.
#     * float: The area of the joined surface, or 0 if joining failed.
#     '''
#     # project all walls in 2d
#     a,b,c = normal
#     d = - (a * list_of_surfaces[0][0][0] + b* list_of_surfaces[0][0][1] + c* list_of_surfaces[0][0][2])

#     # short check whether the other walls also lie on the plane (not exhaustive, but should catch most of the cases. These should in the best case not happen though)
#     for wall2 in list_of_surfaces:
#         check_corner1 = a* wall2[0][0] + b*wall2[0][1] + c*wall2[0][2] + d  # should be 0 or close to 0
#         check_corner2 = a* wall2[1][0] + b*wall2[1][1] + c*wall2[1][2] + d  # should be 0 or close to 0

#         # if not, joining fails
#         if np.abs(check_corner1) > 1 or np.abs(check_corner2) > 1:
#             return False, [], 0

#     # get two vectors, both orthonormal to the normal and to each other, to span the plane
#     v1 = np.array(list_of_surfaces[0][0]) - np.array(list_of_surfaces[0][1])
#     v1 = v1/np.linalg.norm(v1)
#     v2 = np.cross(normal, v1)

#     # project all walls to 2d
#     list_of_polygons_2d = []
#     for surface in list_of_surfaces:
#         surface_2d = make_wall_2d(surface, v1, v2)
#         list_of_polygons_2d.append(Polygon(surface_2d).buffer(0))

#     # join
#     joined_surface = shapely.union_all(list_of_polygons_2d)
#     area = joined_surface.area

#     # get back to 3d again, first need vector pointing from the origin of the 3D space to the origin of the plane where the wall was projected onto
#     distance_origin_to_plane = (a * 0 + b * 0 + c * 0 + d) / np.sqrt(a**2 + b**2 + c**2)
#     # Project the point onto the plane
#     origin_on_plane = [0,0,0] - distance_origin_to_plane * np.array(normal)

#     if isinstance(joined_surface, Polygon):
#         # make all coordinates of polygon to list of coords again
#         joined_surface_coords = list(joined_surface.exterior.coords)
        
#     elif isinstance(joined_surface, MultiPolygon):
#         # try to make one polygon with buffering
#         joined_surface_buffered = shapely.union_all([poly.buffer(0.03, join_style="mitre") for poly in joined_surface.geoms])
#         if isinstance(joined_surface_buffered, Polygon):
#             joined_surface_coords = list(joined_surface_buffered.exterior.coords)
#         else:
#             # not joining            
#             return False, [], 0
#     else:
#         # Resulting shape is no polygon or multipolygon, not joining
#         return False, [], 0
    
#     # back to 3d
#     joined_surface_3d = make_wall_3d(joined_surface_coords, v1, v2, origin_on_plane)    

#     return True, joined_surface_3d, area


def get_joined_surface_of_walls(list_of_surfaces, normal, tol=1e-3):
    """
    Join multiple wall surfaces into a single 3D polygon if coplanar.
    Returns (success, joined_surface_3d, area).
    """
    
    a, b, c = normal
    n = np.array(normal, dtype=float)
    n /= np.linalg.norm(n)

    # Reference plane equation: a*x + b*y + c*z + d = 0
    first_point = np.array(list_of_surfaces[0][0])
    d = -np.dot(n, first_point)

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



# def join_walls(df, to_join):
#     before = len(df)
#     print("[i] Joining walls.", before, "walls before") #1084062

#     # go through rows in to_join and always save new wall as row in a new dataframe which is appended after the process
#     joined = to_join.copy()
#     indices_that_were_not_joined = []
#     counter_not_joined = 0
#     new_walls = pd.DataFrame(columns=["part_id", "wall_id", "surface_coordinates", "area", "normalized_nv3d"])
#     df_temp = df.copy()
#     for index, row in to_join.iterrows():
#         if index % 10000 == 0:
#             print(index, "/", len(to_join))
#         # get a list of the coordinates and corresponding indices from each wall that should be joined
#         list_of_wall_coordinates = list(df[df["wall_id"].isin(row["wall_id_list"])]["surface_coordinates"])
#         list_of_indices = list(df[df["wall_id"].isin(row["wall_id_list"])].index)

#         # get normal vec of one of the walls in list (in this case its just the first one)
#         normal_vec = df[df["wall_id"].isin(row["wall_id_list"])]["normalized_nv3d"].values[0]

#         # get joined surface
#         successfull_join, joined_surface, area = get_joined_surface_of_walls(list_of_wall_coordinates, normal_vec)

#         if successfull_join:
#             new_walls.loc[index] = [row["part_id"], row["new_id"], joined_surface, area, normal_vec]

#             # remove all walls from df that were just joined (to make sure the same wall will not be used again)
#             df_temp = df_temp.drop(list_of_indices)
#         else:
#             # join not successfull: make sure that row is deleted from joined in the end
#             indices_that_were_not_joined.append(index)
#             counter_not_joined += 1
            
#     # delete duplicate coordinates
#     new_walls = remove_double_coordinates(new_walls, "surface_coordinates")

#     print("Start deleting old walls and concatenate new walls")

#     df = pd.concat((df_temp, new_walls)).reset_index(drop = True)

#     joined = joined.drop(indices_that_were_not_joined)

#     print("Dataframes joined successfully")

#     print(len(df), "after joining walls;", before - len(df), "walls less")
#     print(counter_not_joined, "joins were not successful and therefor not performed")
#     return df, joined



# refactor
def join_walls(df, to_join):
    """
    Join walls listed in `to_join` into new combined walls.
    Returns updated df and a filtered to_join DataFrame.
    Df need columns surface coordinates, normal_vector_3d
    to_join needs columns wall_id_list, building_id, new_id
    """
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
    if not df.columns.sort_values().equals(new_df.columns.sort_values()):
        print("[!] dataframes dont have same structure: orig cols:", df.columns, "new cols:", new_df.columns)
        return None
    del new_df
    
    for _, row in tqdm(to_join.iterrows(), total=len(to_join)):
        wall_ids = row["wall_id_list"]
        building_id = row["building_id"]
        new_id = row["new_id"]
        

        walls = [wall_map[w] for w in wall_ids if w in wall_map]
        if not walls:
            continue

        list_of_coords = [w["surface_coordinates"] for w in walls]
        normal_vec = walls[0]["normal_vector_3d"]
        ground_surface = walls[0].get("ground_surface", None)

        success, joined_surface, area = get_joined_surface_of_walls(list_of_coords, normal_vec)
        if not success:
            continue

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
