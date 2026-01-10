import numpy as np
import cv2
import pandas as pd


# -------------
# World from px  (Rectified)
# -------------

def wall_local_frame_from_normal(wall_points, normal_vector):
    """
    wall_points: Nx3 array of 3D wall corners
    normal_vector: 3-element array
    Returns: origin, x_axis, y_axis
    """
    wall_points = np.array(wall_points)
    origin = wall_points[0]

    # pick x-axis along the first edge
    x_axis = wall_points[1] - wall_points[0]
    x_axis /= np.linalg.norm(x_axis)
    # y-axis perpendicular to x-axis in the wall plane
    y_axis = np.cross(normal_vector, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return origin, x_axis, y_axis


def point_to_wall2D(point, origin, x_axis, y_axis):
    vec = point - origin
    u = np.dot(vec, x_axis)
    v = np.dot(vec, y_axis)
    return np.array([u, v])

def compute_rect_to_wall_homography(rect_px, wall_points, normal_vector):
    """
    rect_px: Nx2 rectified image points (pixel coordinates)
    wall_points: Nx3 3D wall points (same correspondences)
    normal_vector: 3-element array
    Returns: H_rect_to_wall2D, origin, x_axis, y_axis
    """
    origin, x_axis, y_axis = wall_local_frame_from_normal(wall_points, normal_vector)
    wall_2D = np.array([point_to_wall2D(p, origin, x_axis, y_axis) for p in wall_points])
    H, _ = cv2.findHomography(rect_px, wall_2D)
    return H, origin, x_axis, y_axis

def rectified_px_to_world(u_r, v_r, H, origin, x_axis, y_axis):
    p = np.array([u_r, v_r, 1.0])
    uv = H @ p
    uv /= uv[2]
    u, v = uv[:2]
    # Map back to 3D
    X_world = origin + u * x_axis + v * y_axis
    return tuple(float(x) for x in X_world)


def calculate_real_world_coordinates(detections: pd.DataFrame, rectified_surface_px: list, wall_surface_3D: list, normal_vector_3D: np.ndarray):
    """
    Calculates real world coordinates of detected objects in rectified images.
    Uses detections dataframe (saved as pkl), returns updated dataframe with new column 'real_world_detections' containing real world coordinates.

    Args:
        detections (pd.DataFrame): DataFrame with 'polygon' column (list of rectified image points)
        rectified_surface_px (list): list of rectified image points corresponding to wall corners
        wall_surface_3D (list): list of 3D wall points (any number of corners)
        normal_vector_3D (np.ndarray): precomputed wall normal

    Returns: detections DataFrame with added 'real_world_detections' column,
             H_rect_to_wall2D, origin, x_axis, y_axis
    """
    detections = detections.copy()
    rect_px = np.array(rectified_surface_px)
    wall_points = np.array(wall_surface_3D)
    H_rect_to_wall2D, origin, x_axis, y_axis = compute_rect_to_wall_homography(rect_px, wall_points, normal_vector_3D)

    all_real_world = []
    for idx, row in detections.iterrows(): 
        real_world_points = []
        for pt in row['polygon']:
            X = rectified_px_to_world(pt[0], pt[1], H_rect_to_wall2D, origin, x_axis, y_axis)
            real_world_points.append(X)
        all_real_world.append((row['class_name'], real_world_points))

    detections['real_world_detections'] = all_real_world
    return detections, H_rect_to_wall2D, origin, x_axis, y_axis



# -------------
# Reconvert World to px
# -------------

def world_to_wall2D(point_3D, origin, x_axis, y_axis):
    vec = point_3D - origin
    u = np.dot(vec, x_axis)
    v = np.dot(vec, y_axis)
    return np.array([u, v])

def wall2D_to_rectified_px(u, v, H_wall2D_to_rect):
    p = np.array([u, v, 1.0])
    uv = H_wall2D_to_rect @ p
    uv /= uv[2]
    return tuple(uv[:2])


def calculate_pixel_coordinates(detection: list, H_rect_to_wall2D: np.ndarray, origin: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray):    
    # Invert homography to map wall 2D → rectified px
    H_wall2D_to_rect = np.linalg.inv(H_rect_to_wall2D)

    pixel_points = []
    for X_world in detection:
        # Map 3D world point to wall-local 2D
        uv_wall = world_to_wall2D(np.array(X_world), origin, x_axis, y_axis)
        # Map wall-local 2D to rectified image pixels
        u_r, v_r = wall2D_to_rectified_px(uv_wall[0], uv_wall[1], H_wall2D_to_rect)
        pixel_points.append((float(u_r), float(v_r)))
   
    return pixel_points
