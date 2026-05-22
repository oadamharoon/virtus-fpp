import cv2
import numpy as np
import os
from scipy.spatial.transform import Rotation as R_scipy

def projector_rw2dt(proj_mat_path, h_p, w_p, scaling_factor=1.0):
    # Load the projector intrinsic matrix from the text file
    projector_matrix = np.loadtxt(proj_mat_path, delimiter='\t', usecols=(0,1,2))

    # Read fx, fy, cx, cy from the projector matrix
    ((fx,_,cx),(_,fy,cy),(_,_,_)) = projector_matrix

    # Calculate scaling factors
    x_scale = scaling_factor * w_p / fx
    y_scale = scaling_factor * h_p / fy

    print("Projector Parameters:")
    print(f"Scaling factors: x_scale: {x_scale}, y_scale: {y_scale}")

    return [x_scale, y_scale]

def inverse_camera(proj_mat_path, proj_rot_path, proj_trans_path,
                   w_p, h_p, screen_dist=1.0):
    """
    Loads projector intrinsics/extrinsics from text files and performs
    inverse pinhole back-projection to compute real-world projection size.
    """

    # --- Load intrinsic matrix (3x3) ---
    Mint = np.loadtxt(proj_mat_path, delimiter='\t', usecols=(0,1,2))

    # --- Load rotation matrix (3x3) ---
    R = np.loadtxt(proj_rot_path, delimiter='\t', usecols=(0,1,2))

    # --- Load translation vector (3x1) ---
    t = np.loadtxt(proj_trans_path, delimiter='\t', usecols=(0,))
    t = t.reshape(3,1)

    # Build extrinsic matrix (3x4)
    # Mext = np.hstack([R, t])
    Mext = R

    # Convert to float64
    Mint = Mint.astype(np.float64)
    Mext = Mext.astype(np.float64)

    # Inverses
    Mint_inv  = np.linalg.inv(Mint)
    Mext_pinv = np.linalg.inv(Mext)

    # Pixel corners
    px_0    = np.array([[0.0], [0.0], [1.0]])
    px_edge = np.array([[float(w_p)], [float(h_p)], [1.0]])

    def backproject(px):
        v = screen_dist * Mext_pinv @ (Mint_inv @ px)
        v = v[:3]
        v = v / v[2]
        return v

    world_0    = backproject(px_0)
    world_edge = backproject(px_edge)

    dims = np.abs(world_edge) + np.abs(world_0)
    width, height, _ = dims

    print("Projector (inverse-camera) Parameters:")
    print(f"World corner [0,0]:         {world_0.flatten()}")
    print(f"World corner [{w_p},{h_p}]: {world_edge.flatten()}")
    print(f"Real World Dimensions: Width: {width}, Height: {height}")

    return [float(width), float(height)]

# Axis remap from OpenCV camera-local frame to Isaac USD-world, assuming the
# camera looks toward +X_world with +Z_world up.
R_OPENCV_TO_ISAAC_WORLD = np.array([
    [0.0,  0.0,  1.0],
    [-1.0, 0.0,  0.0],
    [0.0, -1.0,  0.0],
])

# USD prim default local frame (X right, Y up, Z back) vs OpenCV local frame
# (X right, Y down, Z forward). Used when converting a calibration rotation
# into a prim orient.
R_USD_LOCAL_TO_OPENCV_LOCAL = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
])

MM_TO_M = 1.0 / 1000.0


def read_extrinsics(rot_path, trans_path):
    R = np.loadtxt(rot_path, delimiter='\t', usecols=(0,1,2))
    T = np.asarray(np.loadtxt(trans_path), dtype=np.float64).reshape(3)
    return R, T


def device_pose_in_camera_frame(R, T):
    # OpenCV calibration's world-to-device transform with camera at origin:
    # the device's position in the camera frame is -R^T * T.
    return -R.T @ T


def opencv_offset_to_isaac_world(offset_mm):
    offset_m = np.asarray(offset_mm, dtype=np.float64) * MM_TO_M
    return R_OPENCV_TO_ISAAC_WORLD @ offset_m


def projector_world_position(proj_rot_path, proj_trans_path, camera_world_pos):
    R, T = read_extrinsics(proj_rot_path, proj_trans_path)
    offset_world = opencv_offset_to_isaac_world(device_pose_in_camera_frame(R, T))
    return np.asarray(camera_world_pos, dtype=np.float64) + offset_world


def projector_world_orientation(proj_rot_path):
    """
    Derive the projector's USD prim orient from the OpenCV world-to-projector
    rotation matrix. Returns (Q, euler_xyz_extrinsic_deg, euler_XYZ_intrinsic_deg).
    For an xformOp:rotateXYZ on the prim, pass the xyz-extrinsic angles.

    Sanity check: with R_proj = I, the xyz-extrinsic output is an alias of
    the camera's own (270, 180, 90), i.e. the prim ends up looking +X_world
    with +Z up just like the camera.
    """
    R_proj = np.loadtxt(proj_rot_path, delimiter='\t', usecols=(0,1,2))
    Q = R_OPENCV_TO_ISAAC_WORLD @ R_proj.T @ R_USD_LOCAL_TO_OPENCV_LOCAL
    rot = R_scipy.from_matrix(Q)
    euler_extr = rot.as_euler('xyz', degrees=True)
    euler_intr = rot.as_euler('XYZ', degrees=True)

    print("Projector Orientation (from calibration):")
    print(f"Euler xyz extrinsic (deg): {euler_extr}")
    print(f"Euler XYZ intrinsic (deg): {euler_intr}")

    return Q, euler_extr, euler_intr


def projector_extrinsics(proj_extrinsics_path):
    # Load the projector extrinsics from the text file
    projector_extrinsics = np.loadtxt(proj_extrinsics_path, delimiter='\t', usecols=(0,1,2))

    # Read rotation and translation from the projector extrinsics matrix
    ((r11,r12,r13),(r21,r22,r23),(r31,r32,r33)) = projector_extrinsics[:3,:3]
    ((t1,t2,t3)) = projector_extrinsics[:3,3]

    # Print the projector extrinsics to the console for debugging
    print("Projector Extrinsics:")
    print(f"Rotation: {projector_extrinsics[:3,:3]}")
    print(f"Translation: {projector_extrinsics[:3,3]}")

    return [r11, r12, r13, r21, r22, r23, r31, r32, r33, t1, t2, t3]

def camera_rw2dt(cam_mat_path, h_c, w_c, p_size, f_stop, f_dist):
    # Load the camera matrix from the text file
    camera_matrix = np.loadtxt(cam_mat_path, delimiter='\t', usecols=(0,1,2))
    # Camera sensor resolution in pixels
    width, height = w_c, h_c

    # Pixel size in microns, aperture and focus distance from the camera sensor specification
    # Note: to disable the depth of field effect, set the f_stop to 0.0. This is useful for debugging.
    pixel_size = p_size       # in mm, 3 microns is a common pixel size for high resolution cameras
    # f_stop = 1.8            # f-number, the ratio of the lens focal length to the diameter of the entrance pupil
    f_stop = f_stop
    # focus_distance = 0.6    # in meters, the distance from the camera to the object plane
    focus_distance = f_dist

    # Read fx, fy, cx, cy from the camera matrix
    ((fx,_,cx),(_,fy,cy),(_,_,_)) = camera_matrix

    # Calculate the focal length and aperture size from the camera matrix
    horizontal_aperture =  pixel_size * width                   # The aperture size in mm
    vertical_aperture =  pixel_size * height
    focal_length_x  = fx * pixel_size
    focal_length_y  = fy * pixel_size
    focal_length = (focal_length_x + focal_length_y) / 2        # The focal length in mm

    # Principal-point offset in mm. Linear pinhole intrinsics (NOT lens
    # distortion). Non-zero whenever (cx, cy) is off the image center.
    horizontal_aperture_offset = -(cx - width  / 2.0) * pixel_size
    vertical_aperture_offset   = (cy - height / 2.0) * pixel_size

    # Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
    focal_length = (focal_length / 10.0)                # Convert from mm to cm (or 1/10th of a world unit)
    focus_distance = (focus_distance)                   # The focus distance in meters
    lens_aperture = (f_stop * 100.0)                    # Convert the f-stop to Isaac Sim units
    horizontal_aperture = (horizontal_aperture / 10.0)  # Convert from mm to cm (or 1/10th of a world unit)
    vertical_aperture = (vertical_aperture / 10.0)
    # horizontal_aperture_offset = (horizontal_aperture_offset / 10.0)
    # vertical_aperture_offset   = (vertical_aperture_offset   / 10.0)

    # clipping_range = (0.05, 1.0e5)
    clipping_range = (0.05, 100.0)

    # Print the camera parameters to the console for debugging
    print("Camera Parameters:")
    print(f"Focal Length: {focal_length} cm")
    print(f"Focus Distance: {focus_distance} m")
    print(f"Lens Aperture: {lens_aperture} cm")
    print(f"Horizontal Aperture: {horizontal_aperture} cm")
    print(f"Vertical Aperture: {vertical_aperture} cm")
    print(f"Horizontal Aperture Offset: {horizontal_aperture_offset} cm")
    print(f"Vertical Aperture Offset: {vertical_aperture_offset} cm")
    print(f"Clipping Range: {clipping_range}")

    return [focal_length, focus_distance, lens_aperture,
            horizontal_aperture, vertical_aperture,
            horizontal_aperture_offset, vertical_aperture_offset,
            clipping_range]

def cam_extrinsics(cam_extrinsics_path):
    # Load the camera extrinsics from the text file
    camera_extrinsics = np.loadtxt(cam_extrinsics_path, delimiter='\t', usecols=(0,1,2))

    # Read rotation and translation from the camera extrinsics matrix
    ((r11,r12,r13),(r21,r22,r23),(r31,r32,r33)) = camera_extrinsics[:3,:3]
    ((t1,t2,t3)) = camera_extrinsics[:3,3]

    # Print the camera extrinsics to the console for debugging
    print("Camera Extrinsics:")
    print(f"Rotation: {camera_extrinsics[:3,:3]}")
    print(f"Translation: {camera_extrinsics[:3,3]}")

    return [r11, r12, r13, r21, r22, r23, r31, r32, r33, t1, t2, t3]
