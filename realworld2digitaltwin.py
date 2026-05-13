import cv2
import numpy as np
import os

def projector_rw2dt(proj_mat_path, h_p, w_p, scaling_factor=1.0):
    # Define the paths to rotation and translation matrices
    rot_path = os.path.join(proj_mat_path, "ProjRotationMatrix.txt")
    trans_path = os.path.join(proj_mat_path, "ProjTranslationVector.txt")

    # Loading them into the corresponding matrices
    Rp = np.loadtxt(rot_path, delimiter='\t', usecols=(0,1,2))
    Tp = np.loadtxt(trans_path, delimiter='\t', usecols=(0))

    # Calculate the orientation along individual axes based on the rotation matrix
    Rx = np.arctan2(Rp[2,1], Rp[2,2])

    # Load the projection matrix from the text file
    projector_matrix = np.loadtxt(proj_mat_path, delimiter='\t', usecols=(0,1,2))

    # Read fx, fy, cx, cy from the projector matrix
    ((fx,_,cx),(_,fy,cy),(_,_,_)) = projector_matrix

    # Calculate scaling factors
    x_scale = scaling_factor * fx / w_p
    y_scale = scaling_factor * fy / h_p

    # Printing the scaling factors to the console for debugging
    print("Projector Parameters:")
    print(f"Scaling factors: x_scale: {x_scale}, y_scale: {y_scale}")

    return [x_scale, y_scale]

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

    # Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
    focal_length = (focal_length / 10.0)                # Convert from mm to cm (or 1/10th of a world unit)
    focus_distance = (focus_distance)                   # The focus distance in meters
    lens_aperture = (f_stop * 100.0)                    # Convert the f-stop to Isaac Sim units
    horizontal_aperture = (horizontal_aperture / 10.0)  # Convert from mm to cm (or 1/10th of a world unit)
    vertical_aperture = (vertical_aperture / 10.0)

    # clipping_range = (0.05, 1.0e5)
    clipping_range = (0.05, 100.0)

    # Print the camera parameters to the console for debugging
    print("Camera Parameters:")
    print(f"Focal Length: {focal_length} cm")
    print(f"Focus Distance: {focus_distance} m")
    print(f"Lens Aperture: {lens_aperture} cm")
    print(f"Horizontal Aperture: {horizontal_aperture} cm")
    print(f"Vertical Aperture: {vertical_aperture} cm")
    print(f"Clipping Range: {clipping_range}")

    return [focal_length, focus_distance, lens_aperture, horizontal_aperture, vertical_aperture, clipping_range]

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