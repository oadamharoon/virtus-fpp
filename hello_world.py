from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.world import World
from omni.isaac.core.materials import OmniPBR
from omni.isaac.core import SimulationContext
from omni.isaac.examples.user_examples.CalibrationBoardGenerator import CalibrationBoardGenerator
from omni.isaac.examples.user_examples.realworld2digitaltwin import projector_rw2dt, camera_rw2dt
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils
import numpy as np
import omni
from pxr import Usd, UsdLux, Gf, Sdf, UsdGeom, UsdShade, Vt, UsdPhysics
import cv2
import os
import re
import time
import carb

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self.simulation_context = SimulationContext(set_defaults=True)
        
        # Operation mode
        self.mode = "calibrate"  # "calibrate", "scan", "training", or "ablation"

        # Add real-world system modeling flag and parameters
        self.use_real_world_params = False  # Set to True to use real-world camera/projector parameters
        self.real_world_params = {
            'cam_mat_path': "C:/Users/oadam/Downloads/calibdata062024/CamIntrinsicMatrix.txt",   # "C:/path/to/camera_matrix.txt"
            'proj_mat_path': "C:/Users/oadam/Downloads/calibdata062024/ProjIntrinsicMatrix.txt", # "C:/path/to/projector_matrix.txt"
            'cam_width': 544,               # Camera resolution width in pixels
            'cam_height': 514,              # Camera resolution height in pixels
            'fringe_pattern_width': 880,    # Fringe pattern width in pixels
            'fringe_pattern_height': 2400,  # Fringe pattern height in pixels
            'pixel_size': 5.86 * 1e-3,      # Camera pixel size in mm (5.86 microns)
            'f_stop': 1.8,                  # Camera f-stop
            'focus_distance': 2.0           # Camera focus distance in meters
        }
            
        # Camera and projector parameters (shared between modes)
        self.frame_count = 0
        self.texture_files = []
        self.current_texture_index = 0
        self.last_texture_update_time = 0.0
        self.texture_update_interval = 2 * 1/60
        self.frame_skip = 5
        self.frames_since_update = 0
        self.save_grayscale = True

        # Time tracking
        self.start_time = None
        self.end_time = None

        # Calibration mode parameters
        self.calibration_params = {
            'current_pose_index': 0,
            'frames_per_pose': 0,
            'frames_captured_for_current_pose': 0,
            'positions': self._get_calibration_positions(),
            'orientations': self._get_calibration_orientations()
        }

        # Scanning mode parameters
        self.scanning_params = {
            'current_angle': 0,
            'angle_increment': 60,  # degrees
            'total_angles': 6,     # 360/angle_increment = number of positions, e.g. 12 for 30 degrees
            'frames_per_angle': 0,  # Will be set based on texture files
            'frames_captured_for_current_angle': 0,
            'scan_radius': 0.25,    # Distance from center for rotation
            'object_scale': 1.5,    # Scale factor for the loaded object
            'base_position': Gf.Vec3d(0.6, -0.125, 1.5),  # Centered position in front of camera
            'base_rotation': Gf.Vec3f(0, 0, 0),  # Initial rotation 
            'rotation_axis': 'z',
            'scan_object': 'banana',  # Select from item in self.SCAN_OBJECTS
        }

        self.SCAN_OBJECTS = {
            # ==================================
            # Isaac Sim pre-defined YCB objects
            # ==================================

            "master_chef_can": {
                "asset_path": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Props/YCB/Axis_Aligned/002_master_chef_can.usd",
            },

            "cracker_box": {
                "asset_path": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd",
            },

            "sugar_box": {
                "asset_path": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Props/YCB/Axis_Aligned/004_sugar_box.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(270, self.scanning_params['base_rotation'][1], self.scanning_params['base_rotation'][2]),
                },
            },

            "tomato_soup_can": {
                "asset_path": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd",
            },

            "mustard_bottle": {
                "asset_path": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Props/YCB/Axis_Aligned/006_mustard_bottle.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(270, self.scanning_params['base_rotation'][1], self.scanning_params['base_rotation'][2]),
                },
            },

            "tuna_fish_can": {
                "asset_path": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Props/YCB/Axis_Aligned/007_tuna_fish_can.usd",
            },

            "pudding_box": {
                "asset_path": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Props/YCB/Axis_Aligned/008_pudding_box.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(self.scanning_params['base_rotation'][0], 90, self.scanning_params['base_rotation'][2]),
                },
            },

            "gelatin_box": {
                "asset_path": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Props/YCB/Axis_Aligned/009_gelatin_box.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(self.scanning_params['base_rotation'][0], 90, self.scanning_params['base_rotation'][2]),
                },
            },

            "potted_meat_can": {
                "asset_path": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Props/YCB/Axis_Aligned/010_potted_meat_can.usd",
            },

            "banana": {
                "asset_path": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Props/YCB/Axis_Aligned/011_banana.usd",
            },

            "pitcher": {
                "asset_path": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Props/YCB/Axis_Aligned/019_pitcher_base.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(270, self.scanning_params['base_rotation'][1], self.scanning_params['base_rotation'][2]),
                },
            },

            "bleach_cleanser": {
                "asset_path": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Props/YCB/Axis_Aligned/021_bleach_cleanser.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(270, self.scanning_params['base_rotation'][1], self.scanning_params['base_rotation'][2]),
                },
            },

            "bowl": {
                "asset_path": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Props/YCB/Axis_Aligned/024_bowl.usd",
            },

            "mug": {
                "asset_path": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Props/YCB/Axis_Aligned/025_mug.usd",
            },

            "power_drill": {
                "asset_path": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Props/YCB/Axis_Aligned/035_power_drill.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(270, self.scanning_params['base_rotation'][1], self.scanning_params['base_rotation'][2]),
                },
            },

            "wood_block": {
                "asset_path": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Props/YCB/Axis_Aligned/036_wood_block.usd",
            },

            "scissors": {
                "asset_path": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Props/YCB/Axis_Aligned/037_scissors.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(self.scanning_params['base_rotation'][0], 90, self.scanning_params['base_rotation'][2]),
                },
            },

            "marker": {
                "asset_path": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Props/YCB/Axis_Aligned/040_large_marker.usd",
                "overrides": {
                    "object_scale": 5.0,
                },
            },

            "large_clamp": {
                "asset_path": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Props/YCB/Axis_Aligned/051_large_clamp.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(self.scanning_params['base_rotation'][0], 90, self.scanning_params['base_rotation'][2]),
                },
            },

            # ======================
            # Physical AI dataset
            # ======================

            "eyeglasses": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/accessory_eyeglasses_a/sm_accessory_eyeglasses_a01_simready_01.usd",
            },

            "open_book": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/book_open_a/sm_book_open_a01_simready_01.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(45, self.scanning_params['base_rotation'][1], self.scanning_params['base_rotation'][2]),
                },
            },

            "spray_bottle": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/cleaning_bottle_spray_a/sm_cleaning_bottle_spray_a01_simready_01.usd",
                "overrides": {
                    "base_position": Gf.Vec3d(self.scanning_params['base_position'][0], self.scanning_params['base_position'][1], 1.3),
                },
            },

            "cleaning_bucket": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/cleaning_bucket_a/sm_cleaning_bucket_iron_a01_simready_01.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(90, self.scanning_params['base_rotation'][1], self.scanning_params['base_rotation'][2]),
                },
            },

            "container_bottle": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/container_bottle_a/sm_container_bottle_clay_a01_simready_01.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(90, self.scanning_params['base_rotation'][1], self.scanning_params['base_rotation'][2]),
                    "object_scale": 1.0,
                },
            },

            "container_tin": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/container_can_tin_open_a/sm_container_can_tin_open_a01_simready_01.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(90, self.scanning_params['base_rotation'][1], self.scanning_params['base_rotation'][2]),
                },
            },

            "vial": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/container_vial_glass_a/sm_container_vial_glass_a01_simready_01.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(90, self.scanning_params['base_rotation'][1], self.scanning_params['base_rotation'][2]),
                    "object_scale": 3.0,
                },
            },

            "lighting_candles": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/lighting_candles_tea_a/sm_lighting_candles_tea_a02_simready_01.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(90, self.scanning_params['base_rotation'][1], self.scanning_params['base_rotation'][2]),
                    "object_scale": 5.0,
                },
            },

            "magazine_stack": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/magazines_stack_a/sm_magazines_stack_a01_simready_01.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(45, self.scanning_params['base_rotation'][1], self.scanning_params['base_rotation'][2]),
                },
            },

            "wooden_boards": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/misc_boards_wooden_a/sm_misc_boards_wooden_a02_simready_01.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(45, self.scanning_params['base_rotation'][1], self.scanning_params['base_rotation'][2]),
                    "object_scale": 1.0,
                },
            },

            "paint_container_bottle": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/paint_container_bottle_a/sm_paint_container_bottle_blue_a01_simready_01.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(90, self.scanning_params['base_rotation'][1], self.scanning_params['base_rotation'][2]),
                    "object_scale": 3.0,
                },
            },

            "paint_container_jar": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/paint_container_jar_a/sm_paint_container_jar_green_a01_simready_01.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(90, self.scanning_params['base_rotation'][1], self.scanning_params['base_rotation'][2]),
                    "object_scale": 3.0,
                },
            },

            "paint_container_spraycan": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/paint_container_spraycan_a/sm_paint_container_spraycan_a01_simready_01.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(90, self.scanning_params['base_rotation'][1], self.scanning_params['base_rotation'][2]),
                },
            },

            "paint_container_tube_small": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/paint_container_tube_a/sm_paint_container_tube_a01_simready_01.usd",
                "overrides": {
                    "object_scale": 5.0,
                },
            },

            "paint_container_tube": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/paint_container_tube_a/sm_paint_container_tube_b01_simready_01.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(90, self.scanning_params['base_rotation'][1], self.scanning_params['base_rotation'][2]),
                    "object_scale": 5.0,
                },
            },

            "paint_container_tube_large": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/paint_container_tube_a/sm_paint_container_tube_e01_simready_01.usd",
                "overrides": {
                    "object_scale": 5.0,
                },
            },

            "paint_brush": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/paint_supplies_brushes_a/sm_paint_supplies_brushes_a03_simready_01.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(self.scanning_params['base_rotation'][0], 90, self.scanning_params['base_rotation'][2]),
                },
            },

            "paint_brush_large": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/paint_supplies_brushes_a/sm_paint_supplies_brushes_a04_simready_01.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(self.scanning_params['base_rotation'][0], 90, self.scanning_params['base_rotation'][2]),
                },
            },

            "pallet_knife": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/paint_supplies_palletknife_a/sm_paint_supplies_palletknife_a01_simready_01.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(self.scanning_params['base_rotation'][0], 90, self.scanning_params['base_rotation'][2]),
                    "object_scale": 3.0,
                },
            },

            "pallet_knife_large": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/paint_supplies_palletknife_a/sm_paint_supplies_palletknife_a05_simready_01.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(self.scanning_params['base_rotation'][0], 90, self.scanning_params['base_rotation'][2]),
                    "object_scale": 3.0,
                },
            },

            "spraygun": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/paint_supplies_spraygun_a/sm_paint_supplies_spraygun_a01_simready_01.usd",
                "overrides": {
                    "base_position": Gf.Vec3d(self.scanning_params['base_position'][0], self.scanning_params['base_position'][1], 1.3),
                },
            },

            "paint_thinner": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/paint_supplies_thinner_a/sm_paint_supplies_thinner_a01_simready_01.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(90, self.scanning_params['base_rotation'][1], self.scanning_params['base_rotation'][2]),
                },
            },

            "battery": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/power_battery_9v_a/sm_power_battery_9v_a01_simready_01.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(90, self.scanning_params['base_rotation'][1], self.scanning_params['base_rotation'][2]),
                    "object_scale": 4.0,
                },
            },

            "battery_case": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/power_battery_9v_case_a/sm_power_battery_9v_case_a01_simready_01.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(90, self.scanning_params['base_rotation'][1], self.scanning_params['base_rotation'][2]),
                    "object_scale": 4.0,
                },
            },

            "chalk_stick": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/stationery_chalk_sticks_a/sm_stationery_chalk_sticks_a03_simready_01.usd",
                "overrides": {
                    "object_scale": 5.0,
                },
            },

            "pen": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/stationery_pen_metal_a/sm_stationery_pen_metal_a01_simready_01.usd",
                "overrides": {
                    "base_position": Gf.Vec3d(self.scanning_params['base_position'][0], -0.115, self.scanning_params['base_position'][2]),
                    "object_scale": 4.0,
                },
            },

            "cork": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/stationery_tacks_cork_a/sm_stationery_tacks_cork_a01_simready_01.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(90, self.scanning_params['base_rotation'][1], self.scanning_params['base_rotation'][2]),
                    "object_scale": 5.0,
                },
            },

            "clamp": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/tool_clamp_c/sm_tool_clamp_c01_simready_01.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(90, self.scanning_params['base_rotation'][1], self.scanning_params['base_rotation'][2]),
                },
            },

            "boxcutter": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/tool_cutting_boxcutter_a/sm_tool_cutting_boxcutter_a01_simready_01.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(self.scanning_params['base_rotation'][0], 90, self.scanning_params['base_rotation'][2]),
                    "object_scale": 0.3,
                },
            },

            "hammer": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/tool_hammers_a/sm_tool_hammer_a01_simready_01.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(90, 90, self.scanning_params['base_rotation'][2]),
                },
            },

            "duct_tape": {
                "asset_path": "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/main/Props/general/HandManipulation/tool_tape_duct_a/sm_tool_tape_duct_a02_simready_01.usd",
                "overrides": {
                    "base_rotation": Gf.Vec3f(90, self.scanning_params['base_rotation'][1], self.scanning_params['base_rotation'][2]),
                },
            },
        }

        # Training mode parameters
        self.training_params = {
            'frames_captured': 0,
            'frames_total': 0,      # Will be set based on texture files
            'plane_size': 0.25,     # Larger plane size (2x the calibration board), adjusted from 1.0 to 0.25 to match calibration board size
        }

        # Ablation mode parameters
        self.ablation_params = {
            'material_type': 'Baseline',  # Options: 'Baseline', 'Reflective', 'Metallic', 'AO_to_diffuse_0'
            'lighting_setup': 'Baseline',  # Options: 'No_Ambient', 'Baseline', 'One_Ambient', 'Two_Ambient'
            'frames_captured': 0,
            'frames_total': 0,  # Will be set based on texture files
        }

    def _get_calibration_positions(self):
        # Base positions for calibration board
        base_x_pos = 1.0
        base_y_pos = 0.0
        base_z_pos = 1.5
        x_offset_pos = 0.6 # Adjusted from 0.1 to 0.6 to ensure depth of field is captured
        y_offset_pos = 0.1 # Adjusted from 0.1 to 0.075 to ensure camera captures circular pattern
        z_offset_pos = 0.1

        # return [
        #     # Base x distance
        #     (base_x_pos, base_y_pos, base_z_pos),
        #     (base_x_pos, base_y_pos, base_z_pos + z_offset_pos),
        #     (base_x_pos, base_y_pos + y_offset_pos, base_z_pos - z_offset_pos),
        #     (base_x_pos, base_y_pos - y_offset_pos, base_z_pos + z_offset_pos),
        #     (base_x_pos, base_y_pos - y_offset_pos, base_z_pos - z_offset_pos),
        #     (base_x_pos, base_y_pos + y_offset_pos, base_z_pos + z_offset_pos),
        #     # Backward offset x distance
        #     (base_x_pos + x_offset_pos, base_y_pos, base_z_pos),
        #     (base_x_pos + x_offset_pos, base_y_pos, base_z_pos + z_offset_pos),
        #     (base_x_pos + x_offset_pos, base_y_pos + y_offset_pos, base_z_pos - z_offset_pos),
        #     (base_x_pos + x_offset_pos, base_y_pos - y_offset_pos, base_z_pos + z_offset_pos),
        #     (base_x_pos + x_offset_pos, base_y_pos - y_offset_pos, base_z_pos - z_offset_pos),
        #     (base_x_pos + x_offset_pos, base_y_pos + y_offset_pos, base_z_pos + z_offset_pos),
        #     # Forward offset x distance
        #     (base_x_pos - x_offset_pos, base_y_pos, base_z_pos),
        #     (base_x_pos - x_offset_pos, base_y_pos, base_z_pos + z_offset_pos),
        #     (base_x_pos - x_offset_pos, base_y_pos + y_offset_pos, base_z_pos - z_offset_pos),
        #     (base_x_pos - x_offset_pos, base_y_pos - y_offset_pos, base_z_pos + z_offset_pos),
        #     (base_x_pos - x_offset_pos, base_y_pos - y_offset_pos, base_z_pos - z_offset_pos),
        #     (base_x_pos - x_offset_pos, base_y_pos + y_offset_pos, base_z_pos + z_offset_pos)
        # ]

        return [
            (base_x_pos, base_y_pos, base_z_pos),   # Base position
            (base_x_pos, base_y_pos, base_z_pos),
            (base_x_pos, base_y_pos, base_z_pos),
            (base_x_pos, base_y_pos, base_z_pos),
            (base_x_pos, base_y_pos, base_z_pos),
            (base_x_pos, base_y_pos, base_z_pos + z_offset_pos),
            (base_x_pos, base_y_pos, base_z_pos + z_offset_pos),
            (base_x_pos, base_y_pos, base_z_pos - z_offset_pos),
            (base_x_pos, base_y_pos - y_offset_pos, base_z_pos),
            (base_x_pos, base_y_pos - y_offset_pos, base_z_pos),
            (base_x_pos, base_y_pos - y_offset_pos, base_z_pos),
            (base_x_pos, base_y_pos + y_offset_pos, base_z_pos),
            (base_x_pos, base_y_pos + y_offset_pos, base_z_pos),
            (base_x_pos, base_y_pos + y_offset_pos, base_z_pos),
            (base_x_pos, base_y_pos - y_offset_pos, base_z_pos + z_offset_pos),
            (base_x_pos, base_y_pos - y_offset_pos, base_z_pos - z_offset_pos),
            (base_x_pos, base_y_pos + y_offset_pos, base_z_pos + z_offset_pos),
            (base_x_pos, base_y_pos + y_offset_pos, base_z_pos - z_offset_pos)
        ]

    def _get_calibration_orientations(self):
        # Base rotations for calibration board
        base_x_rot = 90
        base_y_rot = 0
        base_z_rot = 90
        x_offset_rot_sm = 15
        z_offset_rot_sm = 15
        x_offset_rot_lg = 30
        z_offset_rot_lg = 30

        # return [
        #     # Base plane rotation
        #     (base_x_rot, base_y_rot, base_z_rot),
        #     (base_x_rot + x_offset_rot_sm, base_y_rot, base_z_rot - z_offset_rot_sm),
        #     (base_x_rot - x_offset_rot_sm, base_y_rot, base_z_rot + z_offset_rot_sm),
        #     (base_x_rot + x_offset_rot_lg, base_y_rot, base_z_rot - z_offset_rot_sm),
        #     (base_x_rot - x_offset_rot_lg, base_y_rot, base_z_rot + z_offset_rot_lg),
        #     (base_x_rot + x_offset_rot_lg, base_y_rot, base_z_rot + z_offset_rot_lg),
        #     # Backward plane rotation 
        #     (base_x_rot, base_y_rot, base_z_rot),
        #     (base_x_rot + x_offset_rot_sm, base_y_rot, base_z_rot - z_offset_rot_sm),
        #     (base_x_rot - x_offset_rot_sm, base_y_rot, base_z_rot + z_offset_rot_sm),
        #     (base_x_rot + x_offset_rot_lg, base_y_rot, base_z_rot - z_offset_rot_sm),
        #     (base_x_rot - x_offset_rot_lg, base_y_rot, base_z_rot + z_offset_rot_lg),
        #     (base_x_rot + x_offset_rot_lg, base_y_rot, base_z_rot + z_offset_rot_lg),
        #     # Forward plane rotation
        #     (base_x_rot, base_y_rot, base_z_rot),
        #     (base_x_rot + x_offset_rot_sm, base_y_rot, base_z_rot - z_offset_rot_sm),
        #     (base_x_rot - x_offset_rot_sm, base_y_rot, base_z_rot + z_offset_rot_sm),
        #     (base_x_rot + x_offset_rot_lg, base_y_rot, base_z_rot - z_offset_rot_sm),
        #     (base_x_rot - x_offset_rot_lg, base_y_rot, base_z_rot + z_offset_rot_lg),
        #     (base_x_rot + x_offset_rot_lg, base_y_rot, base_z_rot + z_offset_rot_lg)
        # ]

        return [
            (base_x_rot, base_y_rot, base_z_rot),
            (base_x_rot, base_y_rot, base_z_rot - z_offset_rot_sm),
            (base_x_rot, base_y_rot, base_z_rot + z_offset_rot_sm),
            (base_x_rot, base_y_rot, base_z_rot - z_offset_rot_lg),
            (base_x_rot, base_y_rot, base_z_rot + z_offset_rot_lg),
            (base_x_rot - x_offset_rot_sm, base_y_rot, base_z_rot),
            (base_x_rot - x_offset_rot_sm, base_y_rot, base_z_rot - z_offset_rot_lg),
            (base_x_rot + x_offset_rot_sm, base_y_rot, base_z_rot),
            (base_x_rot, base_y_rot, base_z_rot),
            (base_x_rot - x_offset_rot_sm, base_y_rot, base_z_rot - z_offset_rot_sm),
            (base_x_rot + x_offset_rot_sm, base_y_rot, base_z_rot - z_offset_rot_sm),
            (base_x_rot, base_y_rot, base_z_rot),
            (base_x_rot - x_offset_rot_sm, base_y_rot, base_z_rot + z_offset_rot_sm),
            (base_x_rot + x_offset_rot_sm, base_y_rot, base_z_rot + z_offset_rot_sm),
            (base_x_rot - x_offset_rot_lg, base_y_rot, base_z_rot - z_offset_rot_sm),
            (base_x_rot - x_offset_rot_lg, base_y_rot, base_z_rot + z_offset_rot_lg),
            (base_x_rot + x_offset_rot_lg, base_y_rot, base_z_rot - z_offset_rot_sm),
            (base_x_rot + x_offset_rot_lg, base_y_rot, base_z_rot + z_offset_rot_lg)
        ]

    def setup_scene(self):
        """Setup the simulation scene based on selected mode"""
        self.simulation_context = SimulationContext.instance()
        world = self.get_world()
        world.scene.add_default_ground_plane()
        stage = omni.usd.get_context().get_stage()

        # Setup camera and projector (shared between modes)
        if hasattr(self, 'use_real_world_params') and self.use_real_world_params:
            print("Using real-world system parameters for camera and projector")
        else:
            print("Using simulation parameters for camera and projector")
        self._setup_camera()
        self._setup_projector(stage)
        
        # Load and setup texture files
        self._load_texture_files()
        
        if self.mode == "calibrate":
            self.setup_calibration_board(stage)
            self.setup_background_plane(stage)
        elif self.mode == "scan":
            self._setup_scan_object(stage)
            self._apply_uniform_material_to_scan_object(stage)
            self.setup_background_plane(stage)
        elif self.mode == "training":
            self.setup_training_plane(stage)
            self.setup_background_plane(stage)
        elif self.mode == "ablation":
            self.setup_ablation_sphere(stage)
            self.setup_background_plane(stage)
            self.setup_ablation_lighting(stage)
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be either 'calibrate', 'scan', 'training', or 'ablation'")

        # RT Subframes setting for rendering correction and ray tracing rendering setup
        # https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_getting_started.html
        self.rt_subframes = 0 # Default value is 0, increasing will help with rendering errors at the cost of dramatically slowing data generation
        carb_settings = carb.settings.get_settings()
        carb_settings.set("/omni/replicator/RTSubframes", self.rt_subframes)
        carb_settings.set("/rtx/directLighting/sampledLighting/enabled", False) # Enables higher fringe image quality
        carb_settings.set("/rtx/shadows/enabled", False)

    def _setup_camera(self):
        """Setup the camera with appropriate parameters"""
        calculated_dynamic_frequency = int(1 / (self.texture_update_interval))
        self.camera = Camera(
            prim_path="/World/Camera",
            position=np.array([-1.2, -0.125, 1.5]),
            frequency=calculated_dynamic_frequency,
            resolution=(self.real_world_params['cam_width'], self.real_world_params['cam_height']) if self.use_real_world_params else (960, 960)
        )
        # Add camera to the world
        self.camera.initialize()

        # Apply real-world parameters if enabled else use default values
        if self.use_real_world_params:
            # Get camera parameters from the real-world model
            [focal_length, focus_distance, lens_aperture, 
            horizontal_aperture, vertical_aperture, clipping_range] = camera_rw2dt(
                self.real_world_params['cam_mat_path'],
                self.real_world_params['cam_height'],
                self.real_world_params['cam_width'],
                self.real_world_params['pixel_size'],
                self.real_world_params['f_stop'],
                self.real_world_params['focus_distance']
            )
            
            # Apply the calculated parameters to the camera
            self.camera.set_focal_length(focal_length)
            self.camera.set_focus_distance(focus_distance)
            self.camera.set_lens_aperture(lens_aperture)
            self.camera.set_vertical_aperture(vertical_aperture)
            self.camera.set_horizontal_aperture(horizontal_aperture)
            self.camera.set_clipping_range(clipping_range[0], clipping_range[1])
            
            print("Applied real-world camera parameters")

    def _setup_projector(self, stage):
        """Setup the projector light"""
        self.light_prim_path = "/World/Projector"
        light_prim = UsdLux.RectLight.Define(stage, self.light_prim_path)
        light_prim.AddTranslateOp().Set(Gf.Vec3f(-1.2, 0, 1.4))
        light_prim.AddRotateXYZOp().Set(Gf.Vec3f(270, 180, 90))

         # Apply scaling based on whether real-world parameters are used
        if self.use_real_world_params:
            # Get projector parameters from the real-world model
            [x_scale, y_scale] = projector_rw2dt(
                self.real_world_params['proj_mat_path'],
                self.real_world_params['fringe_pattern_height'],
                self.real_world_params['fringe_pattern_width'],
                scaling_factor=0.5
            )
            
            # Apply the calculated scaling to the projector
            light_prim.AddScaleOp().Set(Gf.Vec3f(x_scale, y_scale, 1))
            # For real-world modeling, width and height must be set to 1
            light_prim.GetHeightAttr().Set(1.0)
            light_prim.GetWidthAttr().Set(1.0)
            
            print("Applied real-world projector parameters")
        else:
            # Use simulation values
            light_prim.AddScaleOp().Set(Gf.Vec3f(1, 1, 1))
            # Ensure width and height are set to maintain fringe pattern aspect ratio
            light_prim.GetHeightAttr().Set(0.625)
            light_prim.GetWidthAttr().Set(0.5)

        light_prim.GetIntensityAttr().Set(40.0) # Increased from 20.0 to 40.0 for better fringe pattern contrast
        
        shaping_api = UsdLux.ShapingAPI(light_prim)
        shaping_api.CreateShapingConeAngleAttr(180.0)
        
        prim = stage.GetPrimAtPath(self.light_prim_path)
        if prim:
            is_projector_attr = prim.GetAttribute("isProjector")
            if not is_projector_attr:
                is_projector_attr = prim.CreateAttribute("isProjector", Sdf.ValueTypeNames.Bool)
            is_projector_attr.Set(True)

    def setup_ablation_lighting(self, stage):
        """Setup lighting for the ablation study"""
        # Control default ambient light (sphere light)
        sphere_light_path = "/World/defaultGroundPlane/SphereLight"
        sphere_light = stage.GetPrimAtPath(sphere_light_path)
        if sphere_light:
            intensity_attr = sphere_light.GetAttribute("intensity")
            if intensity_attr:
                # Set intensity to 0 for "No_Ambient" lighting setup
                if self.ablation_params['lighting_setup'] == 'No_Ambient':
                    intensity_attr.Set(0.0)
        
        # Create right ambient light
        right_light_path = "/World/RightAmbientLight"
        right_light = UsdLux.RectLight.Define(stage, right_light_path)
        right_light.AddTranslateOp().Set(Gf.Vec3f(0, 0.6, 1.4))
        right_light.AddRotateXYZOp().Set(Gf.Vec3f(-45, -90, 0))
        right_light.AddScaleOp().Set(Gf.Vec3f(1, 1, 1))
        right_light.GetHeightAttr().Set(1)
        right_light.GetWidthAttr().Set(1)
        
        # Create left ambient light
        left_light_path = "/World/LeftAmbientLight"
        left_light = UsdLux.RectLight.Define(stage, left_light_path)
        left_light.AddTranslateOp().Set(Gf.Vec3f(0, -0.6, 1.4))
        left_light.AddRotateXYZOp().Set(Gf.Vec3f(45, -90, 0))
        left_light.AddScaleOp().Set(Gf.Vec3f(1, 1, 1))
        left_light.GetHeightAttr().Set(1)
        left_light.GetWidthAttr().Set(1)
        
        # Set light intensities based on lighting setup
        right_intensity = 0.0
        left_intensity = 0.0
        
        if self.ablation_params['lighting_setup'] == 'One_Ambient':
            right_intensity = 100000.0
        elif self.ablation_params['lighting_setup'] == 'Two_Ambient':
            right_intensity = 100000.0
            left_intensity = 100000.0
        
        right_light.GetIntensityAttr().Set(right_intensity)
        left_light.GetIntensityAttr().Set(left_intensity)

    def setup_ablation_sphere(self, stage):
        """Setup the ablation sphere with specified properties"""
        # Define the sphere
        sphere_path = "/World/AblationSphere"
        sphere_prim = stage.GetPrimAtPath(sphere_path)
        if not sphere_prim:
            sphere_prim = UsdGeom.Sphere.Define(stage, sphere_path)

        # Set radius to 0.5 (this will result in extent [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
        radius = 0.5
        sphere_prim.GetExtentAttr().Set([(-radius, -radius, -radius), (radius, radius, radius)])
        sphere_prim.GetRadiusAttr().Set(radius)

        prim = stage.GetPrimAtPath(self.light_prim_path)
        if prim:
            is_projector_attr = prim.GetAttribute("isProjector")
            if not is_projector_attr:
                is_projector_attr = prim.CreateAttribute("isProjector", Sdf.ValueTypeNames.Bool)
            is_projector_attr.Set(True)

        # Enable refinement override and set level to 2 on the prim
        prim = stage.GetPrimAtPath(sphere_path)
        if prim: 
            refinement_enable_attr = prim.GetAttribute("refinementEnableOverride")
            if not refinement_enable_attr:
                refinement_enable_attr = prim.CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool)
            refinement_enable_attr.Set(True)

            refinement_attr = prim.GetAttribute("refinementLevel")
            if not refinement_attr:
                refinement_attr = prim.CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int)
            refinement_attr.Set(2)
        
        # Set transform
        transform_api = UsdGeom.Xformable(sphere_prim)
        transform_api.AddTranslateOp().Set(Gf.Vec3d(0.6, 0.0, 1.5))  # Position as specified
        transform_api.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 0))        # Orientation as specified
        transform_api.AddScaleOp().Set(Gf.Vec3d(0.4, 0.4, 0.4))      # Scale as specified
        
        # Create different materials based on the specified material type
        self._setup_ablation_material(stage, sphere_prim)

    def _setup_ablation_material(self, stage, sphere_prim):
        """Setup material for the ablation sphere based on specified material type"""
        material_path = Sdf.Path("/World/AblationSphere/Material")
        material = UsdShade.Material.Define(stage, material_path)
        
        if self.ablation_params['material_type'] == 'Reflective':
            # Create OmniGlass material
            shader = UsdShade.Shader.Define(stage, material_path.AppendPath("Shader"))
            shader.CreateIdAttr("OmniGlass")
            shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
            shader.SetSourceAsset("OmniGlass.mdl", "mdl")
            shader.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
            
            # Connect shader to material
            material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
            material.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
            material.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")

        elif self.ablation_params['material_type'] == 'Metallic':
            # Create OmniPBR material with metallic properties
            shader = UsdShade.Shader.Define(stage, material_path.AppendPath("Shader"))
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
            shader.SetSourceAsset("OmniPBR.mdl", "mdl")
            shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
            
            # Connect shader to material
            material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
            material.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
            material.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
            
            # Set metallic parameters
            shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(1.0)
            shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.2)
            shader.CreateInput("base_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.95, 0.95, 0.95))

        else:
            # Create OmniPBR material with different settings
            shader = UsdShade.Shader.Define(stage, material_path.AppendPath("Shader"))
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
            shader.SetSourceAsset("OmniPBR.mdl", "mdl")
            shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
            
            # Connect shader to material
            material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
            material.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
            material.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
            
            # Set material properties based on type
            shader.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(0.15)
            shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(1.0)
            
            # Set AO to diffuse based on material type
            ao_value = 0.0 if self.ablation_params['material_type'] == 'AO_to_diffuse_0' else 0.95
            shader.CreateInput("ao_to_diffuse", Sdf.ValueTypeNames.Float).Set(ao_value)
        
        # Bind material to the sphere
        UsdShade.MaterialBindingAPI(sphere_prim).Bind(material)

    def setup_training_plane(self, stage):
        """Setup the training plane geometry and material"""
        # Define the training plane as a single-sided plane
        plane_path = "/World/TrainingPlane"
        plane_prim = stage.GetPrimAtPath(plane_path)
        if not plane_prim:
            plane_prim = UsdGeom.Mesh.Define(stage, plane_path)
        
        # Define the mesh geometry
        size = self.training_params['plane_size']  # Adjust size as needed
        points = [
            (-size, -size, 0),
            (size, -size, 0),
            (size, size, 0),
            (-size, size, 0)
        ]
        
        face_vertex_counts = [4]
        face_vertex_indices = [0, 1, 2, 3]
        normals = [(0, 0, 1)] * 4
        uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
        
        # Set mesh attributes
        plane_prim.CreatePointsAttr(points)
        plane_prim.CreateNormalsAttr(normals)
        plane_prim.CreateFaceVertexCountsAttr(face_vertex_counts)
        plane_prim.CreateFaceVertexIndicesAttr(face_vertex_indices)
        
        # Set UV coordinates
        primvars_api = UsdGeom.PrimvarsAPI(plane_prim)
        texCoords = primvars_api.CreatePrimvar("st", 
                                            Sdf.ValueTypeNames.TexCoord2fArray,
                                            UsdGeom.Tokens.vertex)
        texCoords.Set(uvs)
        
        # Set subdivision scheme and extent
        plane_prim.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        plane_prim.CreateExtentAttr().Set([(-size, -size, -.001), (size, size, .001)])
        
        # Set initial transform to match calibration board
        transform_api = UsdGeom.Xformable(plane_prim)
        transform_api.AddTranslateOp().Set(Gf.Vec3d(*self.calibration_params['positions'][0]))
        transform_api.AddRotateXYZOp().Set(Gf.Vec3f(*self.calibration_params['orientations'][0]))
        transform_api.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))
        
        # Create and setup material
        material_path = Sdf.Path("/World/TrainingPlane/Material")
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path.AppendPath("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset("OmniPBR.mdl", "mdl")
        shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        
        # Connect shader to material
        material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        material.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        material.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        
        # Set material properties
        shader.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(0.15)
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("ao_to_diffuse", Sdf.ValueTypeNames.Float).Set(0.95)

        # Bind material to the plane
        UsdShade.MaterialBindingAPI(plane_prim).Bind(material)

    def _scan_object_override(self, object_name):
        """Applies object specific overrides to the initial scanning_params"""
        if object_name not in self.SCAN_OBJECTS:
            raise ValueError(f"Unknown scan object: {object_name}")

        cfg = self.SCAN_OBJECTS[object_name]

        # Apply per-object overrides
        for key, value in cfg.get("overrides", {}).items():
            if key not in self.scanning_params:
                raise KeyError(f"Invalid scanning param override: {key}")
            self.scanning_params[key] = value

        # Store paths for future reference
        self.scanning_params['asset_path'] = cfg["asset_path"]

    def _setup_scan_object(self, stage):
        """Setup the USD object for scanning with proper scale and position"""
        # Create a parent Xform for the scan object
        parent_path = "/World/Scan"
        parent_prim = stage.DefinePrim(parent_path, "Xform")
        
        # Load the USD object as a reference under the parent
        scan_object = self.scanning_params['scan_object']
        self._scan_object_override(scan_object)
        object_path = f"{parent_path}/Target"
        asset_path = self.scanning_params['asset_path']
        object_prim = stage.DefinePrim(object_path, "Xform")
        object_prim.GetReferences().AddReference(asset_path)
        
        # Set up transforms for parent (position and rotation)
        parent_xform = UsdGeom.Xformable(parent_prim)
        parent_xform_ops = parent_xform.GetOrderedXformOps()
        
        # Setup parent transform operations
        translate_op = next((op for op in parent_xform_ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
        rotate_op = next((op for op in parent_xform_ops if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ), None)
        
        if not translate_op:
            translate_op = parent_xform.AddTranslateOp()
        if not rotate_op:
            rotate_op = parent_xform.AddRotateXYZOp()
        
        # Setup initial position and rotation for object
        translate_op.Set(self.scanning_params['base_position'])
        rotate_op.Set(self.scanning_params['base_rotation'])
        
        # Setup transforms for the object (scale)
        object_xform = UsdGeom.Xformable(object_prim)
        object_xform_ops = object_xform.GetOrderedXformOps()
        
        # Setup object transform operations
        scale_op = next((op for op in object_xform_ops if op.GetOpType() == UsdGeom.XformOp.TypeScale), None)
        if not scale_op:
            scale_op = object_xform.AddScaleOp()
        
        scale_op.Set(Gf.Vec3d(
            self.scanning_params['object_scale'],
            self.scanning_params['object_scale'],
            self.scanning_params['object_scale']
        ))
        
        # Store paths for future reference
        self.scanning_params['parent_path'] = parent_path
        self.scanning_params['object_path'] = object_path
        
        print(f"Scan object setup complete at {object_path} for {scan_object}")
        print(f"Initial position: {self.scanning_params['base_position']}")
        print(f"Initial rotation: {self.scanning_params['base_rotation']}")
        print(f"Scale: {self.scanning_params['object_scale']}")

    def _apply_uniform_material_to_scan_object(self, stage):
        """
        Recursively traverse all meshes under the scan object and bind them 
        to a uniform material identical to the background plane material.
        """
        # Create the uniform material (identical to background plane)
        material_path = Sdf.Path("/World/UniformMaterial")
        
        # Check if material already exists, if not create it
        material_prim = stage.GetPrimAtPath(material_path)
        if not material_prim or not material_prim.IsValid():
            material = UsdShade.Material.Define(stage, material_path)
            shader = UsdShade.Shader.Define(stage, material_path.AppendPath("Shader"))
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
            shader.SetSourceAsset("OmniPBR.mdl", "mdl")
            shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
            
            # Connect shader to material
            material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
            material.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
            material.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
            
            # Set material properties (identical to background plane)
            shader.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(0.15)
            shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(1.0)
            shader.CreateInput("ao_to_diffuse", Sdf.ValueTypeNames.Float).Set(0.95)
        else:
            material = UsdShade.Material(material_prim)
        
        # Recursively find and bind all meshes under /World/Scan
        scan_path = "/World/Scan"
        scan_prim = stage.GetPrimAtPath(scan_path)
        
        if not scan_prim or not scan_prim.IsValid():
            print(f"Warning: Scan object not found at {scan_path}")
            return
        
        mesh_count = 0
        
        def traverse_and_bind(prim):
            """Recursive function to traverse hierarchy and bind meshes"""
            nonlocal mesh_count
            
            # Check if this prim is a Mesh
            if prim.IsA(UsdGeom.Mesh):
                # Bind the material to this mesh
                UsdShade.MaterialBindingAPI(prim).Bind(material)
                mesh_count += 1
                print(f"  Bound material to mesh: {prim.GetPath()}")
            
            # Recursively process all children
            for child in prim.GetChildren():
                traverse_and_bind(child)
        
        # Start traversal from scan object
        print(f"Applying uniform material to scan object...")
        traverse_and_bind(scan_prim)
        print(f"Successfully bound material to {mesh_count} mesh(es)")

    def _update_scan_object_pose(self):
        """Update the scan object's rotation around its vertical axis"""
        stage = omni.usd.get_context().get_stage()
        parent_prim = stage.GetPrimAtPath(self.scanning_params['parent_path'])
        
        if parent_prim:
            transform_api = UsdGeom.Xformable(parent_prim)
            xform_ops = transform_api.GetOrderedXformOps()
            
            # Find existing rotation operation
            rotate_op = next((op for op in xform_ops if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ), None)
            
            if not rotate_op:
                # If no rotation operation exists, create one
                rotate_op = transform_api.AddRotateXYZOp()
            
            # Update rotation rotate around one axis while keeping other axes at their initial rotation
            current_angle = float(self.scanning_params['current_angle'])
            axis = self.scanning_params.get("rotation_axis")
            if axis == "x":
                rotate_op.Set(Gf.Vec3f(current_angle, self.scanning_params['base_rotation'][1], self.scanning_params['base_rotation'][2]))
                pass
            elif axis == "y":
                rotate_op.Set(Gf.Vec3f(self.scanning_params['base_rotation'][0], current_angle, self.scanning_params['base_rotation'][2]))
                pass
            elif axis == "z":
                rotate_op.Set(Gf.Vec3f(self.scanning_params['base_rotation'][0], self.scanning_params['base_rotation'][1], current_angle))
                pass
            else:
                raise ValueError(f"Invalid rotation_axis: {axis}")
            print(f"Updated rotation axis {axis} to {current_angle} degrees")

    def _load_texture_files(self):
        """Load texture files from directory"""
        texture_directory = "C:/Users/oadam/downloads/full_fringe_patterns" # Modify texture directory to folder of n-step fringe patterns
        self.texture_files = sorted(
            [os.path.join(texture_directory, f) for f in os.listdir(texture_directory) if (f.endswith('.bmp') or f.endswith('.png'))],
            key=natural_sort_key
        )
        if not self.texture_files:
            raise FileNotFoundError(f"No texture files found in directory: {texture_directory}")
        
        # Set frames per pose/angle based on number of texture files
        if self.mode == "calibrate":
            self.calibration_params['frames_per_pose'] = len(self.texture_files)
        elif self.mode == "scan":
            self.scanning_params['frames_per_angle'] = len(self.texture_files)
        elif self.mode == "ablation":
            self.ablation_params['frames_total'] = len(self.texture_files)
        else:  # Training mode
            self.training_params['frames_total'] = len(self.texture_files)
        
        self.set_projector_texture(self.texture_files[0])
        print(f"{len(self.texture_files)} texture files loaded.")

    def update_texture_callback(self, step_size):
        """Update texture and capture frames based on current mode"""
        current_time = self.simulation_context.current_time
        
        if current_time - self.last_texture_update_time >= self.texture_update_interval:
            self.frames_since_update += 1
            
            if self.frames_since_update >= self.frame_skip:
                self.capture_camera_frames()
                
                if self.mode == "calibrate":
                    self._handle_calibration_update()
                elif self.mode == "scan":
                    self._handle_scanning_update()
                elif self.mode == "ablation":
                    self._handle_ablation_update()
                else: # Training mode
                    self._handle_training_update()
                
                # Update texture
                self.current_texture_index = (self.current_texture_index + 1) % len(self.texture_files)
                self.set_projector_texture(self.texture_files[self.current_texture_index])
                self.last_texture_update_time = current_time
                self.frames_since_update = 0

    def _handle_ablation_update(self):
        """Handle updates for ablation mode"""
        self.ablation_params['frames_captured'] += 1
        
        if self.ablation_params['frames_captured'] >= self.ablation_params['frames_total']:
            self.end_time = time.time()
            print("Ablation sequence complete!")
            print(f"Material: {self.ablation_params['material_type']}, Lighting: {self.ablation_params['lighting_setup']}")
            print(f"Total images captured: {self.frame_count}")
            self.simulation_context.stop()

    def _handle_training_update(self):
        """Handle updates for training mode"""
        self.training_params['frames_captured'] += 1
        
        if self.training_params['frames_captured'] >= self.training_params['frames_total']:
            self.end_time = time.time()
            # elapsed_time = self.end_time - self.start_time
            print("Training sequence complete!")
            print(f"Total images captured: {self.frame_count}")
            # print(f"Total capture time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
            self.simulation_context.stop()

    def _handle_scanning_update(self):
        """Handle updates for scanning mode"""
        self.scanning_params['frames_captured_for_current_angle'] += 1
        
        if self.scanning_params['frames_captured_for_current_angle'] >= self.scanning_params['frames_per_angle']:
            self.scanning_params['current_angle'] += self.scanning_params['angle_increment']
            self.scanning_params['frames_captured_for_current_angle'] = 0
            
            if self.scanning_params['current_angle'] < 360:
                self._update_scan_object_pose()
                print(f"Moving to angle {self.scanning_params['current_angle']}")
            else:
                self.end_time = time.time()
                # elapsed_time = self.end_time - self.start_time
                print("Scanning sequence complete!")
                print(f"Total images captured: {self.frame_count}")
                # print(f"Total capture time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
                self.simulation_context.stop()

    def _handle_calibration_update(self):
        """Handle updates for calibration mode"""
        self.calibration_params['frames_captured_for_current_pose'] += 1
        
        if self.calibration_params['frames_captured_for_current_pose'] >= self.calibration_params['frames_per_pose']:
            self.calibration_params['current_pose_index'] += 1
            self.calibration_params['frames_captured_for_current_pose'] = 0
            
            if self.calibration_params['current_pose_index'] < len(self.calibration_params['positions']):
                self.update_calibration_board_pose()
                print(f"Moving to pose P{self.calibration_params['current_pose_index'] + 1}")
            else:
                self.end_time = time.time()
                # elapsed_time = self.end_time - self.start_time
                print("Calibration sequence complete!")
                print(f"Total images captured: {self.frame_count}")
                # print(f"Total capture time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
                self.simulation_context.stop()

    def capture_camera_frames(self):
        """Capture and save camera frames"""
        # Determine save directory based on mode
        if self.mode == "calibrate":
            pose_number = self.calibration_params['current_pose_index'] + 1
            save_directory = f"C:/Users/oadam/Downloads/isaac_calib_scans_dtwin_old_poses/P{pose_number}"
            image_index = self.calibration_params['frames_captured_for_current_pose']
        elif self.mode == "scan":
            angle = self.scanning_params['current_angle']
            save_directory = f"C:/Users/oadam/Downloads/fpp_synthetic_dataset/{self.scanning_params['scan_object']}/A{angle}"
            image_index = self.scanning_params['frames_captured_for_current_angle']
        elif self.mode == "ablation":
            material = self.ablation_params['material_type']
            lighting = self.ablation_params['lighting_setup']
            save_directory = f"C:/Users/oadam/Downloads/ablation_scans/{material}_{lighting}"
            image_index = self.ablation_params['frames_captured']
        else: # Training mode
            save_directory = "C:/Users/oadam/Downloads/isaac_calib_scans_rt0_albation_test_sphere_rectambientlight2"
            image_index = self.training_params['frames_captured']
        
        os.makedirs(save_directory, exist_ok=True)
        
        frame = self.camera.get_current_frame()
        if frame is not None and 'rgba' in frame:
            rgba_image = frame['rgba']
            if rgba_image.dtype != np.uint8:
                rgba_image = rgba_image.astype(np.uint8)
            
            rgb_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2RGB)
            
            if self.save_grayscale:
                image_to_save = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            else:
                image_to_save = rgb_image
            
            filename = os.path.join(save_directory, f"{'P' if self.mode == 'calibrate' else 'A'}_{image_index}.png")
            cv2.imwrite(filename, image_to_save)
            
            mode_str = "calibration" if self.mode == "calibrate" else "scanning" if self.mode == "scan" else "training"
            print(f"Captured {'grayscale' if self.save_grayscale else 'RGB'} image for {mode_str}")
            
            self.frame_count += 1
        else:
            print("Failed to capture frame or 'rgba' key is missing.")

    def set_projector_texture(self, texture_file_path):
        """Set the texture file for the projector light"""
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.light_prim_path)
        if prim:
            texture_attr = prim.GetAttribute("inputs:texture:file")
            if not texture_attr:
                texture_attr = prim.CreateAttribute("inputs:texture:file", Sdf.ValueTypeNames.Asset)
            texture_attr.Set(Sdf.AssetPath(texture_file_path))

    def setup_calibration_board(self, stage):
        """Setup the calibration board pattern, geometry, and material"""
        # Define the calibration board as a single-sided plane
        plane_path = "/World/CalibrationBoard"
        plane_prim = stage.GetPrimAtPath(plane_path)
        if not plane_prim:
            plane_prim = UsdGeom.Mesh.Define(stage, plane_path)
        
        # Define the mesh geometry
        points = [
            (-0.25, -0.25, 0),  # Bottom-left
            (0.25, -0.25, 0),   # Bottom-right
            (0.25, 0.25, 0),    # Top-right
            (-0.25, 0.25, 0)    # Top-left
        ]
        
        face_vertex_counts = [4]
        face_vertex_indices = [0, 1, 2, 3]
        normals = [(0, 0, 1)] * 4
        uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
        
        # Set mesh attributes
        plane_prim.CreatePointsAttr(points)
        plane_prim.CreateNormalsAttr(normals)
        plane_prim.CreateFaceVertexCountsAttr(face_vertex_counts)
        plane_prim.CreateFaceVertexIndicesAttr(face_vertex_indices)
        
        # Set UV coordinates
        primvars_api = UsdGeom.PrimvarsAPI(plane_prim)
        texCoords = primvars_api.CreatePrimvar("st", 
                                            Sdf.ValueTypeNames.TexCoord2fArray,
                                            UsdGeom.Tokens.vertex)
        texCoords.Set(uvs)
        
        # Set subdivision scheme and extent
        plane_prim.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        plane_prim.CreateExtentAttr().Set([(-.25, -.25, -.001), (.25, .25, .001)])
        
        # Set initial transform
        transform_api = UsdGeom.Xformable(plane_prim)
        transform_api.AddTranslateOp().Set(Gf.Vec3d(*self.calibration_params['positions'][0]))
        transform_api.AddRotateXYZOp().Set(Gf.Vec3f(*self.calibration_params['orientations'][0]))
        transform_api.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))
        
        # Create and setup material
        material_path = Sdf.Path("/World/CalibrationBoard/Material")
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path.AppendPath("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset("OmniPBR.mdl", "mdl")
        shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        
        # Connect shader to material
        material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        material.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        material.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        
        # Set material properties
        shader.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(0.15)
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("ao_to_diffuse", Sdf.ValueTypeNames.Float).Set(0.95)

        # Generate calibration board texture
        generator = CalibrationBoardGenerator(
            rows=5,
            cols=9,
            circle_diameter=10.0,
            circle_center_distance=20.0,
            border_length=50.0,
            scale_factor=2.0,
            plane_width=0.5,
            plane_height=0.5,
            # inverted=True
        )
        texture_path = generator.generate_texture(size=2048)

        # Apply texture to material
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_path))

        # Bind material to the plane
        UsdShade.MaterialBindingAPI(plane_prim).Bind(material)

    def update_calibration_board_pose(self):
        """Update the calibration board pose based on the current index"""
        stage = omni.usd.get_context().get_stage()
        plane_prim = stage.GetPrimAtPath("/World/CalibrationBoard")
        if plane_prim:
            # Ensure normal vector is correct after transformation
            geom = UsdGeom.Mesh(plane_prim)
            normals = geom.GetNormalsAttr()
            if not normals:
                normals = geom.CreateNormalsAttr()
                normals.Set([(0, 0, 1)] * 4)
            
            # Get transformable API
            transform_api = UsdGeom.Xformable(plane_prim)

            # Update position and orientation
            current_pose = self.calibration_params['current_pose_index']
            
            # Get existing operations or create new ones if they don't exist
            xform_ops = transform_api.GetOrderedXformOps()
            
            # Find or create translate operation
            translate_op = next((op for op in xform_ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
            if not translate_op:
                translate_op = transform_api.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(*self.calibration_params['positions'][current_pose]))
            
            # Find or create rotate operation
            rotate_op = next((op for op in xform_ops if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ), None)
            if not rotate_op:
                rotate_op = transform_api.AddRotateXYZOp()
            rotate_op.Set(Gf.Vec3f(*self.calibration_params['orientations'][current_pose]))

    def setup_background_plane(self, stage):
        """Setup the background plane geometry and material"""
        # Define the training plane as a single-sided plane
        plane_path = "/World/BackgroundPlane"
        plane_prim = stage.GetPrimAtPath(plane_path)
        if not plane_prim:
            plane_prim = UsdGeom.Mesh.Define(stage, plane_path)
        
        # Define the mesh geometry
        size = 1.0  # Adjust size as needed
        points = [
            (-size, -size, 0),
            (size, -size, 0),
            (size, size, 0),
            (-size, size, 0)
        ]
        
        face_vertex_counts = [4]
        face_vertex_indices = [0, 1, 2, 3]
        normals = [(0, 0, 1)] * 4
        uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
        
        # Set mesh attributes
        plane_prim.CreatePointsAttr(points)
        plane_prim.CreateNormalsAttr(normals)
        plane_prim.CreateFaceVertexCountsAttr(face_vertex_counts)
        plane_prim.CreateFaceVertexIndicesAttr(face_vertex_indices)
        
        # Set UV coordinates
        primvars_api = UsdGeom.PrimvarsAPI(plane_prim)
        texCoords = primvars_api.CreatePrimvar("st", 
                                            Sdf.ValueTypeNames.TexCoord2fArray,
                                            UsdGeom.Tokens.vertex)
        texCoords.Set(uvs)
        
        # Set subdivision scheme and extent
        plane_prim.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        plane_prim.CreateExtentAttr().Set([(-size, -size, -.001), (size, size, .001)])
        
        # Set initial transform to match calibration board
        transform_api = UsdGeom.Xformable(plane_prim)
        x_pos = self.calibration_params['positions'][0][0] + 1
        y_pos = self.calibration_params['positions'][0][1]
        z_pos = self.calibration_params['positions'][0][2]
        transform_api.AddTranslateOp().Set(Gf.Vec3d(x_pos, y_pos, z_pos))
        transform_api.AddRotateXYZOp().Set(Gf.Vec3f(*self.calibration_params['orientations'][0]))
        transform_api.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))
        
        # Create and setup material
        material_path = Sdf.Path("/World/BackgroundPlane/Material")
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path.AppendPath("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset("OmniPBR.mdl", "mdl")
        shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        
        # Connect shader to material
        material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        material.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        material.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        
        # Set material properties
        shader.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(0.15)
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("ao_to_diffuse", Sdf.ValueTypeNames.Float).Set(0.95)

        # Bind material to the plane
        UsdShade.MaterialBindingAPI(plane_prim).Bind(material)

    async def setup_post_load(self):
        """Setup post-load simulation state"""
        world = self.get_world()
        world.add_physics_callback("update_texture", callback_fn=self.update_texture_callback)
        self.start_time = time.time()
        print("Starting capture sequence.")
        return

    async def setup_pre_reset(self):
        """Pre-reset cleanup"""
        return

    async def setup_post_reset(self):
        """Reset simulation state"""
        world = self.get_world()
        world.reset()
        
        # Reset common parameters
        self.frame_count = 0
        self.current_texture_index = 0
        self.last_texture_update_time = 0.0
        self.set_projector_texture(self.texture_files[0])

        # Reset time tracking
        self.start_time = None
        self.end_time = None

        # Reset mode-specific parameters
        if self.mode == "calibrate":
            self.calibration_params['current_pose_index'] = 0
            self.calibration_params['frames_captured_for_current_pose'] = 0
            self.update_calibration_board_pose()
        elif self.mode == "scan":
            self.scanning_params['current_angle'] = 0
            self.scanning_params['frames_captured_for_current_angle'] = 0
            self._update_scan_object_pose()
        elif self.mode == "ablation":
            self.ablation_params['frames_captured'] = 0
        else: # Training mode
            self.training_params['frames_captured'] = 0
        return

    def world_cleanup(self):
        """Cleanup simulation state"""
        return