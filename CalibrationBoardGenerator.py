"""Calibration board texture generation for VIRTUS-FPP.

Builds an asymmetric circle-grid calibration pattern (the kind OpenCV's
findCirclesGrid expects) and renders it to a square PNG texture that the main
sample binds to the calibration board plane. Physical measurements (circle
diameter, spacing, border) are specified in millimeters and scaled to fit a
fixed-size plane mesh while preserving their real-world ratios.
"""

import numpy as np
import cv2
import os
import tempfile
from typing import Tuple

class CalibrationBoardGenerator:
    """Generates a circle-grid calibration board texture from physical
    (millimeter) measurements, scaled to fit a fixed plane mesh."""

    def __init__(self,
                 rows: int = 10,                        # Number of circle rows
                 cols: int = 18,                        # Number of circle columns
                 circle_diameter: float = 20.0,         # mm
                 circle_center_distance: float = 40.0,  # mm
                 border_length: float = 50.0,           # mm
                 scale_factor: float = 1.0,
                 plane_width: float = 0.5,              # meters - width of the fixed plane mesh
                 plane_height: float = 0.5) -> None:    # meters - height of the fixed plane mesh
        """
        Initialize the calibration board generator with given parameters.
        Adapts the pattern to fit within a fixed-size plane while maintaining physical measurements.
        
        Args:
            rows: Number of circle rows
            cols: Number of circle columns
            circle_diameter: Diameter of each circle in mm
            circle_center_distance: Distance between circle centers in mm
            border_length: Length of border in mm
            scale_factor: Scale factor for texture resolution
            plane_width: Width of the plane mesh in meters
            plane_height: Height of the plane mesh in meters
        """
        # Input validation
        self._validate_inputs(rows, cols, circle_diameter, circle_center_distance, 
                            border_length, scale_factor, plane_width, plane_height)

        # Initialize base paramters 
        self.rows = rows
        self.cols = cols
        self.circle_diameter = circle_diameter
        self.circle_center_distance = circle_center_distance
        self.border_length = border_length
        self.scale_factor = scale_factor
        self.plane_width = plane_width
        self.plane_height = plane_height

        # Calculate derived parameters
        self._calculate_derived_parameters()

    def _validate_inputs(self, rows: float, cols: float, circle_diameter: float,
                        circle_center_distance: float, border_length: float,
                        scale_factor: float, plane_width: float, plane_height: float) -> None:
        """Validate the constructor parameters.

        Raises TypeError if any value is non-numeric and ValueError if any is
        non-positive, if circles would overlap (diameter >= center distance),
        or if the border is too small to contain a half circle.
        """
        if not all(isinstance(x, (int, float)) for x in [rows, cols, circle_diameter, 
                                                        circle_center_distance, border_length, 
                                                        scale_factor, plane_width, plane_height]):
            raise TypeError("All parameters must be numeric")
            
        if not all(x > 0 for x in [rows, cols, circle_diameter, circle_center_distance, 
                                  border_length, scale_factor, plane_width, plane_height]):
            raise ValueError("All parameters must be positive")
            
        if circle_diameter >= circle_center_distance:
            raise ValueError("Circle diameter must be less than center distance")
            
        if border_length < circle_diameter/2:
            raise ValueError("Border length must be at least half the circle diameter")
        
    def _calculate_derived_parameters(self) -> None:
        """Compute the physical pattern size, the scale that fits it to the
        plane, and the resulting pixel dimensions (circle radius, spacing,
        border) used when rasterizing the texture."""
        self.mm_to_meters = 0.001
        
        # Physical dimensions in meters
        self.pattern_width = (self.border_length * 2 + 
                            (self.cols - 1) * self.circle_center_distance / 2 + 
                            self.circle_diameter) * self.mm_to_meters
        self.pattern_height = (self.border_length * 2 + 
                             (self.rows - 1) * self.circle_center_distance + 
                             self.circle_diameter) * self.mm_to_meters
        
        # Calculate scaling factors
        self.pattern_scale = min(self.plane_width / self.pattern_width,
                               self.plane_height / self.pattern_height)
        
        # Pixel calculations
        base_pixels_per_meter = 4096  # Base resolution for 1 meter
        self.pixels_per_mm = (base_pixels_per_meter * self.pattern_scale * 
                            self.mm_to_meters * self.scale_factor)
        
        self.circle_radius_pixels = int((self.circle_diameter / 2) * self.pixels_per_mm)
        self.circle_center_distance_pixels = int(self.circle_center_distance * self.pixels_per_mm)
        self.border_pixels = int(self.border_length * self.pixels_per_mm)
        
    def _draw_circle(self, image: np.ndarray, x: int, y: int, r: int) -> np.ndarray:
        """Draw a filled circle on the image using vectorized operations"""
        y_indices, x_indices = np.ogrid[:image.shape[0], :image.shape[1]]
        dist_from_center = (x_indices - x)**2 + (y_indices - y)**2
        circle_mask = dist_from_center <= r**2
        image[circle_mask] = 255
        return image
    
    def _calculate_dimensions(self) -> Tuple[int, int]:
        """Calculate the dimensions of the circle pattern area"""
        height = int(self.border_pixels * 2 + 
                    (self.rows - 1) * self.circle_center_distance_pixels + 
                    2 * self.circle_radius_pixels)
        width = int(self.border_pixels * 2 + 
                   (self.cols - 1) * self.circle_center_distance_pixels/2 + 
                   2 * self.circle_radius_pixels)
        return height, width

    def generate_texture(self, size: int = 2048) -> str:
        """
        Generate a square calibration board texture and save it to a temporary file.
        
        Args:
            size: Size of the square texture in pixels (should be power of 2)
            
        Returns:
            Path to the generated texture file
        """
        # Calculate pattern dimensions
        pattern_height, pattern_width = self._calculate_dimensions()
        
        # Create the pattern image
        pattern = np.zeros((pattern_height, pattern_width), dtype=np.uint8)
        crp2 = self.circle_center_distance_pixels / 2
        
        # Calculate starting position to center the pattern
        start_x = self.border_pixels
        start_y = self.border_pixels
        
        # Draw circles
        for r in range(self.rows):
            ssy = int(start_y + r * self.circle_center_distance_pixels)
            for c in range(self.cols):
                ssx = int(start_x + c * self.circle_center_distance_pixels/2)
                if c % 2:
                    pattern = self._draw_circle(pattern, ssx, ssy, 
                                             self.circle_radius_pixels)
                else:
                    pattern = self._draw_circle(pattern, ssx, 
                                             int(ssy + crp2), 
                                             self.circle_radius_pixels)
        
        # Create square canvas
        canvas = np.zeros((size, size), dtype=np.uint8)
        
        # Scale pattern to fit within the square canvas while maintaining aspect ratio
        scale = min(size / pattern_height, size / pattern_width) * 0.95  # 0.95 to ensure border
        new_height = int(pattern_height * scale)
        new_width = int(pattern_width * scale)
        
        # Resize pattern using NEAREST neighbor interpolation to preserve circle edges
        pattern_resized = cv2.resize(pattern, (new_width, new_height), 
                                   interpolation=cv2.INTER_NEAREST)
        
        # Calculate centering offsets
        y_offset = (size - new_height) // 2
        x_offset = (size - new_width) // 2
        
        # Place pattern on canvas with centering offsets
        canvas[y_offset:y_offset + new_height, 
               x_offset:x_offset + new_width] = pattern_resized
        
        # Create temporary file path with PNG format for lossless quality
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "calibration_board_texture.png")
        print(f"Calibration board texture saved to: {temp_path}")
        
        # Save the image in PNG format for better quality
        cv2.imwrite(temp_path, canvas, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        # Output expected measurements to console
        self._get_expected_measurements()
        
        return temp_path
    
    def _get_expected_measurements(self) -> None:
        """Print a summary of the pattern's expected physical measurements
        (size on the plane, circle diameter and spacing) for verification."""
        print("\nApproximate expected measurements:")
        print(f"Pattern physical size: {self.pattern_width:.3f}m x {self.pattern_height:.3f}m")
        print(f"Scaled to fit plane: {self.pattern_width * self.pattern_scale:.3f}m x {self.pattern_height * self.pattern_scale:.3f}m")
        print(f"Circle diameter in plane: {self.circle_diameter * self.mm_to_meters * self.pattern_scale * 1000:.1f}mm")
        print(f"Circle center distance (adjacent) in plane: {self.circle_center_distance * self.mm_to_meters * self.pattern_scale * 1000:.1f}mm")