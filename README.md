# VIRTUS-FPP

**Virtual Sensor Modeling for Fringe Projection Profilometry in NVIDIA Isaac Sim**

VIRTUS-FPP is an end-to-end virtual sensor model for fringe projection
profilometry (FPP) built as a user extension for [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)
4.1. It places a virtual camera and an inverse-camera-modeled projector (a
textured `RectLight` that cycles through a stack of fringe patterns) in an RTX
ray-traced scene, captures one image per projected pattern, and writes the
frames to disk for downstream phase unwrapping and 3D reconstruction. The
framework supports both fully simulated parameters and digital-twin replication
of a real, pre-calibrated FPP system.

This is the reference implementation for the paper:

> A. Haroon, A. Lakshman, B. Balasubramaniam, and B. Li, "VIRTUS-FPP: Virtual
> Sensor Modeling for Fringe Projection Profilometry in NVIDIA Isaac Sim,"
> *IEEE Sensors Journal*, 2026. DOI: 10.1109/JSEN.2026.3698278.

## Repository layout

| File | Purpose |
| --- | --- |
| `hello_world.py` | Main sample. Builds the scene, drives the capture loop, and saves frames for the selected mode. |
| `hello_world_extension.py` | Isaac Sim extension entry point that registers the sample in the UI. |
| `CalibrationBoardGenerator.py` | Generates the circle-grid calibration board texture. |
| `realworld2digitaltwin.py` | Converts real camera/projector calibration (intrinsics and extrinsics) into Isaac Sim parameters, including the inverse-camera projector model. |
| `objects.py` | Scan object asset definitions. |

## Requirements

- NVIDIA Isaac Sim 4.1
- An RTX-capable NVIDIA GPU
- Python dependencies used by the sample: `numpy`, `opencv-python` (`cv2`)

## Installation

Place this repository in the Isaac Sim user examples directory so it is picked
up as a user extension:

```
<isaac-sim>/exts/omni.isaac.examples/omni/isaac/examples/user_examples/
```

Launch Isaac Sim, open the sample from the examples menu, and press **LOAD**.
The capture sequence starts automatically once the world finishes loading.

## Usage

All tunable parameters live in `HelloWorld.__init__` in `hello_world.py`.
Set the operating mode there:

```python
self.mode = "calibrate"  # "calibrate", "scan", "training", "ablation", or "factory_arm"
```

Before running, point the sample at your local paths:

- `texture_directory` in `_load_texture_files` -> folder of N-step fringe pattern images (`.bmp` / `.png`)
- the `save_directory` values in `capture_camera_frames` -> where captured frames are written
- `calib_path` in `__init__` -> your calibration files, only needed when `use_real_world_params = True`

### Modes

| Mode | What it does |
| --- | --- |
| `calibrate` | Sweeps a circle-grid calibration board through a series of poses to recover camera and projector intrinsics and extrinsics. Poses are defined by `_get_calibration_positions` / `_get_calibration_orientations`. |
| `scan` | Rotates a YCB / SimReady object on a turntable and captures fringe images at each angle for synthetic dataset generation. Object and turntable settings are in `scanning_params`; available objects are in `SCAN_OBJECTS`. |
| `training` | Projects fringes onto a flat plane, e.g. to generate phase-unwrapping training data. |
| `ablation` | Varies surface material and ambient lighting to study their effect on reconstruction quality. Controlled by `ablation_params`. |
| `factory_arm` | Mounts the FPP rig on a Franka end-effector in a warehouse environment for a robot-mounted scanning demo. Controlled by `factory_arm_params`. |

### Real-world parameters (digital twin)

Set `self.use_real_world_params = True` to drive the virtual camera and
projector from a real calibration instead of the default simulated values. The
calibration matrices and extrinsics are read from the paths in
`self.real_world_params`, and `realworld2digitaltwin.py` maps them into Isaac
Sim, including the inverse-camera projector model.

## Output

Captured frames are written as PNG images (grayscale by default, controlled by
`self.save_grayscale`) into mode-specific subdirectories under the configured
save path. Each frame corresponds to one projected fringe pattern at one
calibration pose / scan angle.

## Citation

If you use VIRTUS-FPP in your research, please cite the paper:

```bibtex
@ARTICLE{11554253,
  author={Haroon, Adam and Lakshman, Anush and Balasubramaniam, Badrinath and Li, Beiwen},
  journal={IEEE Sensors Journal}, 
  title={VIRTUS-FPP: Virtual Sensor Modeling for Fringe Projection Profilometry in NVIDIA Isaac Sim}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Modeling;Optical projectors;Calibration;Cameras;Lighting;Image sensors;Simulation;Ray tracing;Measurement;Digital twins;Digital twin;fringe projection profilometry;NVIDIA Isaac Sim;optical metrology;ray-tracing simulation;structured light;synthetic data generation;virtual sensor modeling},
  doi={10.1109/JSEN.2026.3698278}}
```

## License

See [LICENSE](LICENSE).
