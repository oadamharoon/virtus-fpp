# Installation and Setup

VIRTUS-FPP is built for **NVIDIA Isaac Sim 4.1.0** and runs as a user extension
inside it. The steps below cover downloading the matching Isaac Sim release,
installing the project, and launching it.

> **Note on Isaac Sim version:** Isaac Sim 4.1.0 is an older release and is no
> longer officially supported by NVIDIA. It is the version VIRTUS-FPP was
> developed and tested against, so these instructions target it directly. The
> code relies on standard Isaac Sim APIs (camera, lights, USD, physics
> callbacks) and should transfer to newer Isaac Sim versions with little to no
> change.

## 1. Download Isaac Sim 4.1.0

This project targets Isaac Sim 4.1.0 specifically. Get it from the Isaac Sim
[download archive](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html),
or use the direct links below:

- **Linux (x86_64):** [isaac-sim-standalone@4.1.0-rc.7 (Linux)](https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone%404.1.0-rc.7%2B4.1.14801.71533b68.gl.linux-x86_64.release.zip)
- **Windows (x86_64):** [isaac-sim-standalone@4.1.0-rc.7 (Windows)](https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone%404.1.0-rc.7%2B4.1.14801.71533b68.gl.windows-x86_64.release.zip)

Make sure your machine meets the Isaac Sim
[system requirements](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html)
(an RTX-capable NVIDIA GPU is required).

## 2. Extract Isaac Sim

Unzip the downloaded archive. This gives you an `isaac-sim-4.1.0` directory,
for example:

- Windows: `C:\Users\<you>\Downloads\isaac-sim-4.1.0`
- Linux: `~/isaac-sim-4.1.0`

## 3. Install the project

Copy the contents of this repository into the Isaac Sim user examples directory:

```
isaac-sim-4.1.0/exts/omni.isaac.examples/omni/isaac/examples/user_examples
```

You can either clone the repository directly into that folder or copy the files
in manually. For example, with git:

```bash
# Windows (PowerShell)
cd "C:\path\to\isaac-sim-4.1.0\exts\omni.isaac.examples\omni\isaac\examples"
git clone https://github.com/oadamharoon/virtus-fpp.git user_examples

# Linux
cd ~/isaac-sim-4.1.0/exts/omni.isaac.examples/omni/isaac/examples
git clone https://github.com/oadamharoon/virtus-fpp.git user_examples
```

If a `user_examples` directory already exists, copy the project files into it so
that `hello_world.py`, `hello_world_extension.py`, `CalibrationBoardGenerator.py`,
and `realworld2digitaltwin.py` sit directly under `user_examples`.

## 4. Configure paths

Before running, edit `HelloWorld.__init__` in `hello_world.py` and point the
sample at your local paths (all set in `__init__`; see the [README](README.md)
for details):

- `self.texture_directory` -> the fringe pattern texture folder
- each mode's `save_directory` -> the output root where that mode's frames are saved
- `calib_path` -> the calibration files, only needed when `use_real_world_params = True`

Also set the operating mode:

```python
self.mode = "calibrate"  # "calibrate", "scan", "training", "ablation", or "factory_arm"
```

## 5. Launch

Start the Isaac Sim 4.1.0 application. Once it has loaded, open the
**Isaac Examples** dropdown in the top menu bar. You should see an item named
**Robot test development**, which is the VIRTUS-FPP sample. Select it to open
its popup, then press **LOAD**.

The capture sequence starts automatically once the world finishes loading, and
frames are written to the configured output directory.

> **Editing the code:** Any change you make to the code must be saved and then
> applied by pressing **LOAD** again. The sample is only re-read when the world
> is (re)loaded, so unsaved or un-reloaded changes will not take effect.

## Known issues

### `'SimulationContext' object has no attribute 'scene'` on first load

The first time you load the custom user extension, you may see an error like
this in the terminal:

```
[152.964s] Isaac Sim App is loaded.
2026-06-09 23:29:26 [163,987ms] [Error] [asyncio] Task exception was never retrieved
future: <Task finished name='Task-186' coro=<BaseSampleExtension._on_load_world.<locals>._on_load_world_async() done, defined at .../base_sample/base_sample_extension.py:152> exception=AttributeError("'SimulationContext' object has no attribute 'scene'")>
Traceback (most recent call last):
  File ".../base_sample/base_sample_extension.py", line 153, in _on_load_world_async
    await self._sample.load_world_async()
  File ".../base_sample/base_sample.py", line 43, in load_world_async
    self.setup_scene()
  File ".../user_examples/hello_world.py", line 677, in setup_scene
    world.scene.add_default_ground_plane()
AttributeError: 'SimulationContext' object has no attribute 'scene'
```

This happens because the world/scene is not fully initialized on the very first
load. **Just press LOAD again** and the scene will build normally.
