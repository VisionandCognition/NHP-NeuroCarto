# NHP-NeuroCarto: A Neuropixels Channelmap Editor for NHP probes

## NeuroCarto
**NeuroCarto** is a neural probe channel map editor for the Neuropixels probe family. It allows user to create a blueprint for arranging electrodes in a desired density and generate a custom channel map. It was developed by Ta-Shun Su and Fabian Kloosterman at NERF/KUL.

**NHP-NeuroCarto** is an adaptation of the NeuroCarto package created by Chris Klink ([c.klink@nin.knaw.nl](mailto:c.klink@nin.knaw.nl)) to increase compatibility with long NHP Neuropixels probes. The main difference in approach is that for NHPs we cannot rely on standard atlases, but we need an individualized planning. In our lab at the [Netherlands Institute for Neuroscience](nin.nl), we use [3D Slicer](https://www.slicer.org/) for much of our offline trajectory planning. NHP-NeuroCarto is configured to take a screenshot of an in-plane ('probe-view') and align it to an NHP-Neuropixels probe model for typical NeuroCarto channel selection. An extensive how-to is available in [How to use NHP-NeuroCarto](HowToNHPNeuroCarto.md).

The original documentation of the NeuroCarto version this adaptation was based on is available [here](README_NeuroCartoOrg.md) and the latest version can be found on the original repository [https://github.com/AntonioST/NeuroCarto](https://github.com/AntonioST/NeuroCarto). For more information, see also the corresponding paper: 

Su, TS., Kloosterman, F. NeuroCarto: A Toolkit for Building Custom Read-out Channel Maps for High Electrode-count Neural Probes. *Neuroinform* **23**, 1–16 (2025). https://doi.org/10.1007/s12021-024-09705-2

If you want to run the NHP adaptation you should **not** install through PyPi (pip), but build the adapted version using this repository instead.

## Features
- Read/Visualize/Modify/Write Neuropixels channelmap files (`*.imro`).
- Read SpikeGLX meta file (`*.meta`).
- Read/Visualize/Modify/Write Blueprint (a blueprint for generating a channelmap by a programming way).
- Show screenshot of NHP brain as a background image.
- Customize electrode selection.
- Show channel efficiency and electrode density.


## Install and Run

#### Prepare an environment.

Requires `Python 3.10` or later. We suggest creating a `conda` environment ([https://www.anaconda.com/](https://www.anaconda.com/)).

#### Build from source
##### Create a Python environment. 
Here use conda as example. NHP-NeuroCarto requires `Python 3.10` or later.

```shell
conda create -n neurocarto python~=3.10.0
conda activate neurocarto
```
##### Clone repository.
```shell
git clone https://github.com/VisionandCognition/NHP-NeuroCarto.git
cd NHP-NeuroCarto
```

##### Update pip
```shell
python -m pip install --upgrade pip
python -m pip install --upgrade build
```

##### Build
```shell
python -m build
```

#### Install
```shell
pip install dist/neurocarto-0.1.0-py3-none-any.whl
# change version 0.1.0 to latest when needed.
```

### Optional dependency

* `bg-atlasapi` Atlas Brain background image supporting. We're not really using this for NHP-NeuroCarto.
* `Pillow`, `tifffile` other background image format supporting.
* `probeinterface` probe/channelmap format import/export
* `pandas`, `polars` channelmap data export.

Full optional dependencies are list in [requirements-opt.txt](requirements-opt.txt).