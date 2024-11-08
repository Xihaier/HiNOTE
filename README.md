<div align="center">

# Hierarchical Neural Operator Transformer with Learnable Frequency-aware Loss Prior for Arbitrary-scale Super-resolution (ICML 2024)

<a href="https://pytorch.org/get-started/locally/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch -ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra-89b8cd?style=for-the-badge&labelColor=gray"></a>

<h3> ✨Official implementation of our <a href="https://proceedings.mlr.press/v235/luo24g.html">HiNOTE</a> model✨ </h3>
 
</div>

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Citation](#citation)
- [License](#license)

## Project Overview

In this work, we present an arbitrary-scale superresolution (SR) method to enhance the resolution of scientific data, which often involves complex challenges such as continuity, multi-scale physics, and the intricacies of high-frequency signals. Grounded in operator learning, the proposed method is resolution-invariant. The core of our model is a hierarchical neural operator that leverages a Galerkin-type self-attention mechanism, enabling efficient learning of mappings between function spaces. Sinc filters are used to facilitate the information transfer across different levels in the hierarchy, thereby ensuring representation equivalence in the proposed neural operator. Additionally, we introduce a learnable prior structure that is derived from the spectral resizing of the input data. This loss prior is model-agnostic and is designed to dynamically adjust the weighting of pixel contributions, thereby balancing gradients effectively across the model. We conduct extensive experiments on diverse datasets from different domains and demonstrate consistent improvements compared to strong baselines, which consist of various state-of-the-art SR methods.

## Directory Structure

The repository is organized as follows:

```plaintext
├── configs                   # Configuration files for different models and datasets
│   ├── callback              # Callback configuration
│   │   └── default.yaml
│   ├── datamodule            # Dataset-specific configurations
│   │   ├── baryon_density.yaml
│   │   ├── kinetic_energy.yaml
│   │   ├── MRI.yaml
│   │   ├── SEVIR_2018.yaml
│   │   ├── temperature.yaml
│   │   ├── vorticity_Re_16000.yaml
│   │   ├── vorticity_Re_32000.yaml
│   │   └── water_vapor.yaml
│   ├── default.yaml          # Default configuration
│   ├── hydra                 # Hydra configurations for experiment management
│   │   └── default.yaml
│   ├── logger                # Configuration for logging (e.g., Weights & Biases)
│   │   └── wandb.yaml
│   ├── model                 # Model-specific configurations
│   │   ├── DFNO.yaml
│   │   ├── EDSR.yaml
│   │   ├── ESPCN.yaml
│   │   ├── HiNOTE.yaml
│   │   ├── LIIF.yaml
│   │   ├── LTE.yaml
│   │   ├── MetaSR.yaml
│   │   ├── SRCNN.yaml
│   │   ├── SRNO.yaml
│   │   ├── SwinIR.yaml
│   │   └── WDSR.yaml
│   ├── optim                 # Optimizer configurations
│   │   └── default.yaml
│   ├── scheduler             # Scheduler configurations
│   │   └── default.yaml
│   └── trainer               # Trainer configurations
│       └── default.yaml
├── data                      # Data generation scripts
│   └── data_generation.py    # Script for generating and preprocessing data
├── data_interface.py         # Main interface for handling data input/output
├── dataloaders               # Custom dataloaders for various data sources
│   ├── ASdataloader.py       # Dataloader for AS data
│   ├── data_utils.py         # Utility functions for data loading
│   └── SSdataloder.py        # Dataloader for SS data
├── model_interface.py        # Interface for managing model loading and interaction
├── models                    # Implementations of various models
│   ├── DFNO.py               
│   ├── EDSR.py               
│   ├── ESPCN.py              
│   ├── HiNOTE.py             
│   ├── LIIF.py               
│   ├── LTE.py                
│   ├── MetaSR.py             
│   ├── SRCNN.py              
│   ├── SRNO.py               
│   ├── SwinIR.py             
│   └── WDSR.py               
├── run.sh                    # Shell script for running the project
├── test.py                   # Script for testing models
├── train.py                  # Script for training models
└── utils.py                  # Utility functions
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Xihaier/HiNOTE.git
   cd HiNOTE
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate HiNOTE
   ```

3. **Verify the installation:**

    Ensure that all dependencies are correctly installed by running:

    ```bash
    python -c "import torch; print(torch.__version__)"
    ```

## Usage

### Data

This project utilizes several datasets, each of which is available online. Here is an overview of the datasets and links to access them:

1. **Turbulence Data**  
   - **Description**: This dataset contains two-dimensional Kraichnan turbulence in a doubly periodic square domain spanning.
   - **Access**: Available at [NERSC's Data Portal](https://portal.nersc.gov/project/dasrepo/superbench/).
   
2. **Weather Data**  
   - **Description**: This dataset contains ERA5 reanalysis data, high-resolution simulated surface temperature at 2 meters, kinetic energy at 10 meters above the surface, and total column water vapor.
   - **Access**: Available at [NERSC's Data Portal](https://portal.nersc.gov/project/dasrepo/superbench/).

3. **SEVIR Data**  
   - **Description**: This dataset encompasses various weather phenomena, including thunderstorms, convective systems, and related events
   - **Access**: Available at [AWS Open Data Registry](https://registry.opendata.aws/sevir/).
   
4. **MRI Data**  
   - **Description**: This dataset contains Magnetic Resonance Imaging data.
   - **Access**: Available on [GitHub - UTCSILAB's repository](https://github.com/utcsilab/csgm-mri-langevin).

### Training

To train a model, modify the configuration files in `configs/` as needed. Then, run:

```bash
python train.py
```

### Testing

To test a trained model, use the following command:

```bash
python test.py
```

### Running the Script

For convenience, you can also use the shell script `run.sh` to automate multiple training/testing runs.

```bash
bash run.sh
```

## Model
<p align="center">
<img src=".\img\model.png" height = "250" alt="" align=center />
</p>

## Quantitative Results
<p align="center">
<img src=".\img\quan.png" height = "500" alt="" align=center />
</p>

## Qualitative Results
<p align="center">
<img src=".\img\qual.png" height = "250" alt="" align=center />
</p>

## Citation

If you find this repo useful, please cite our paper. 

```
@article{luo2024hierarchical,
  title={Hierarchical Neural Operator Transformer with Learnable Frequency-aware Loss Prior for Arbitrary-scale Super-resolution},
  author={Luo, Xihaier and Qian, Xiaoning and Yoon, Byung-Jun},
  journal={arXiv preprint arXiv:2405.12202},
  year={2024}
}
```

## Contact

If you have any questions or want to use the code, please contact [xluo@bnl.gov](mailto:xluo@bnl.gov).
