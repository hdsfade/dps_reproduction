# Reproduction of Diffusion Posterior Sampling (DPS)

This repository contains a unofficial PyTorch implementation of **Diffusion Posterior Sampling (DPS)** for solving general noisy inverse problems, as introduced in the paper *Diffusion Posterior Sampling for General Noisy Inverse Problems*.

## Quick Start

### 1. Installation

To get started, clone this repository and install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/your_username/dps_repository.git
cd dps_repository

# Create a conda environment (optional)
conda create -n dps_reproduction python=3.8
conda activate dps_reproduction

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Pre-trained Models

Pre-trained diffusion models are required to run DPS. Download the models from the provided links and place them in the `models/` directory.

```bash
gdown -O ./models https://drive.google.com/uc?id=1BGwhRWUoguF-D8wlZ65tf227gp3cDUDh
gdown -O ./models https://drive.google.com/uc?id=1HAy7P19PckQLczVNXmVF-e_CRxq098uW
```

### 3. Download Datasets

Prepare the dataset required for your inverse problem. Example commands for downloading datasets are provided below:

```bash
# samples will be saved in data/samples/ 
python download_ffhq.py 
```

### 4. Running the DPS Method

Use the provided scripts to run DPS on a chosen dataset. Modify configuration parameters as needed in the `config/` directory.

```bash
python sample_condition.py --model_config=configs/ffhq_model_config.yaml --diffusion_config=configs/diffusion_config.yaml --task_config=configs/inpainting_config.yaml --gpu 1 --batch_size 1
```

### 5. Evaluating Results

After running DPS, evaluate the results using the provided evaluation scripts.

```bash
# Compute performance metrics (e.g., FID, LPSIS)
bash cal_metrics.sh
```

### 6. Visual Comparison

Visualize the results to compare reconstructed images with ground truth by visualization.ipynb file.

## Directory Structure

```
dps_repository/
├── config/                 # Configuration files
├── data/                   # Datasets
├── models/                 # Pre-trained models
├── results/                # Output results
├── requirements.txt        # Required Python packages
├── sample_condition.py     # Main script for DPS
├── cal_metrics.sh          # Evaluation script
├── evaluation.py           # Evaluation codes
├── visualization.ipynb     # Visualization codes
└── README.md               # Repository description (this file)
```

---

## Original Paper
The paper being reproduced is as follows:
```
@article{
    chung2023diffusion,
    title={Diffusion Posterior Sampling for General Noisy Inverse Problems},
    author={Hyungjin Chung and Jeongsol Kim and Michael Thompson Mccann and Marc Louis Klasky and Jong Chul Ye},
    booktitle={The Eleventh International Conference on Learning Representations },
    year={2023},
    url={https://openreview.net/forum?id=OnD9zGAGT0k}
}
```