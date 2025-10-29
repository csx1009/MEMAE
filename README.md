# MEMAE: Microstructure Informed Mamba Vision Masked Autoencoder for Personalized Brain Injury Detection from Diffusion MRI

This repository contains the official implementation for **MEMAE**, a model designed for personalized brain injury detection from diffusion MRI data.

## 1. Environment Setup

Follow these steps to set up the necessary conda environment and install dependencies.

1.  **Create and activate the conda environment:**
    ```bash
    conda create -n memae python=3.10.13
    conda activate memae
    ```

2.  **Install PyTorch and CUDA:**
    ```bash
    # Install CUDA Toolkit
    conda install cudatoolkit==11.8 -c nvidia
    
    # Install PyTorch (v2.1.1 for cu118)
    pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    ```

3.  **Install Mamba and other dependencies:**
    ```bash
    # Install CUDA compiler (needed for Mamba)
    conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
    
    # Install packaging
    conda install packaging
    
    # Install Mamba (SSM)
    pip install mamba-ssm
    ```
    > For more details on the Mamba architecture, visit [state-spaces/mamba](https://github.com/state-spaces/mamba).

## 2. Data Preprocessing

* **Module:** `data_set/`
* **Description:** This step involves standardizing the resolution and dimensions of all input images. Data is also normalized to prepare it for model training.

## 3. Model Training

To begin training the MEMAE model, run the main training script.

* **Command:**
    ```bash
    python train.py -pdir /MEMAE /parameter/par.yml -gpu 0
    ```
    *(Note: The arguments `-pdir /MEMAE /parameter/par.yml` are based on your input. Please adjust paths and arguments as needed.)*

## 4. Testing (Inference)

To run inference on the test set using a trained model.

* **Command:**
    ```bash
    python test.py
    ```

## 5. Prior Knowledge Base

This module is used for the creation and utilization of the prior knowledge base.

* **Script:** `jkzxd.py`
