# MAVSS-SR: Multi-Band Attention and Visual State Space for Image Super-Resolution

## Project Overview

This repository provides the released code for **MAVSS-SR: An Image Super-Resolution Network Using Multi-Band Attention and Visual State Space**.

MAVSS-SR is designed for single-image super-resolution. The network separates image features into multiple frequency bands and applies band-specific modeling strategies to improve detail reconstruction and structural preservation. High-frequency features are modeled with a visual state space module, mid-frequency texture information is processed with deformable convolution, and low-frequency structural information is enhanced with dilated convolution. A gated cross-band interaction module is further used to exchange complementary information among different frequency branches.

## Released Components

This repository contains the following components.

### 1. Model Code

The core MAVSS-SR architecture is provided in:

```text
models/mavss_sr_arch.py
```

The model code includes:

- band separation;
- visual state space modeling;
- frequency gated modulation;
- deformable convolution branch;
- dilated convolution branch;
- multi-band attention module;
- gated cross-band interaction;
- MAVSS-SR reconstruction network.

### 2. Model Wrapper

The model wrapper is provided in:

```text
models/mavss_sr_model.py
```

It contains the model-side logic used for validation, image padding, tile-based inference, metric calculation, and result saving.

### 3. Dataset Code

The paired image dataset code and metadata file are provided in:

```text
data/imagenet_paired_dataset.py
data/meta_info/meta_info_DF2Ksub_GT.txt
```

Dataset paths can be modified directly in the training and testing `.yml` files.

### 4. Training and Testing Configuration Files

Training configs:

```text
options/train/train_MAVSS_SR_x2.yml
options/train/train_MAVSS_SR_x3.yml
options/train/train_MAVSS_SR_x4.yml
```

Testing configs:

```text
options/test/test_MAVSS_SR_x2.yml
options/test/test_MAVSS_SR_x3.yml
options/test/test_MAVSS_SR_x4.yml
```

These configuration files define dataset paths, model parameters, optimizer settings, learning rate schedules, validation settings, pretrained checkpoint paths, and result-saving options.

### 5. Complexity Script

The VSSM and window-attention complexity comparison script is provided in:

```text
tools/test_attention_vs_vssm_complexity_256.py
```

## Dependencies

Please prepare the runtime environment according to your training and testing framework. The released model code uses the following main Python packages:

```text
torch
torchvision
einops
mamba-ssm
causal-conv1d
thop
```

## How to Use

Place the released model, dataset, and configuration files into your project, then register or import the custom modules according to your own training framework.

A typical project layout can be:

```text
project/
  models/
    mavss_sr_arch.py
    mavss_sr_model.py
  data/
    imagenet_paired_dataset.py
    meta_info/
      meta_info_DF2Ksub_GT.txt
  options/
    train/
    test/
  tools/
```

Update dataset paths and pretrained model paths in the corresponding `.yml` files before running experiments.

## Training

Example config files:

```text
options/train/train_MAVSS_SR_x2.yml
options/train/train_MAVSS_SR_x3.yml
options/train/train_MAVSS_SR_x4.yml
```

Use these files with your training entry script.

## Testing

Example config files:

```text
options/test/test_MAVSS_SR_x2.yml
options/test/test_MAVSS_SR_x3.yml
options/test/test_MAVSS_SR_x4.yml
```

Before testing, update `pretrain_network_g` in the corresponding `.yml` file to the path of the released pretrained checkpoint.

## Repository Structure

```text
MAVSS-SR/
  README.md
  data/
    imagenet_paired_dataset.py
    meta_info/
      meta_info_DF2Ksub_GT.txt
  models/
    mavss_sr_arch.py
    mavss_sr_model.py
  options/
    train/
      train_MAVSS_SR_x2.yml
      train_MAVSS_SR_x3.yml
      train_MAVSS_SR_x4.yml
    test/
      test_MAVSS_SR_x2.yml
      test_MAVSS_SR_x3.yml
      test_MAVSS_SR_x4.yml
  tools/
    test_attention_vs_vssm_complexity_256.py
```
