# MAVSS-SR: Multi-Band Attention and Visual State Space for Image Super-Resolution

## Project Overview

This repository provides the released code for **MAVSS-SR: An Image Super-Resolution Network Using Multi-Band Attention and Visual State Space**.

MAVSS-SR is designed for single-image super-resolution. It improves reconstruction quality by separating features into multiple frequency bands and applying band-specific processing. High-frequency details are modeled with a visual state space module, mid-frequency texture features are modeled with deformable convolution, and low-frequency structural information is modeled with dilated convolution. A gated cross-band interaction module is used to exchange complementary information among different frequency bands.

The project is built upon the open-source image restoration framework **BasicSR**:

> BasicSR: https://github.com/XPixelGroup/BasicSR

BasicSR provides the training, validation, testing, logging, metric calculation, and registry mechanisms. This repository releases the MAVSS-SR model code and experiment configuration files that can be used with a BasicSR-style workflow.

## Released Components

This repository contains the following components.

### 1. Model Code

The core MAVSS-SR architecture is provided in:

```text
models/mavss_sr_arch.py
```

It includes the main modules used by MAVSS-SR:

- band separation;
- visual state space modeling;
- frequency gated modulation;
- deformable convolution branch;
- dilated convolution branch;
- multi-band attention module;
- gated cross-band interaction;
- MAVSS-SR reconstruction network.

The architecture is registered in the BasicSR registry as `MAVSSSR`.

### 2. Model Wrapper

The BasicSR-style model wrapper is provided in:

```text
models/mavss_sr_model.py
```

It handles validation, image padding, tile-based inference, metric calculation, and result saving. The wrapper is registered in the BasicSR registry as `MAVSSSRModel`.

### 3. Dataset Code

The paired image dataset code and metadata file are provided in:

```text
data/imagenet_paired_dataset.py
data/meta_info/meta_info_DF2Ksub_GT.txt
```

Dataset paths can be changed directly in the training and testing `.yml` files.

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

These files define dataset paths, model parameters, optimizer settings, learning rate schedules, validation settings, pretrained checkpoint paths, and result-saving options.

### 5. Complexity Script

The VSSM and window-attention complexity comparison script is provided in:

```text
tools/test_attention_vs_vssm_complexity_256.py
```

## Framework Dependency

This project depends on BasicSR. Please install and configure BasicSR before running training or testing.

```bash
git clone https://github.com/XPixelGroup/BasicSR.git
cd BasicSR
pip install -r requirements.txt
python setup.py develop
```

Additional dependencies used by MAVSS-SR include:

```text
torch
torchvision
einops
mamba-ssm
causal-conv1d
thop
```

## How to Use

Copy or place the released files into a BasicSR-style project, then make sure the custom model, architecture, and dataset files are imported by the framework registry.

A typical placement is:

```text
basicsr/
  archs/
    mavss_sr_arch.py
  models/
    mavss_sr_model.py
  data/
    imagenet_paired_dataset.py
options/
  train/
  test/
```

Then run training or testing with the corresponding config files.

## Training

Example commands:

```bash
python basicsr/train.py -opt options/train/train_MAVSS_SR_x2.yml
python basicsr/train.py -opt options/train/train_MAVSS_SR_x3.yml
python basicsr/train.py -opt options/train/train_MAVSS_SR_x4.yml
```

For distributed training, use the PyTorch distributed launcher according to the BasicSR training instructions.

## Testing

Example commands:

```bash
python basicsr/test.py -opt options/test/test_MAVSS_SR_x2.yml
python basicsr/test.py -opt options/test/test_MAVSS_SR_x3.yml
python basicsr/test.py -opt options/test/test_MAVSS_SR_x4.yml
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

## Acknowledgements

This project is developed based on BasicSR. We also acknowledge the Mamba selective scan implementation used by the visual state space module.

## Citation

If this code is useful for your work, please cite:

```bibtex
@article{mavsssr2026,
  title={MAVSS-SR: An Image Super-Resolution Network Using Multi-Band Attention and Visual State Space},
  author={Yang, Xin and Li, Hui and Liu, Jiufu and Hong, Chaming},
  year={2026}
}
```
