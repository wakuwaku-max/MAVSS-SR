# MAVSS-SR

Official code release for **MAVSS-SR: An Image Super-Resolution Network Using Multi-Band Attention and Visual State Space**.

MAVSS-SR is an image super-resolution network built with multi-band feature separation, band-specific feature modeling, visual state space modeling, and gated cross-band interaction.

## Main Components

- Band separation for low-, mid-, and high-frequency feature branches
- Multi-band attention module (MBAM)
- Visual state space model (VSSM) for high-frequency feature modeling
- Deformable convolution branch for mid-frequency texture modeling
- Dilated convolution branch for low-frequency structure modeling
- Gated cross-band interaction module (GCBI)
- MAVSS-SR reconstruction network for image super-resolution

The core implementation is in:

```text
hat/archs/hat_arch.py
```

## Environment

Create the environment with either:

```bash
conda env create -f environment.yml
```

or install the Python dependencies:

```bash
pip install -r requirements.txt
python setup.py develop
```

The model uses PyTorch, BasicSR-style training/testing utilities, torchvision deformable convolution, and Mamba selective scan.

## Dataset Preparation

Place datasets under:

```text
datasets/
```

Expected benchmark dataset layout follows the BasicSR paired-image format, for example:

```text
datasets/
  Set5/
    GTmod4/
    LRbicx4/
  Set14/
    GTmod4/
    LRbicx4/
  BSD100/
    GTmod4/
    LRbicx4/
  Urban100/
    GTmod4/
    LRbicx4/
  Manga109/
    GTmod4/
    LRbicx4/
```

Training metadata is provided in:

```text
hat/data/meta_info/meta_info_DF2Ksub_GT.txt
```

## Pretrained Models

Place pretrained weights under:

```text
pretrained/
```

or update the `pretrain_network_g` path in the corresponding test config.

## Testing

Run x2, x3, or x4 testing with:

```bash
python hat/test.py -opt options/test/test_MAVSS_SR_x2.yml
python hat/test.py -opt options/test/test_MAVSS_SR_x3.yml
python hat/test.py -opt options/test/test_MAVSS_SR_x4.yml
```

Results are saved under:

```text
results/
```

## Training

Run training with:

```bash
python hat/train.py -opt options/train/train_MAVSS_SR_x2.yml
python hat/train.py -opt options/train/train_MAVSS_SR_x3.yml
python hat/train.py -opt options/train/train_MAVSS_SR_x4.yml
```

For distributed training, launch `hat/train.py` with the PyTorch distributed launcher and the same config files.

## Complexity Evaluation

The VSSM and window-attention complexity comparison script is provided at:

```text
tools/test_attention_vs_vssm_complexity_256.py
```

Run:

```bash
python tools/test_attention_vs_vssm_complexity_256.py
```

## Repository Structure

```text
MAVSS-SR/
  README.md
  LICENSE
  requirements.txt
  environment.yml
  setup.py
  setup.cfg
  VERSION
  hat/
    train.py
    test.py
    archs/
      hat_arch.py
    models/
      hat_model.py
    data/
      imagenet_paired_dataset.py
      meta_info/
  options/
    train/
    test/
  tools/
  pretrained/
  datasets/
  results/
```

## Acknowledgements

This project is built on the BasicSR training/testing framework and reuses parts of the HAT project structure. We also acknowledge the Mamba selective scan implementation used by the VSSM module.

## Citation

If this code is useful for your work, please cite:

```bibtex
@article{mavsssr2026,
  title={MAVSS-SR: An Image Super-Resolution Network Using Multi-Band Attention and Visual State Space},
  author={Yang, Xin and Li, Hui and Liu, Jiufu and Hong, Chaming},
  year={2026}
}
```

