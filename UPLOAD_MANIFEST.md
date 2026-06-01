# MAVSS-SR Public Code Manifest

This directory is a cleaned public-release version of the MAVSS-SR code.

## Included

- Core architecture:
  - `hat/archs/hat_arch.py`
  - `hat/archs/__init__.py`
- Training and testing entry points:
  - `hat/train.py`
  - `hat/test.py`
  - `hat/__init__.py`
  - `hat/version.py`
- Model wrapper:
  - `hat/models/hat_model.py`
  - `hat/models/__init__.py`
- Dataset code:
  - `hat/data/imagenet_paired_dataset.py`
  - `hat/data/meta_info/meta_info_DF2Ksub_GT.txt`
  - `hat/data/__init__.py`
- MAVSS-SR configs:
  - `options/train/train_MAVSS_SR_x2.yml`
  - `options/train/train_MAVSS_SR_x3.yml`
  - `options/train/train_MAVSS_SR_x4.yml`
  - `options/test/test_MAVSS_SR_x2.yml`
  - `options/test/test_MAVSS_SR_x3.yml`
  - `options/test/test_MAVSS_SR_x4.yml`
- Complexity script:
  - `tools/test_attention_vs_vssm_complexity_256.py`
- Project files:
  - `README.md`
  - `LICENSE`
  - `requirements.txt`
  - `environment.yml`
  - `setup.py`
  - `setup.cfg`
  - `VERSION`
  - `.gitignore`

## Excluded

- `.eggs/`
- `.idea/`
- `hat.egg-info/`
- `__pycache__/`
- `*.pyc`
- backup files such as `*.bak`
- `datasets/` contents
- `experiments/` contents
- `results/` contents
- local checkpoints such as `*.pth`, `*.pt`, `*.ckpt`
- unused HAT-S/HAT-L/Real-HAT configs and debug scripts

