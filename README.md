# Dataset-Specific Watermarking

This repository contains the source code of the paper "I've Got Proof! Dataset-Specific Watermarking for Detecting Excessive Dataset Usage in Text-to-Image Diffusion Model Fine-Tuning". The proposed approach is "DSW: Dataset-Specific Watermarking".

# Requirement

- torch==1.12.0
- numpy==1.21.0
- torchvision==0.13.0

# Dataset

The experiments are evaluated on three datasets:

- pokemon-blip-captions  can be downloaded from [here](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions).
- simpsons-blip-captions  can be downloaded from [here](https://huggingface.co/datasets/Norod78/simpsons-blip-captions).
- FaceCaption-1M-image-text-pairs can be downloaded from [here](https://huggingface.co/datasets/OpenFace-CQUPT/FaceCaption-1M-image-text-pairs/viewer).

# Stable Diffusion 

The method of fine-tuning the Stable Diffusion model using LoRA can be found [here](https://github.com/justinpinkney/stable-diffusion).

# Experiment

### Train the encoder and decoder

```
python train.py 
```

