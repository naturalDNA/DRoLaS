<div align="center">
<h1> DRoLaS: Diffusion-based Coarse-to-Fine Conditional Synthesis of Hierarchical Road Layouts
 </h1>
</div>


<div align="center">
  📃 <a href="https://dl.acm.org/doi/abs/10.1145/3731715.3733316">Paper</a>
</div>

## 📝 Introduction

Road layouts embody a city’s spatial structure, making their design fundamental to urban planning applications. While deep learning-based methods have demonstrated potential in road layout synthesis, they struggle to adapt to diverse input conditions and capture the hierarchical nature of real-world road networks. In this paper, we present DRoLaS, a novel diffusion-based method for conditional hierarchical road layout synthesis. Our method incorporates a local adaptation module and a global perception module to enable effective conditional control. Meanwhile, we address data imbalance regarding road classes through a class-weighted denoising loss, and introduce a connectivity refinement strategy to enhance the quality of generated layouts. Experimental results on a self-constructed dataset demonstrate that our DRoLaS successfully synthesizes high quality road layouts that reflect real-world structures and hierarchical patterns while effectively responding to input conditions.
<div align="center">
<img src="./assets/teaser.jpg" style="width: 100%;height: 100%">
</div>

## 🎉 What's New
- **[2025.04.18]** 📣 DRoLaS has been accepted for ACM International Conference on Multimedia Retrieval 2025!

## 📄 Table of Contents

<details>
<summary>
Click to expand the table of contents
</summary>

- [📝 Introduction](#-introduction)
  - [Diagnostic Framework and Scope](#diagnostic-framework-and-scope)
- [🎉 What's New](#-whats-new)
- [📄 Table of Contents](#-table-of-contents)
- [🔧 Setup Environment](#-setup-environment)
- [📚 Data](#-data)
- [🚀 Code Workflow](#-code-workflow)
- [🔁 Reproduction](#-reproduction)
- [📖 Citation](#-citation)

</details>

## 🔧 Setup Environment

```shell
conda env create -f environment.yml
conda activate pytorch
```

## 🚀 Code Workflow

### 1. Configuration
Edit `config.py` to set your parameters.

### 2. Training (Run in order)
```shell
python resnet_vae.py          # Train VAE
python main_region.py         # Train region diffusion
python mask_detector_unet.py  # Train mask detector
python main_refine.py         # Train refinement model
```

### 3. Evaluation

```shell
python test.py                        # Compute FID/KID
python eval.py                        # Compute other metrics
```



## 📚 Data

Dataset available upon request from the author.


## 📖 Citation

If you find this repository useful, please consider giving star and citing our paper:

```plaintext

@inproceedings{dong2025drolas,
  title={DRoLaS: Diffusion-Based Coarse-to-Fine Conditional Synthesis of Hierarchical Road Layouts},
  author={Dong, Shenao and Li, Weitao and Li, Bo and Li, Long and Shen, Junao and Feng, Tian},
  booktitle={Proceedings of the 2025 International Conference on Multimedia Retrieval},
  pages={237--245},
  year={2025}
}
```
