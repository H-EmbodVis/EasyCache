<div align="center">
  <h1>Less is Enough: Training-Free Video Diffusion Acceleration via Runtime-Adaptive Caching</h1>

  <a href="https://lmd0311.github.io/" target="_blank" rel="noopener noreferrer">Xin Zhou</a><sup>1\*</sup>,
  <a href="https://dk-liang.github.io/" target="_blank" rel="noopener noreferrer">Dingkang Liang</a><sup>1\*</sup>,
Kaijin Chen<sup>1</sup>, Tianrui Feng<sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=PVMQa-IAAAAJ&hl=en" target="_blank" rel="noopener noreferrer">Xiwu Chen</a><sup>2</sup>, Hongkai Lin<sup>1</sup>, <br>
  <a href="https://scholar.google.com/citations?user=gdP9StQAAAAJ&hl=en" target="_blank" rel="noopener noreferrer">Yikang Ding</a><sup>2</sup>, Feiyang Tan<sup>2</sup>,
  <a href="https://scholar.google.com/citations?user=4uE10I0AAAAJ&hl=en" target="_blank" rel="noopener noreferrer">Hengshuang Zhao</a><sup>3</sup>,
  <a href="https://scholar.google.com/citations?user=UeltiQ4AAAAJ&hl=en" target="_blank" rel="noopener noreferrer">Xiang Bai</a><sup>1†</sup>

  <sup>1</sup> Huazhong University of Science and Technology, <sup>2</sup> MEGVII Technology, <sup>3</sup> University of Hong Kong <br>

(\*) Equal contribution. (†) Corresponding author.

  [![Project](https://img.shields.io/badge/Homepage-project-orange.svg?logo=googlehome)](https://H-EmbodVis.github.io/EasyCache/)
  [![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/LMD0311/EasyCache/blob/main/LICENSE)

</div>

---

This document provides the implementation for accelerating the [**Wan2.1**](https://github.com/Wan-Video/Wan2.1) model using **EasyCache**.

### ✨ Visual Comparison

EasyCache significantly accelerates inference speed while maintaining high visual fidelity.

**Prompt: "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."**

| Wan2.1-14B (Baseline, 720p, H20) | EasyCache (Ours, 720p, H20) |
| :---: | :---: |
| ![Baseline Video](./videos/gt_14b_720p.gif) | ![Our Video](./videos/easycache_14b_720p.gif) |
| **Inference Time: ~6862s** | **Inference Time: ~2884s (~2.4x Speedup)** |



---

### 🚀 Usage Instructions

#### **1. EasyCache Acceleration for Wan2.1 T2V**

**a. Prerequisites** ⚙️

Before you begin, please follow the instructions in the [official Wan2.1 repository](https://github.com/Wan-Video/Wan2.1) to configure the required environment and download the pretrained model weights.

**b. Copy Files** 📂

Copy `easycache_generate.py` into the root directory of your local `Wan2.1` project.

**c. Run Inference** ▶️

Execute the following command from the root of the `Wan2.1` project to generate a video. To generate videos in 720p resolution, set the `--size` argument to `1280*720`. You can also specify your own custom prompts.

```bash
python easycache_generate.py \
	--task t2v-14B \
	--size "1280*720" \
	--ckpt_dir ./Wan2.1-T2V-14B \
	--prompt "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about." \
	--base_seed 0
```


### 📊 Evaluating Video Similarity

We provide a simple script to quickly evaluate the similarity between two videos (e.g., the baseline result and your generated result) using common metrics.

**Usage**

```bash
# install required packages.
pip install lpips numpy tqdm torchmetrics

python tools/video_metrics.py --original_video video1.mp4 --generated_video video2.mp4
```

- `--original_video`: Path to the first video (e.g., the baseline).
- `--generated_video`: Path to the second video (e.g., the one generated with EasyCache).

## 🌹 Acknowledgements
We would like to thank the contributors to the [Wan2.1](https://github.com/Wan-Video/Wan2.1) repository, for the open research and exploration.

## 📖 Citation

If you find this repository useful in your research, please consider giving a star ⭐ and a citation.
```bibtex
@article{zhou2025easycache,
  title={Less is Enough: Training-Free Video Diffusion Acceleration via Runtime-Adaptive Caching},
  author={Zhou, Xin and Liang, Dingkang and Chen, Kaijin and and Feng, Tianrui and Chen, Xiwu and Lin, Hongkai and Ding, Yikang and Tan, Feiyang and Zhao, Hengshuang and Bai, Xiang},
  journal={arXiv preprint arXiv:2507.02860},
  year={2025}
}
```
