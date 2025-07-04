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

This document provides the implementation for accelerating the [**HunyuanVideo**](https://github.com/Tencent/HunyuanVideo) model using **EasyCache**.

### ✨ Visual Comparison

EasyCache significantly accelerates inference speed while maintaining high visual fidelity.

**Prompt: "A cat walks on the grass, realistic style." (Base Acceleration)**

| HunyuanVideo (Baseline, 544p, H20) | EasyCache (Ours) |
| :---: | :---: |
| ![Baseline Video](./videos/baseline_544p.gif) | ![Our Video](./videos/easycache_544p.gif) |
| **Inference Time: ~2327s** | **Inference Time: ~1025s (2.3x Speedup)** |

**Prompt: "A young man at his 20s is sitting on a piece of cloud in the sky, reading a book." (SVG with EasyCache)**

| HunyuanVideo (Baseline, 720p, H20) | SVG with EasyCache (Ours) |
|:---:|:---:|
| ![Baseline 720p GIF](./videos/baseline_720p.gif) | ![EasyCache+SVG 720p GIF](./videos/svg_with_easycache_720p.gif) |
| Inference Time: ~6572s | Inference Time: ~1773s (3.71x Speedup) |


---

### 🚀 Usage Instructions

This section provides instructions for two settings: base acceleration with EasyCache alone and combined acceleration using EasyCache with SVG.

#### **1. Base Acceleration (EasyCache Only)**

**a. Prerequisites** ⚙️

Before you begin, please follow the instructions in the [official HunyuanVideo repository](https://github.com/Tencent/HunyuanVideo) to configure the required environment and download the pretrained model weights.

**b. Copy Files** 📂

Copy `easycache_sample_video.py` into the root directory of your local `HunyuanVideo` project.

**c. Run Inference** ▶️

Execute the following command from the root of the `HunyuanVideo` project to generate a video. To generate videos in 720p resolution, set the `--video-size` argument to `720 1280`. You can also specify your own custom prompts.

```bash
python3 easycache_sample_video.py \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --use-cpu-offload \
    --save-path ./results \
    --seed 42
```

#### **2. Combined Acceleration (SVG with EasyCache)**

**a. Prerequisites** ⚙️

Ensure you have set up the environments for both [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) and [SVG](https://github.com/svg-project/Sparse-VideoGen).

**b. Copy Files** 📂

Copy `hyvideo_svg_easycache.py` into the root directory of your local `HunyuanVideo` project.

**c. Run Inference** ▶️

Execute the following command to generate a 720p video using both SVG and EasyCache for maximum acceleration. You can also specify your own custom prompts.

```bash
python3 hyvideo_svg_easycache.py \
        --video-size 720 1280 \
        --video-length 129 \
        --infer-steps 50 \
        --prompt "A young man at his 20s is sitting on a piece of cloud in the sky, reading a book." \
        --embedded-cfg-scale 6.0 \
        --flow-shift 7.0 \
        --flow-reverse \
        --use-cpu-offload \
        --save-path ./results \
        --output_path ./results \
        --pattern "SVG" \
        --num_sampled_rows 64 \
        --sparsity 0.2 \
        --first_times_fp 0.055 \
        --first_layers_fp 0.025 \
        --record_attention \
        --seed 42
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
We would like to thank the contributors to the [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo), and [SVG](https://github.com/svg-project/Sparse-VideoGen) repositories, for their open research and exploration.

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
