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

(\*) Equal contribution. (†) Project leader.

  [![Project](https://img.shields.io/badge/Homepage-project-orange.svg?logo=googlehome)](https://H-EmbodVis.github.io/EasyCache/)
  [![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/LMD0311/EasyCache/blob/main/LICENSE)

</div>

## 📰 News
- **[2025/07/02]** EasyCache for [**HunyuanVideo**](https://github.com/H-EmbodVis/EasyCache/tree/main/HunyuanVideo-EasyCache) is released.

## Abstract
Video generation models have demonstrated remarkable performance, yet their broader adoption remains constrained by slow inference speeds and substantial computational costs, primarily due to the iterative nature of the denoising process. Addressing this bottleneck is essential for democratizing advanced video synthesis technologies and enabling their integration into real-world applications. This work proposes EasyCache, a training-free acceleration framework for video diffusion models. EasyCache introduces a lightweight, runtime-adaptive caching mechanism that dynamically reuses previously computed transformation vectors, avoiding redundant computations during inference. Unlike prior approaches, EasyCache requires no offline profiling, pre-computation, or extensive parameter tuning. We conduct comprehensive studies on various large-scale video generation models, including OpenSora, Wan2.1, and HunyuanVideo. Our method achieves leading acceleration performance, reducing inference time by up to 2.1-3.3× compared to the original baselines while maintaining high visual fidelity with a significant up to 36% PSNR improvement compared to the previous SOTA method. This improvement makes our EasyCache a efficient and highly accessible solution for high-quality video generation in both research and practical applications.


## 🚀 Main Performance

We validated the performance of EasyCache on leading video generation models and compared it with other state-of-the-art training-free acceleration methods.

### Comparison with SOTA Methods 

Tested on Vbench prompts with NVIDIA A800.

**Performance on Wan2.1-1.3B:**

| Method | Latency (s)↓ | Speedup ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Wan2.1 (Baseline) | 175.35 | 1.00x | - | - | - |
| PAB | 102.03 | 1.72x | 22.57 | 0.6484 | 0.3010 |
| TeaCache | 87.77 | 2.00x | 25.24 | 0.8057 | 0.1277 |
| **EasyCache (Ours)** | **69.11** | **2.54x** | **24.74** | **0.8337** | **0.0952** |

**Performance on HunyuanVideo:**
| Method | Latency (s)↓ | Speedup ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| HunyuanVideo (Baseline) | 1124.30 | 1.00x | - | - | - |
| PAB | 958.23 | 1.17x | 18.58 | 0.7023 | 0.3827 |
| TeaCache | 674.04 | 1.67x | 23.85 | 0.8185 | 0.1730 |
| SVG | 802.70 | 1.40x | 26.57 | 0.8596 | 0.1368 |
| **EasyCache (Ours)** | **507.97** | **2.21x** | **32.66** | **0.9313** | **0.0533** |

### Compatibility with Other Acceleration Techniques

EasyCache is orthogonal to other acceleration techniques, such as the efficient attention mechanism SVG, and can be combined with them for even greater performance gains.

**Combined Performance on HunyuanVideo (720p):**
*Tested on NVIDIA H20 GPUs.*
| Method | Latency (s)↓ | Speedup ↑ | PSNR (dB) ↑ |
|:---:|:---:|:---:|:---:|
| Baseline | 6594s | 1.00x | - |
| SVG | 3474s | 1.90x | 27.56 |
| SVG (w/ **Ours**) | **1981s** | **3.33x** | **27.26** |


## 🎬 Visual Comparisons
Video synchronization issues may occur due to network load, for improved visualization, see the [project page](https://H-EmbodVis.github.io/EasyCache/)

**Prompt: "Grassland at dusk, wild horses galloping, golden light flickering across manes."**
*(HunyuanVideo)*

| Baseline | Ours (2.3x) | TeaCache (1.7x) | PAB (1.2x) |
| :---: | :---: | :---: | :---: |
| ![Baseline Video](./static/videos/Comparison/gt/6.gif) | ![Our Video](./static/videos/Comparison/our/6.gif) | ![TeaCache Video](./static/videos/Comparison/teacache/6.gif) | ![PAB Video](./static/videos/Comparison/pab/6.gif) |

**Prompt: "A top-down view of a barista creating latte art, skillfully pouring milk to form the letters 'TPAMI' on coffee."**
*(Wan2.1-14B)*

| Baseline | Ours (2.34x) | TeaCache (1.46x) | PAB (1.87x) |
| :---: | :---: | :---: | :---: |
| ![Baseline Latte](./static/videos/Comparison/gt/7.gif) | ![Our Latte](./static/videos/Comparison/our/7.gif) | ![TeaCache Latte](./static/videos/Comparison/teacache/7.gif) | ![PAB Latte](./static/videos/Comparison/pab/7.gif) |

## 🛠️ Usage
Detailed instructions for each supported model are provided in their respective directories. We are continuously working to extend support to more models.

### HunyuanVideo
1. **Prerequisites**: Set up the environment and download weights from the official HunyuanVideo repository.
2. **Copy Files**: Place the EasyCache script files into your local HunyuanVideo project directory.
3. **Run**: Execute the provided Python script to run inference with acceleration.
For complete instructions, please refer to the [README](./HunyuanVideo-EasyCache/README.md).

## 🎯 To Do

- [x] Support HunyuanVideo
- [x] Support Sparse-VideoGen on HunyuanVideo
- [ ] Support Wan2.1 T2V

## Acknowledgements
We would like to thank the contributors to the [Wan2.1](https://github.com/Wan-Video/Wan2.1), [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo), [OpenSora](https://github.com/hpcaitech/Open-Sora), and [SVG](https://github.com/svg-project/Sparse-VideoGen) repositories, for their open research and exploration.
