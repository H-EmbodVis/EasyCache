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

This document provides the implementation for accelerating the [**Wan2.2**](https://github.com/Wan-Video/Wan2.2) model using **EasyCache**. 
> This is a nightly preview version that has only been tested on Wan2.2-TI2V-5B.

### ✨ Visual Comparison

**Prompt: "A campfire burns in a sunlit forest clearing, with bright sparks occasionally leaping out."**

| Wan2.1-TI2V-5B (T2V task, 720p, H20) | EasyCache (Ours, 720p, H20) |
| :---: | :---: |
| ![Baseline Video](./videos/ti2v-5B-t2v-gt.gif) | ![Our Video](./videos/ti2v-5B-t2v-ours.gif) |
| **Inference Time: ~578s** | **Inference Time: ~255s (~2.3x Speedup)** |


**Prompt: "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."**

| Wan2.2-TI2V-5B (I2V task, 720p, H20) | EasyCache (Ours, 720p, H20) |
| :---: | :---: |
| ![Baseline Video](./videos/ti2v-5B-i2v-gt.gif) | ![Our Video](./videos/ti2v-5B-i2v-ours.gif) |
| **Inference Time: ~560s** | **Inference Time: ~224s (~2.5x Speedup)** |

---

### 🚀 Usage Instructions

**a. Prerequisites** ⚙️

Before you begin, please follow the instructions in the [official Wan2.2 repository](https://github.com/Wan-Video/Wan2.2) to configure the required environment and download the pretrained model weights.

**b. Copy Files** 📂

Copy `easycache_generate.py` into the root directory of your local `Wan2.2` project.

**c. Run Inference** ▶️

#### **1. EasyCache Acceleration for Wan2.1 TI2V-5B (T2V task)**

Execute the following command from the root of the `Wan2.2` project to generate a video. You can also specify your own custom prompts.

```bash
python easycache_generate.py \
  --task ti2v-5B \
  --size "1280*704" \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --prompt "A campfire burns in a sunlit forest clearing, with bright sparks occasionally leaping out." \
  --base_seed 42
```
#### **2. EasyCache Acceleration for Wan2.2 TI2V-5B (I2V task)**
Execute the following command from the root of the `Wan2.2` project to generate a video. You can also specify your own custom prompts and images.

```bash
python easycache_generate.py \
  --task ti2v-5B \
  --size "1280*704" \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --image examples/i2v_input.JPG \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
  --base_seed 42
```

## 🌹 Acknowledgements
We would like to thank the contributors to the [Wan2.2](https://github.com/Wan-Video/Wan2.1) repository, for the open research and exploration. We also extend our gratitude to [@AChowdhury1211](https://github.com/AChowdhury1211) for the assistance.

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
