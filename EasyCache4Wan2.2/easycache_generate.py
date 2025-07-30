# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import save_video, str2bool

import gc
from contextlib import contextmanager
import torchvision.transforms.functional as TF
import torch.cuda.amp as amp
import numpy as np
import math
from tqdm import tqdm
from time import time

from wan.modules.model import sinusoidal_embedding_1d
from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                  get_sampling_sigmas, retrieve_timesteps)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

EXAMPLE_PROMPT = {
    "t2v-A14B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "i2v-A14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
    "ti2v-5B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
}


def _prepare_model_for_timestep(self, t, boundary, offload_model):
    r"""
    Prepares and returns the required model for the current timestep.
    Args:
        t (torch.Tensor):
            current timestep.
        boundary (`int`):
            The timestep threshold. If `t` is at or above this value,
            the `high_noise_model` is considered as the required model.
        offload_model (`bool`):
            A flag intended to control the offloading behavior.
    Returns:
        torch.nn.Module:
            The active model on the target device for the current timestep.
    """
    if t.item() >= boundary:
        required_model_name = 'high_noise_model'
        offload_model_name = 'low_noise_model'
    else:
        required_model_name = 'low_noise_model'
        offload_model_name = 'high_noise_model'
    if offload_model or self.init_on_cpu:
        if next(getattr(
                self,
                offload_model_name).parameters()).device.type == 'cuda':
            getattr(self, offload_model_name).to('cpu')
        if next(getattr(
                self,
                required_model_name).parameters()).device.type == 'cpu':
            getattr(self, required_model_name).to(self.device)
    return getattr(self, required_model_name)

def easycache_forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
):
    """
    Args:
        x (List[Tensor]): List of input video tensors with shape [C_in, F, H, W]
        t (Tensor): Diffusion timesteps tensor of shape [B]
        context (List[Tensor]): List of text embeddings each with shape [L, C]
        seq_len (int): Maximum sequence length for positional encoding
        clip_fea (Tensor, optional): CLIP image features for image-to-video mode
        y (List[Tensor], optional): Conditional video inputs for image-to-video mode
    Returns:
        List[Tensor]: List of denoised video tensors with original input shapes
    """
    if self.model_type == 'i2v':
        assert y is not None

    # Store original raw input for end-to-end caching
    raw_input = [u.clone() for u in x]

    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # Track which type of step (even=condition, odd=uncondition)
    self.is_even = (self.cnt % 2 == 0)

    # Only make decision on even (condition) steps
    if self.is_even:
        # Always compute first ret_steps and last steps
        if self.cnt < self.ret_steps or self.cnt >= (
                ((getattr(self, "low_start_step", None) is not None and getattr(self, "is_high_noise", False)) and (
                        self.low_start_step - 1) * 2 - 2) or
                ((getattr(self, "low_start_step", None) is not None and not getattr(self, "is_high_noise", False)) and (
                        self.num_steps - self.low_start_step) * 2 - 2) or
                (self.num_steps * 2 - 2)
        ):
            self.should_calc_current_pair = True
            self.accumulated_error_even = 0
        else:
            # Check if we have previous step data for comparison
            if hasattr(self, 'previous_raw_input_even') and hasattr(self, 'previous_raw_output_even') and \
                    self.previous_raw_input_even is not None and self.previous_raw_output_even is not None:
                # Calculate input changes
                raw_input_change = torch.cat([
                    (u - v).flatten() for u, v in zip(raw_input, self.previous_raw_input_even)
                ]).abs().mean()

                # Compute predicted change if we have k factors
                if hasattr(self, 'k') and self.k is not None:
                    # Calculate output norm for relative comparison
                    output_norm = torch.cat([
                        u.flatten() for u in self.previous_raw_output_even
                    ]).abs().mean()
                    pred_change = self.k * (raw_input_change / output_norm)
                    combined_pred_change = pred_change
                    # Accumulate predicted error
                    if not hasattr(self, 'accumulated_error_even'):
                        self.accumulated_error_even = 0
                    self.accumulated_error_even += combined_pred_change
                    # Decide if we need full calculation
                    if self.accumulated_error_even < self.thresh:
                        self.should_calc_current_pair = False
                    else:
                        self.should_calc_current_pair = True
                        self.accumulated_error_even = 0
                else:
                    # First time after ret_steps or missing k factors, need to calculate
                    self.should_calc_current_pair = True
            else:
                # No previous data yet, must calculate
                self.should_calc_current_pair = True

        # Store current input state
        self.previous_raw_input_even = [u.clone() for u in raw_input]

    # Check if we can use cached output and return early
    if self.is_even and not self.should_calc_current_pair and \
            hasattr(self, 'previous_raw_output_even') and self.previous_raw_output_even is not None:
        # Use cached output directly
        self.cnt += 1
        return [(u + v).float() for u, v in zip(raw_input, self.cache_even)]

    elif not self.is_even and not self.should_calc_current_pair and \
            hasattr(self, 'previous_raw_output_odd') and self.previous_raw_output_odd is not None:
        # Use cached output directly
        self.cnt += 1
        # return [u.float() for u in self.previous_raw_output_odd]
        return [(u + v).float() for u, v in zip(raw_input, self.cache_odd)]

    # Continue with normal processing since we need to calculate
    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                  dim=1) for u in x
    ])

    # time embeddings
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast('cuda', dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t).unflatten(0, (bt, seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        context = torch.concat([context_clip, context], dim=1)

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)

    # Apply transformer blocks
    for block in self.blocks:
        x = block(x, **kwargs)

    # Apply head
    x = self.head(x, e)

    # Unpatchify
    output = self.unpatchify(x, grid_sizes)

    # Update cache and calculate change rates if needed
    if self.is_even:  # Condition path
        # If we have previous output, calculate k factors for future predictions
        if hasattr(self, 'previous_raw_output_even') and self.previous_raw_output_even is not None:
            # Calculate output change at the raw level
            output_change = torch.cat([
                (u - v).flatten() for u, v in zip(output, self.previous_raw_output_even)
            ]).abs().mean()

            # Check if we have previous input state for comparison
            if hasattr(self, 'prev_prev_raw_input_even') and self.prev_prev_raw_input_even is not None:
                # Calculate input change
                input_change = torch.cat([
                    (u - v).flatten() for u, v in zip(
                        self.previous_raw_input_even, self.prev_prev_raw_input_even
                    )
                ]).abs().mean()

                self.k = output_change / input_change

                # Update history
        self.prev_prev_raw_input_even = getattr(self, 'previous_raw_input_even', None)
        self.previous_raw_output_even = [u.clone() for u in output]
        self.cache_even = [u - v for u, v in zip(output, raw_input)]

    else:  # Uncondition path
        # Store output for unconditional path
        self.previous_raw_output_odd = [u.clone() for u in output]
        self.cache_odd = [u - v for u, v in zip(output, raw_input)]

    # Update counter
    self.cnt += 1
    return [u.float() for u in output]


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]

    if args.task == "i2v-A14B":
        assert args.image is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")
    parser.add_argument(
        "--thresh",
        type=float,
        default=0.05,
        help="Threshold for EasyCache decision making")
    parser.add_argument(
        "--thresh_high",
        type=float,
        default=0.08,
        help="Threshold for EasyCache decision making")
    parser.add_argument(
        "--thresh_low",
        type=float,
        default=0.07,
        help="Threshold for EasyCache decision making")
    parser.add_argument(
        "--ret_steps",
        type=int,
        default=8,
        help="Number of steps to retain in cache")
    parser.add_argument(
        "--ret_steps_high",
        type=int,
        default=10,
        help="Number of steps to retain in cache")
    parser.add_argument(
        "--ret_steps_low",
        type=int,
        default=2,
        help="Number of steps to retain in cache")

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
                args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
                args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    # prompt extend
    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt_output = prompt_expander(
                args.prompt,
                image=img,
                tar_lang=args.prompt_extend_target_lang,
                seed=args.base_seed)
            if prompt_output.status == False:
                logging.info(
                    f"Extending prompt failed: {prompt_output.message}")
                logging.info("Falling back to original prompt.")
                input_prompt = args.prompt
            else:
                input_prompt = prompt_output.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")

    if "t2v" in args.task:
        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )
        # TODO: has not been tested for MoE models
        # EasyCache setup
        wan_t2v.__class__.low_start_step = 0
        # low_noise_model
        wan_t2v.low_noise_model.__class__.forward = easycache_forward
        wan_t2v.low_noise_model.__class__.cnt = 0
        wan_t2v.low_noise_model.__class__.num_steps = args.sample_steps
        wan_t2v.low_noise_model.__class__.thresh = args.thresh_low
        wan_t2v.low_noise_model.__class__.accumulated_error_even = 0
        wan_t2v.low_noise_model.__class__.should_calc_current_pair = True
        wan_t2v.low_noise_model.__class__.k = None
        wan_t2v.low_noise_model.__class__.previous_raw_input_even = None
        wan_t2v.low_noise_model.__class__.previous_raw_output_even = None
        wan_t2v.low_noise_model.__class__.previous_raw_output_odd = None
        wan_t2v.low_noise_model.__class__.prev_prev_raw_input_even = None
        wan_t2v.low_noise_model.__class__.cache_even = None
        wan_t2v.low_noise_model.__class__.cache_odd = None
        wan_t2v.low_noise_model.__class__.ret_steps = args.ret_steps_low * 2
        wan_t2v.low_noise_model.__class__.is_high_noise = False
        # high_noise_model
        wan_t2v.high_noise_model.__class__.forward = easycache_forward
        wan_t2v.high_noise_model.__class__.cnt = 0
        wan_t2v.high_noise_model.__class__.num_steps = args.sample_steps
        wan_t2v.high_noise_model.__class__.thresh = args.thresh_high
        wan_t2v.high_noise_model.__class__.accumulated_error_even = 0
        wan_t2v.high_noise_model.__class__.should_calc_current_pair = True
        wan_t2v.high_noise_model.__class__.k = None
        wan_t2v.high_noise_model.__class__.previous_raw_input_even = None
        wan_t2v.high_noise_model.__class__.previous_raw_output_even = None
        wan_t2v.high_noise_model.__class__.previous_raw_output_odd = None
        wan_t2v.high_noise_model.__class__.prev_prev_raw_input_even = None
        wan_t2v.high_noise_model.__class__.cache_even = None
        wan_t2v.high_noise_model.__class__.cache_odd = None
        wan_t2v.high_noise_model.__class__.ret_steps = args.ret_steps_high * 2
        wan_t2v.high_noise_model.__class__.is_high_noise = True

        logging.info(f"Generating video ...")
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

    elif "ti2v" in args.task:
        logging.info("Creating WanTI2V pipeline.")
        wan_ti2v = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        wan_ti2v.model.__class__.forward = easycache_forward
        wan_ti2v.model.__class__.cnt = 0
        wan_ti2v.model.__class__.num_steps = args.sample_steps * 2
        wan_ti2v.model.__class__.thresh = args.thresh
        wan_ti2v.model.__class__.accumulated_error_even = 0
        wan_ti2v.model.__class__.should_calc_current_pair = True
        wan_ti2v.model.__class__.k = None
        wan_ti2v.model.__class__.previous_raw_input_even = None
        wan_ti2v.model.__class__.previous_raw_output_even = None
        wan_ti2v.model.__class__.previous_raw_output_odd = None
        wan_ti2v.model.__class__.prev_prev_raw_input_even = None
        wan_ti2v.model.__class__.cache_even = None
        wan_ti2v.model.__class__.cache_odd = None

        wan_ti2v.model.__class__.ret_steps = args.ret_steps * 2

        logging.info(f"Generating video ...")
        video = wan_ti2v.generate(
            args.prompt,
            img=img,
            size=SIZE_CONFIGS[args.size],
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
    else:
        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )
        # TODO: has not been tested for MoE models
        # EasyCache setup
        # low_noise_model
        wan_i2v.low_noise_model.__class__.forward = easycache_forward
        wan_i2v.low_noise_model.__class__.cnt = 0
        wan_i2v.low_noise_model.__class__.num_steps = args.sample_steps * 2
        wan_i2v.low_noise_model.__class__.thresh = args.thresh
        wan_i2v.low_noise_model.__class__.accumulated_error_even = 0
        wan_i2v.low_noise_model.__class__.should_calc_current_pair = True
        wan_i2v.low_noise_model.__class__.k = None
        wan_i2v.low_noise_model.__class__.previous_raw_input_even = None
        wan_i2v.low_noise_model.__class__.previous_raw_output_even = None
        wan_i2v.low_noise_model.__class__.previous_raw_output_odd = None
        wan_i2v.low_noise_model.__class__.prev_prev_raw_input_even = None
        wan_i2v.low_noise_model.__class__.cache_even = None
        wan_i2v.low_noise_model.__class__.cache_odd = None
        wan_i2v.low_noise_model.__class__.ret_steps = args.ret_steps_low * 2
        wan_i2v.low_noise_model.__class__.cutoff_steps = args.cutoff_steps if args.cutoff_steps is not None else args.sample_steps * 2 - 2
        # high_noise_model
        wan_i2v.high_noise_model.__class__.forward = easycache_forward
        wan_i2v.high_noise_model.__class__.cnt = 0
        wan_i2v.high_noise_model.__class__.num_steps = args.sample_steps * 2
        wan_i2v.high_noise_model.__class__.thresh = args.thresh
        wan_i2v.high_noise_model.__class__.accumulated_error_even = 0
        wan_i2v.high_noise_model.__class__.should_calc_current_pair = True
        wan_i2v.high_noise_model.__class__.k = None
        wan_i2v.high_noise_model.__class__.previous_raw_input_even = None
        wan_i2v.high_noise_model.__class__.previous_raw_output_even = None
        wan_i2v.high_noise_model.__class__.previous_raw_output_odd = None
        wan_i2v.high_noise_model.__class__.prev_prev_raw_input_even = None
        wan_i2v.high_noise_model.__class__.cache_even = None
        wan_i2v.high_noise_model.__class__.cache_odd = None
        wan_i2v.high_noise_model.__class__.ret_steps = args.ret_steps_high * 2
        wan_i2v.high_noise_model.__class__.cutoff_steps = args.cutoff_steps if args.cutoff_steps is not None else args.sample_steps * 2 - 2

        logging.info("Generating video ...")
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.mp4'
            args.save_file = f"{args.task}_{args.size.replace('*', 'x') if sys.platform == 'win32' else args.size}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}" + suffix

        logging.info(f"Saving generated video to {args.save_file}")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
    del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
