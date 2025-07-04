import os
import time
import math
import json
from pathlib import Path
from loguru import logger
from datetime import datetime

import torch
from svg.models.hyvideo.utils.file_utils import save_videos_grid
from svg.models.hyvideo.config import parse_args
from svg.models.hyvideo.inference import HunyuanVideoSampler
from torch.utils.data import Dataset, DataLoader
from typing import Any, List, Tuple, Optional, Union, Dict
import portalocker
from tqdm import tqdm


def get_cu_seqlens(text_mask, img_len):
    """Calculate cu_seqlens_q, cu_seqlens_kv using text_mask and img_len

    Args:
        text_mask (torch.Tensor): the mask of text
        img_len (int): the length of image

    Returns:
        torch.Tensor: the calculated cu_seqlens for flash attention
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens


@torch.compile()
def easycache_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,  # Now we don't use it.
        text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        return_dict: bool = True,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    torch.cuda.synchronize()
    start_time = time.time()

    out = {}
    raw_input = x.clone()
    img = x
    txt = text_states
    _, _, ot, oh, ow = x.shape
    tt, th, tw = (
        ot // self.patch_size[0],
        oh // self.patch_size[1],
        ow // self.patch_size[2],
    )

    # Prepare modulation vectors.
    vec = self.time_in(t)

    # text modulation
    vec = vec + self.vector_in(text_states_2)

    # guidance modulation
    if self.guidance_embed:
        if guidance is None:
            raise ValueError(
                "Didn't get guidance strength for guidance distilled model."
            )

        # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
        vec = vec + self.guidance_in(guidance)

    if self.cnt < self.ret_steps or self.cnt >= self.num_steps - 1:
        should_calc = True
        self.accumulated_error = 0
    else:
        # Check if previous inputs and outputs exist
        if hasattr(self, 'previous_raw_input') and hasattr(self, 'previous_output') \
                and self.previous_raw_input is not None and self.previous_output is not None:

            raw_input_change = (raw_input - self.previous_raw_input).abs().mean()

            if hasattr(self, 'k') and self.k is not None:

                output_norm = self.previous_output.abs().mean()
                pred_change = self.k * (raw_input_change / output_norm)
                self.accumulated_error += pred_change

                if self.accumulated_error < self.thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_error = 0
            else:
                should_calc = True
        else:
            should_calc = True

    self.previous_raw_input = raw_input.clone()  # (1, 16, 33, 68, 120)

    if not should_calc and self.cache is not None:
        result = raw_input + self.cache
        self.cnt += 1

        if self.cnt >= self.num_steps:
            self.cnt = 0

        torch.cuda.synchronize()
        end_time = time.time()
        self.total_time += (end_time - start_time)

        if return_dict:
            out["x"] = result
            return out
        return result

    img = self.img_in(img)
    if self.text_projection == "linear":
        txt = self.txt_in(txt)
    elif self.text_projection == "single_refiner":
        txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
    else:
        raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

    txt_seq_len = txt.shape[1]
    img_seq_len = img.shape[1]

    # Compute cu_squlens and max_seqlen for flash attention
    cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
    cu_seqlens_kv = cu_seqlens_q
    max_seqlen_q = img_seq_len + txt_seq_len
    max_seqlen_kv = max_seqlen_q

    freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

    # --------------------- Pass through DiT blocks ------------------------
    for _, block in enumerate(self.double_blocks):
        double_block_args = [
            img,
            txt,
            vec,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            freqs_cis,
            t,
        ]
        img, txt = block(*double_block_args)

    # Merge txt and img to pass through single stream blocks.
    x = torch.cat((img, txt), 1)
    if len(self.single_blocks) > 0:
        for _, block in enumerate(self.single_blocks):
            single_block_args = [
                x,
                vec,
                txt_seq_len,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                (freqs_cos, freqs_sin),
                t,
            ]
            x = block(*single_block_args)
    img = x[:, :img_seq_len, ...]

    # ---------------------------- Final layer ------------------------------
    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

    result = self.unpatchify(img, tt, th, tw)

    # store the cache for next step
    self.cache = result - raw_input
    if hasattr(self, 'previous_output') and self.previous_output is not None:
        output_change = (result - self.previous_output).abs().mean()
        if hasattr(self, 'prev_prev_raw_input') and self.prev_prev_raw_input is not None:
            input_change = (self.previous_raw_input - self.prev_prev_raw_input).abs().mean()
            self.k = output_change / input_change

    # update the previous state
    self.prev_prev_raw_input = getattr(self, 'previous_raw_input', None)
    self.previous_output = result.clone()

    self.cnt += 1
    if self.cnt >= self.num_steps:
        self.cnt = 0

    torch.cuda.synchronize()
    end_time = time.time()
    self.total_time += (end_time - start_time)

    if return_dict:
        out["x"] = result
        return out
    return result


def sparsity_to_width(sparsity, context_length, num_frame, frame_size):
    seq_len = context_length + num_frame * frame_size
    total_elements = seq_len ** 2

    sparsity = (sparsity * total_elements - 2 * seq_len * context_length) / total_elements

    width = seq_len * (1 - math.sqrt(1 - sparsity))
    width_frame = width / frame_size

    return width_frame


def main():
    args = parse_args()
    print(args)
    models_root_path = Path("./HunyuanVideo")
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Create save folder to save the samples
    os.makedirs(args.save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)

    # Get the updated args
    args = hunyuan_video_sampler.args

    # Sparsity Related
    transformer = hunyuan_video_sampler.pipeline.transformer
    for _, block in enumerate(transformer.double_blocks):
        block.sparse_args = args
    for _, block in enumerate(transformer.single_blocks):
        block.sparse_args = args
    transformer.sparse_args = args

    print(
        f"Memory: {torch.cuda.memory_allocated() // 1024 ** 2} / {torch.cuda.max_memory_allocated() // 1024 ** 2} MB before Inference")

    cfg_size, num_head, head_dim, dtype, device = 1, 24, 128, torch.bfloat16, "cuda"
    context_length, num_frame, frame_size = 256, 33, 3600

    # Calculation
    spatial_width = temporal_width = sparsity_to_width(args.sparsity, context_length, num_frame, frame_size)

    print(f"Spatial_width: {spatial_width}, Temporal_width: {temporal_width}. Sparsity: {args.sparsity}")

    save_path = args.output_path
    if args.pattern == "SVG":
        masks = ["spatial", "temporal"]
        def get_attention_mask(mask_name):

            context_length = 256
            num_frame = 33
            frame_size = 3600
            attention_mask = torch.zeros(
                (context_length + num_frame * frame_size, context_length + num_frame * frame_size), device="cpu")

            # TODO: fix hard coded mask
            if mask_name == "spatial":
                pixel_attn_mask = torch.zeros_like(attention_mask[:-context_length, :-context_length], dtype=torch.bool,
                                                   device="cpu")
                block_size, block_thres = 128, frame_size * 1.5
                num_block = math.ceil(num_frame * frame_size / block_size)
                for i in range(num_block):
                    for j in range(num_block):
                        if abs(i - j) < block_thres // block_size:
                            pixel_attn_mask[i * block_size: (i + 1) * block_size,
                            j * block_size: (j + 1) * block_size] = 1
                attention_mask[:-context_length, :-context_length] = pixel_attn_mask

                attention_mask[-context_length:, :] = 1
                attention_mask[:, -context_length:] = 1

            else:
                pixel_attn_mask = torch.zeros_like(attention_mask[:-context_length, :-context_length], dtype=torch.bool,
                                                   device=device)

                block_size, block_thres = 128, frame_size * 1.5
                num_block = math.ceil(num_frame * frame_size / block_size)
                for i in range(num_block):
                    for j in range(num_block):
                        if abs(i - j) < block_thres // block_size:
                            pixel_attn_mask[i * block_size: (i + 1) * block_size,
                            j * block_size: (j + 1) * block_size] = 1

                pixel_attn_mask = pixel_attn_mask.reshape(frame_size, num_frame, frame_size, num_frame).permute(1, 0, 3,
                                                                                                                2).reshape(
                    frame_size * num_frame, frame_size * num_frame)
                attention_mask[:-context_length, :-context_length] = pixel_attn_mask

                attention_mask[-context_length:, :] = 1
                attention_mask[:, -context_length:] = 1
            attention_mask = attention_mask[:args.sample_mse_max_row].cuda()
            return attention_mask

    hunyuan_video_sampler.pipeline.transformer.__class__.cnt = 0
    hunyuan_video_sampler.pipeline.transformer.__class__.num_steps = args.infer_steps
    hunyuan_video_sampler.pipeline.transformer.__class__.thresh = 0.025
    hunyuan_video_sampler.pipeline.transformer.__class__.forward = easycache_forward
    hunyuan_video_sampler.pipeline.transformer.__class__.ret_steps = 5
    hunyuan_video_sampler.pipeline.transformer.__class__.k = None
    hunyuan_video_sampler.pipeline.transformer.__class__.total_time = 0.0

    if args.pattern == "SVG":
        from svg.models.hyvideo.modules.attenion import Hunyuan_SparseAttn, prepare_flexattention
        from svg.models.hyvideo.modules.custom_models import replace_sparse_forward

        AttnModule = Hunyuan_SparseAttn
        AttnModule.num_sampled_rows = args.num_sampled_rows
        AttnModule.sample_mse_max_row = args.sample_mse_max_row
        AttnModule.attention_masks = [get_attention_mask(mask_name) for mask_name in masks]
        AttnModule.first_layers_fp = args.first_layers_fp
        AttnModule.first_times_fp = args.first_times_fp

    generation_time = []
    time_cost = {
        "GPU_Device": torch.cuda.get_device_name(0),
        "number_prompt": None,
        "avg_cost_time": None
    }
    # Start sampling
    if args.pattern == "SVG":
        # We need to get the prompt len in advance, since HunyuanVideo handle the attention mask in a special way
        prompt_mask = hunyuan_video_sampler.get_prompt_mask(
            prompt=args.prompt,
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            negative_prompt=args.neg_prompt,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            num_videos_per_prompt=args.num_videos,
            embedded_guidance_scale=args.embedded_cfg_scale
        )
        prompt_len = prompt_mask.sum()

        block_mask = prepare_flexattention(
            cfg_size, num_head, head_dim, dtype, device,
            context_length, prompt_len, num_frame, frame_size,
            diag_width=spatial_width, multiplier=temporal_width
        )
        AttnModule.block_mask = block_mask
        replace_sparse_forward()

    hunyuan_video_sampler.pipeline.transformer.total_time = 0.0
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt,
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=1,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale
    )
    generation_time.append(hunyuan_video_sampler.pipeline.transformer.total_time)
    samples = outputs['samples']

    # Save samples
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            save_path = f"{args.save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/', '')}.mp4"
            save_videos_grid(sample, save_path, fps=24)
            logger.info(f'Sample save to: {save_path}')

    if generation_time:
        time_cost["number_prompt"] = len(generation_time)
        time_cost["avg_cost_time"] = sum(generation_time) / len(generation_time) if generation_time else 0

        print(
            f"GPU_Device: {time_cost['GPU_Device']}, number_prompt: {time_cost['number_prompt']}, avg_cost_time: {time_cost['avg_cost_time']}")

        try:
            with open(f"{args.save_path}/time_cost.json", "a+") as f:
                portalocker.lock(f, portalocker.LOCK_EX)
                f.seek(0)
                try:
                    existing_data = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_data = []

                existing_data.append(time_cost)
                f.seek(0)
                f.truncate()
                json.dump(existing_data, f, indent=4)
        except Exception as e:
            print(f"Error writing time cost to file: {e}")


if __name__ == "__main__":
    main()