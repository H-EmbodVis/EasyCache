import os
import cv2
import argparse
import torch
import lpips
import numpy as np
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure

def load_video_frames(path, resize_to=None):
    """
    Load all frames from a video file as a list of HxWx3 uint8 arrays.
    Optionally resize each frame to `resize_to` (w, h).
    """

    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        if resize_to is not None:
            img = cv2.resize(img, resize_to)
        frames.append(np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), axis=0))
    cap.release()
    return np.concatenate(frames)


def compute_video_metrics(frames_gt, frames_gen,
                          device, ssim_metric, lpips_fn):
    """
    Compute PSNR, SSIM, LPIPS for two lists of frames (uint8 BGR).
    All computations on `device`.
    Returns (psnr, ssim, lpips) scalars.
    """
    # ensure same frame count
    # convert to tensors [N,3,H,W], normalize to [0,1]
    gt_t = torch.from_numpy(frames_gt).float().to(device).permute(0, 3, 1, 2).div_(255).contiguous()

    gen_t = torch.from_numpy(frames_gen).float().to(device).permute(0, 3, 1, 2).div_(255).contiguous()

    # PSNR (data_range=1.0): -10 * log10(mse)
    mse = torch.mean((gt_t - gen_t) ** 2)
    psnr = -10.0 * torch.log10(mse)

    # SSIM: returns average over batch
    ssim_val = ssim_metric(gen_t, gt_t)

    # LPIPS: expects [-1,1]
    with torch.no_grad():
        lpips_val = lpips_fn(gt_t * 2.0 - 1.0, gen_t * 2.0 - 1.0).mean()

    return psnr.item(), ssim_val.item(), lpips_val.item()


def main():
    parser = argparse.ArgumentParser(
        description="Compute PSNR/SSIM/LPIPS on GPU for two folders of .mp4 videos"
    )
    parser.add_argument("--original_video", required=True,
                        help="ground-truth .mp4 videos")
    parser.add_argument("--generated_video", required=True,
                        help="generated .mp4 videos")
    parser.add_argument("--device", default="cuda",
                        help="Torch device, e.g. 'cuda' or 'cpu'")
    parser.add_argument("--lpips_net", default="alex", choices=["alex", "vgg"],
                        help="Backbone for LPIPS")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    # instantiate metrics on device
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = lpips.LPIPS(net=args.lpips_net, spatial=True).to(device)

    # gather .mp4 filenames
    gt_files = args.original_video
    gen_set = args.generated_video

    psnrs, ssims, lpips_vals = [], [], []
    for fname in tqdm([gt_files], desc="Videos"):
        path_gt = gt_files
        path_gen = gen_set

        # load frames; resize generated to match GT dimensions
        frames_gt = load_video_frames(path_gt)
        frames_gen = load_video_frames(path_gen)

        res = compute_video_metrics(frames_gt, frames_gen,
                                    device, ssim_metric, lpips_fn)
        if res is None:
            continue
        p, s, l = res
        psnrs.append(p);
        ssims.append(s);
        lpips_vals.append(l)

    if not psnrs:
        print("No valid videos processed.")
        return

    print("\n=== Overall Averages ===")
    print(f"Average PSNR : {np.mean(psnrs):.2f} dB")
    print(f"Average SSIM : {np.mean(ssims):.4f}")
    print(f"Average LPIPS: {np.mean(lpips_vals):.4f}")


if __name__ == "__main__":
    main()
