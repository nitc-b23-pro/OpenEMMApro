"""
main.py
Inference + ADE evaluation entrypoint for OpenEMMA on a Qwen2.5-VL-3B-
Instruct backbone (build_model.py): loads a trained checkpoint, runs it
over nuScenes scenes, reconstructs predicted trajectories with a
kinematic bicycle-model integration, and reports per-scene ADE@1s/2s/3s.
"""
import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import time
from datetime import datetime
from math import atan2

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from nuscenes import NuScenes
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from openemma.YOLO3D.inference import yolo3d_nuScenes
from utils import EstimateCurvatureFromTrajectory, IntegrateCurvatureForPoints, OverlayTrajectory, WriteImageSequenceToVideo
from build_model import build_openemma
from prompts import build_prompt

OBS_LEN = 10
FUT_LEN = 10
TTL_LEN = OBS_LEN + FUT_LEN
DT = 0.5

# EDIT ME (or set the OPENEMMA_QWEN_PATH env var): local directory holding
# Qwen2.5-VL-3B-Instruct's weights/config/processor files -- must be a real
# path on disk, not a HF Hub repo id. If your Kaggle input's
# preprocessor_config.json isn't recognized by the installed transformers
# version, fix it in a working copy (e.g. symlink the input dir into
# /kaggle/working/ and patch image_processor_type/processor_class there)
# and point this at that copy -- not the raw input. Keep in sync with
# train_openemma.py's PRETRAINED.
BASE_PRETRAINED = os.environ.get("OPENEMMA_QWEN_PATH", "/kaggle/working/qwen2.5-vl-3b-fixed")

# Caps the number of visual tokens Qwen2.5-VL turns each frame into
# (pixels / 28^2, since each visual token covers a 28x28 patch after the
# 2x2 patch merge). The default range (256..1280 tokens, i.e.
# min_pixels=256*28*28..max_pixels=1280*28*28) is tuned for general
# scene understanding; a single front-camera frame doesn't need that
# much detail for trajectory prediction, and every extra visual token
# lengthens the sequence the LoRA'd decoder has to backprop through.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28


def load_model_and_processor(base_pretrained=BASE_PRETRAINED, trained_checkpoint_path=None):
    model = build_openemma(base_pretrained, trained_checkpoint_path=trained_checkpoint_path).cuda().eval()
    processor = AutoProcessor.from_pretrained(base_pretrained, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    return model, processor


def predict_step(image_path, obs_velocities, obs_curvatures, model, processor):
    obs_norm = np.linalg.norm(obs_velocities, axis=1)
    obs_curv = obs_curvatures * 100
    obs = np.stack([obs_norm, obs_curv], axis=1)
    state = torch.from_numpy(obs.flatten().astype(np.float32)).unsqueeze(0).cuda()

    raw_lang = build_prompt(obs)

    img = Image.open(image_path).convert("RGB")
    message = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": raw_lang},
    ]}]
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(message)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)

    with torch.inference_mode():
        pred = model(**inputs, states=state, eval=True)

    return pred[0].cpu().numpy().tolist(), raw_lang


def run_inference_on_scenes(nusc, scenes, model, processor, output_dir, args):
    results = []
    frame_count, frame_time_total = 0, 0.0

    for scene in scenes:
        name = scene["name"]
        token = scene["token"]
        first_sample_token, last_sample_token = scene["first_sample_token"], scene["last_sample_token"]

        front_camera_images, ego_poses, camera_params = [], [], []
        tok = first_sample_token
        while True:
            sample = nusc.get("sample", tok)
            cam = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            front_camera_images.append(os.path.join(nusc.dataroot, cam["filename"]))
            ego_poses.append(nusc.get("ego_pose", cam["ego_pose_token"]))
            camera_params.append(nusc.get("calibrated_sensor", cam["calibrated_sensor_token"]))
            if tok == last_sample_token:
                break
            tok = sample["next"]

        scene_length = len(front_camera_images)
        print(f"Scene {name} has {scene_length} frames")
        if scene_length < TTL_LEN:
            print(f"Scene {name} has less than {TTL_LEN} frames, skipping...")
            continue

        ego_world = np.array([p["translation"][:3] for p in ego_poses])
        ego_velocities = np.zeros_like(ego_world)
        ego_velocities[1:] = (ego_world[1:] - ego_world[:-1]) / DT
        ego_velocities[0] = ego_velocities[1]
        ego_curvatures = EstimateCurvatureFromTrajectory(ego_world)
        ego_speed = np.linalg.norm(ego_velocities, axis=1)

        if args.plot:
            estimated_points = IntegrateCurvatureForPoints(
                ego_curvatures[1:], ego_speed[1:], ego_world[0],
                atan2(ego_velocities[0][1], ego_velocities[0][0]), DT,
            )
            plt.plot(ego_world[:, 0], ego_world[:, 1], "r-", label="GT")
            plt.quiver(ego_world[:, 0], ego_world[:, 1], ego_velocities[:, 0], ego_velocities[:, 1], color="b")
            plt.plot(estimated_points[:, 0], estimated_points[:, 1], "g-", label="Reconstruction")
            plt.legend()
            plt.savefig(f"{output_dir}/{name}_interpolation.jpg")
            plt.close()

        cam_images_sequence = []
        ade1s_list, ade2s_list, ade3s_list = [], [], []

        for i in range(scene_length - TTL_LEN):
            frame_start = time.time()

            fut_world = ego_world[i + OBS_LEN:i + TTL_LEN, :2]
            obs_velocities = ego_velocities[i:i + OBS_LEN]
            obs_curvatures = ego_curvatures[i:i + OBS_LEN]
            current_position = ego_world[i + OBS_LEN - 1]
            current_pose = ego_poses[i + OBS_LEN - 1]
            current_camera_params = camera_params[i + OBS_LEN - 1]
            current_image = front_camera_images[i + OBS_LEN - 1]

            img = cv2.imread(current_image)
            img = yolo3d_nuScenes(img, calib=current_camera_params)[0]

            speed_curvature_pred, raw_lang = predict_step(current_image, obs_velocities, obs_curvatures, model, processor)
            speed_curvature_pred = speed_curvature_pred[:FUT_LEN]
            print(f"Got {len(speed_curvature_pred)} future actions: {speed_curvature_pred}")

            pred_len = min(FUT_LEN, len(speed_curvature_pred))
            pred_speeds = np.array(speed_curvature_pred)[:, 0]
            pred_curvatures = np.array(speed_curvature_pred)[:, 1] / 100
            pred_traj = np.zeros((pred_len, 3))
            pred_traj[:, :2] = IntegrateCurvatureForPoints(
                pred_curvatures, pred_speeds, current_position,
                atan2(obs_velocities[-1][1], obs_velocities[-1][0]), DT,
            )
            OverlayTrajectory(img, pred_traj.tolist(), current_camera_params, current_pose, color=(255, 0, 0), args=args)

            err = np.linalg.norm(fut_world[:pred_len] - pred_traj[:pred_len, :2], axis=1)
            ade = float(np.mean(err))
            ade1s_list.append(float(np.mean(err[:min(pred_len, 2)])))
            ade2s_list.append(float(np.mean(err[:min(pred_len, 4)])))
            ade3s_list.append(float(np.mean(err[:min(pred_len, 6)])))

            if args.plot:
                cam_images_sequence.append(img.copy())
                cv2.imwrite(f"{output_dir}/{name}_{i}_front_cam.jpg", img)

                plt.plot(fut_world[:, 0], fut_world[:, 1], "r-", label="GT")
                plt.plot(pred_traj[:, 0], pred_traj[:, 1], "b-", label="Pred")
                plt.legend()
                plt.title(f"Scene: {name}, Frame: {i}, ADE: {ade}")
                plt.savefig(f"{output_dir}/{name}_{i}_traj.jpg")
                plt.close()

                with open(f"{output_dir}/{name}_{i}_logs.txt", "w") as f:
                    f.write(f"Prompt: {raw_lang}\n")
                    f.write(f"Predicted speed/curvature (x100): {speed_curvature_pred}\n")
                    f.write(f"Average Displacement Error: {ade}\n")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            frame_count += 1
            frame_time_total += time.time() - frame_start
            print(f"[frame {frame_count}] scene={name} idx={i} "
                  f"running_avg_frame_time={frame_time_total / frame_count:.3f}s")

        mean_ade1s, mean_ade2s, mean_ade3s = float(np.mean(ade1s_list)), float(np.mean(ade2s_list)), float(np.mean(ade3s_list))
        failure_rate = 100.0 * sum(1 for f in ade1s_list if f > 10) / len(ade1s_list)

        result = {
            "name": name, "token": token,
            "ade1s": mean_ade1s, "ade2s": mean_ade2s, "ade3s": mean_ade3s,
            "avgade": float(np.mean([mean_ade1s, mean_ade2s, mean_ade3s])),
            "failure_rate": failure_rate,
        }
        results.append(result)

        with open(f"{output_dir}/ade_results.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")

        if args.plot:
            WriteImageSequenceToVideo(cam_images_sequence, f"{output_dir}/{name}")

    print(f"Inference done on {len(results)} scene(s). Frames processed: {frame_count}, "
          f"average frame time: {frame_time_total / max(frame_count, 1):.3f}s")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="qwen")
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--epoch", type=str, default="/kaggle/working/OpenEMMApro/openemma_epoch0/")
    parser.add_argument("--dataroot", type=str, default="datasets/NuScenes")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--method", type=str, default="openemma")
    args = parser.parse_args()

    model, processor = load_model_and_processor(BASE_PRETRAINED, args.epoch)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"{args.model_path}_results/{args.method}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot)
    print(f"Number of scenes: {len(nusc.scene)}")

    run_inference_on_scenes(nusc, nusc.scene, model, processor, output_dir, args)
