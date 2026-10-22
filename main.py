"""
main_qwen.py
Inference + ADE evaluation entrypoint for the Qwen2.5-VL-3B-Instruct
backend (build_model_qwen.py) -- the Qwen analogue of main.py (which
drives the LLaVA-Pythia backend). Branch: kaggle-openvlm.

Everything about HOW a prediction becomes an ADE number (route-intent
lookup, kinematic bicycle-model trajectory reconstruction, per-scene
ADE@1s/2s/3s + failure-rate aggregation, overlay video writing) is
copied byte-for-byte from main.py's run_inference_on_scenes -- none of
that is backbone-specific. The only things that actually change for
Qwen are how a single prediction step turns (image, prompt) into
model-ready tensors and how the model/processor get loaded -- see
predict_step() and load_model_and_processors() below, and this
repo's own pre-TinyVLA Qwen integration (git history, commit b33a7b7's
vlm_inference()) for the message/processor pattern this reuses.

load_model_and_processors()/BASE_PRETRAINED/TTL_LEN keep the same names
main.py exports so eval_train_vs_test_qwen.py's import line is a
one-line swap of `from main import ...` -> `from main_qwen import ...`
(see that file). The one real signature difference: this backend has a
single `processor` (AutoProcessor) instead of a separate tokenizer +
image_processor, so run_inference_on_scenes()/load_model_and_processors()
here take (model, processor) instead of (model, tokenizer, image_processor).
"""
import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
from datetime import datetime
from math import atan2

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from nuscenes import NuScenes

import json
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from PIL import Image

from openemma.YOLO3D.inference import yolo3d_nuScenes
from utils import EstimateCurvatureFromTrajectory, IntegrateCurvatureForPoints, OverlayTrajectory, WriteImageSequenceToVideo
from build_model import build_openemma_qwen
from prompts import build_prompt
from route_intent import route_intent_for_window

OBS_LEN = 10
FUT_LEN = 10
TTL_LEN = OBS_LEN + FUT_LEN

# EDIT ME: same Kaggle input path as train_openemma_qwen.py's PRETRAINED --
# keep these two in sync (base weights/tokenizer/processor files live here,
# never inside a trained-checkpoint dir, which only holds the LoRA adapter
# + diffusion_head.pt).
BASE_PRETRAINED = "/kaggle/input/qwen2.5-vl-3b-instruct/transformers/default/1/"


def load_model_and_processors(base_pretrained=BASE_PRETRAINED, trained_checkpoint_path=None):
    model = build_openemma_qwen(base_pretrained, trained_checkpoint_path=trained_checkpoint_path).cuda().eval()
    processor = AutoProcessor.from_pretrained(base_pretrained)
    return model, processor


def predict_step(image_path, obs_velocities, obs_curvatures, model, processor, intent=None):
    """
    Qwen2.5-VL analogue of main.py's predict_step: same shared prompt
    (prompts.build_prompt), same diffusion-head output contract (a ready-
    made (1, 10, 2) tensor via eval=True, no text/regex involved) -- only
    the prompt -> model-input step differs, using Qwen's own chat-template
    + qwen_vl_utils.process_vision_info pattern instead of LLaVA's
    tokenizer_image_token + CLIPImageProcessor.
    """
    obs_norm = np.linalg.norm(obs_velocities, axis=1)
    obs_curv = obs_curvatures * 100
    obs = np.stack([obs_norm, obs_curv], axis=1)
    state = torch.from_numpy(obs.flatten().astype(np.float32)).unsqueeze(0).cuda()

    raw_lang = build_prompt(obs, intent=intent)   # identical prompt-building fn used by the LLaVA-Pythia backend

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

    speed_curvature_pred = pred[0].cpu().numpy().tolist()
    return speed_curvature_pred, raw_lang


def run_inference_on_scenes(nusc, scenes, model, processor, timestamp, args):
    """
    Byte-for-byte main.py's run_inference_on_scenes, with its predict_step
    call swapped for this file's Qwen version and (model, tokenizer,
    image_processor) narrowed to (model, processor) -- see this file's
    module docstring for why nothing else needed to change.
    """
    results = []
    local_frame_count = 0
    local_frame_time = 0.0

    for scene in scenes:
        token = scene['token']
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        name = scene['name']
        front_camera_images = []
        ego_poses = []
        camera_params = []
        curr_sample_token = first_sample_token
        while True:
            sample = nusc.get('sample', curr_sample_token)
            cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            front_camera_images.append(os.path.join(nusc.dataroot, cam_front_data['filename']))
            pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
            ego_poses.append(pose)
            camera_params.append(nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token']))
            if curr_sample_token == last_sample_token:
                break
            curr_sample_token = sample['next']

        scene_length = len(front_camera_images)
        print(f"Scene {name} has {scene_length} frames")

        if scene_length < TTL_LEN:
            print(f"Scene {name} has less than {TTL_LEN} frames, skipping...")
            continue

        DT = 0.5
        ego_poses_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]
        ego_poses_world = np.array(ego_poses_world)
        plt.plot(ego_poses_world[:, 0], ego_poses_world[:, 1], 'r-', label='GT')

        ego_velocities = np.zeros_like(ego_poses_world)
        ego_velocities[1:] = (ego_poses_world[1:] - ego_poses_world[:-1]) / DT
        ego_velocities[0] = ego_velocities[1]

        ego_curvatures = EstimateCurvatureFromTrajectory(ego_poses_world)
        ego_velocities_norm = np.linalg.norm(ego_velocities, axis=1)
        estimated_points = IntegrateCurvatureForPoints(
            ego_curvatures[1:],
            ego_velocities_norm[1:],
            ego_poses_world[0],
            atan2(ego_velocities[0][1], ego_velocities[0][0]),
            DT,
        )

        if args.plot:
            plt.quiver(ego_poses_world[:, 0], ego_poses_world[:, 1], ego_velocities[:, 0], ego_velocities[:, 1],
                       color='b')
            plt.plot(estimated_points[:, 0], estimated_points[:, 1], 'g-', label='Reconstruction')
            plt.legend()
            plt.savefig(f"{timestamp}/{name}_interpolation.jpg")
            plt.close()

        ego_traj_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]

        cam_images_sequence = []
        ade1s_list = []
        ade2s_list = []
        ade3s_list = []
        for i in range(scene_length - TTL_LEN):
            frame_start = time.time()

            fut_ego_traj_world = ego_traj_world[i+OBS_LEN:i+TTL_LEN]
            obs_ego_velocities = ego_velocities[i:i+OBS_LEN]
            obs_ego_curvatures = ego_curvatures[i:i+OBS_LEN]

            current_ego_position = ego_traj_world[i + OBS_LEN - 1]
            current_ego_pose = ego_poses[i + OBS_LEN - 1]
            current_camera_params = camera_params[i + OBS_LEN - 1]
            current_image = front_camera_images[i + OBS_LEN - 1]
            img = cv2.imread(current_image)
            img = yolo3d_nuScenes(img, calib=current_camera_params)[0]

            intent = None
            if getattr(args, "use_intent", False):
                intent = route_intent_for_window(
                    ego_curvatures, ego_velocities_norm,
                    start_idx=i + getattr(args, "lookahead_offset", TTL_LEN),
                )

            speed_curvature_pred, raw_lang = predict_step(
                current_image, obs_ego_velocities, obs_ego_curvatures, model, processor,
                intent=intent)
            speed_curvature_pred = speed_curvature_pred[:10]
            print(f"Got {len(speed_curvature_pred)} future actions: {speed_curvature_pred}")

            pred_len = min(FUT_LEN, len(speed_curvature_pred))
            pred_curvatures = np.array(speed_curvature_pred)[:, 1] / 100
            pred_speeds = np.array(speed_curvature_pred)[:, 0]
            pred_traj = np.zeros((pred_len, 3))
            pred_traj[:, :2] = IntegrateCurvatureForPoints(
                pred_curvatures,
                pred_speeds,
                current_ego_position,
                atan2(obs_ego_velocities[-1][1], obs_ego_velocities[-1][0]),
                DT,
            )
            OverlayTrajectory(img, pred_traj.tolist(), current_camera_params, current_ego_pose, color=(255, 0, 0), args=args)

            fut_ego_traj_world = np.array(fut_ego_traj_world)
            ade = np.mean(np.linalg.norm(fut_ego_traj_world[:pred_len, :2] - pred_traj[:pred_len, :2], axis=1))

            pred1_len = min(pred_len, 2)
            ade1s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred1_len, :2] - pred_traj[:pred1_len, :2], axis=1))
            ade1s_list.append(ade1s)

            pred2_len = min(pred_len, 4)
            ade2s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred2_len, :2] - pred_traj[:pred2_len, :2], axis=1))
            ade2s_list.append(ade2s)

            pred3_len = min(pred_len, 6)
            ade3s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred3_len, :2] - pred_traj[:pred3_len, :2], axis=1))
            ade3s_list.append(ade3s)

            if args.plot == True:
                cam_images_sequence.append(img.copy())
                cv2.imwrite(f"{timestamp}/{name}_{i}_front_cam.jpg", img)

                plt.plot(fut_ego_traj_world[:, 0], fut_ego_traj_world[:, 1], 'r-', label='GT')
                plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'b-', label='Pred')
                plt.legend()
                plt.title(f"Scene: {name}, Frame: {i}, ADE: {ade}")
                plt.savefig(f"{timestamp}/{name}_{i}_traj.jpg")
                plt.close()

                np.save(f"{timestamp}/{name}_{i}_pred_traj.npy", pred_traj)
                np.save(f"{timestamp}/{name}_{i}_pred_curvatures.npy", pred_curvatures)
                np.save(f"{timestamp}/{name}_{i}_pred_speeds.npy", pred_speeds)

                with open(f"{timestamp}/{name}_{i}_logs.txt", 'w') as f:
                    f.write(f"Prompt: {raw_lang}\n")
                    f.write(f"Predicted speed/curvature (x10): {speed_curvature_pred}\n")
                    f.write(f"Average Displacement Error: {ade}\n")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            frame_time = time.time() - frame_start
            local_frame_count += 1
            local_frame_time += frame_time
            running_avg_frame_time = local_frame_time / local_frame_count
            print(f"[frame {local_frame_count}] scene={name} idx={i} "
                  f"frame_time={frame_time:.3f}s running_avg_frame_time={running_avg_frame_time:.3f}s")

        mean_ade1s = np.mean(ade1s_list)
        mean_ade2s = np.mean(ade2s_list)
        mean_ade3s = np.mean(ade3s_list)
        failure_rate = 0
        for f in ade1s_list:
            if f > 10:
                failure_rate += 1
        failure_rate = (failure_rate * 100) / len(ade1s_list)

        aveg_ade = np.mean([mean_ade1s, mean_ade2s, mean_ade3s])

        result = {
            "name": name,
            "token": token,
            "ade1s": mean_ade1s,
            "ade2s": mean_ade2s,
            "ade3s": mean_ade3s,
            "avgade": aveg_ade,
            "failure_rate": failure_rate
        }
        results.append(result)

        with open(f"{timestamp}/ade_results.jsonl", "a") as f:
            f.write(json.dumps(result))
            f.write("\n")

        if args.plot:
            WriteImageSequenceToVideo(cam_images_sequence, f"{timestamp}/{name}")

    print(f"Inference done on {len(results)} scene(s). Frames processed: {local_frame_count}, "
          f"average frame-processing time: {local_frame_time / max(local_frame_count, 1):.3f}s")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="qwen")
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--epoch", type=str, default="/kaggle/working/OpenEMMApro/openemma_qwen_epoch0/")
    parser.add_argument("--dataroot", type=str, default='datasets/NuScenes')
    parser.add_argument("--version", type=str, default='v1.0-mini')
    parser.add_argument("--method", type=str, default='openemma')
    parser.add_argument("--use-intent", dest="use_intent", action="store_true", default=True,
                         help="Must match the checkpoint's own training setting (default: on).")
    parser.add_argument("--no-intent", dest="use_intent", action="store_false")
    parser.add_argument("--lookahead-offset", type=int, default=TTL_LEN)
    args = parser.parse_args()

    print(f"{args.model_path}")

    TRAINED_CHECKPOINT = args.epoch

    model, processor = load_model_and_processors(BASE_PRETRAINED, TRAINED_CHECKPOINT)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    timestamp = args.model_path + f"_results/{args.method}/" + timestamp
    os.makedirs(timestamp, exist_ok=True)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot)

    scenes = nusc.scene
    print(f"Number of scenes: {len(scenes)}")

    run_inference_on_scenes(nusc, scenes, model, processor, timestamp, args)
