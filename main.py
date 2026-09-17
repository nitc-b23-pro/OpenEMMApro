import argparse
import os

# OOM mitigation (see build_model.py's docstring for the fp16/sdpa half of this
# fix): set BEFORE torch is imported. This is literally what the "CUDA out of
# memory" error you hit earlier suggested trying -- it reduces allocator
# fragmentation that can trigger an OOM even when total usage looks like it
# should fit.
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
from openemma.YOLO3D.inference import yolo3d_nuScenes
from utils import EstimateCurvatureFromTrajectory, IntegrateCurvatureForPoints, OverlayTrajectory, WriteImageSequenceToVideo
from build_model import build_openemma_tinyvla
from llava_pythia.mm_utils import tokenizer_image_token
from llava_pythia.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from transformers import AutoTokenizer, CLIPImageProcessor
from PIL import Image
from prompts import build_prompt

OBS_LEN = 10
FUT_LEN = 10
TTL_LEN = OBS_LEN + FUT_LEN


def predict_step(image_path, obs_velocities, obs_curvatures,
                  model, tokenizer, image_processor):
    """
    One OpenEMMA-TinyVLA inference step: image + the SAME shared prompt
    used at training time (prompts.build_prompt, via openemma_dataset.py)
    -> diffusion head -> 10 future [speed, curvature] pairs.

    FIXED: this used to reference an undefined `prev_intent` variable
    (a NameError on every call, since no scene/object/intent text step
    exists any more) -- it now builds the shared prompt directly from
    this call's own observed history, exactly like training does.
    """
    obs_norm = np.linalg.norm(obs_velocities, axis=1)
    obs_curv = obs_curvatures * 100
    obs = np.stack([obs_norm, obs_curv], axis=1)                 # (10, 2) -- same layout as training
    state = torch.from_numpy(obs.flatten().astype(np.float32)).unsqueeze(0).cuda()   # (1, 20)

    raw_lang = build_prompt(obs)   # identical prompt-building fn used by openemma_dataset.py at train time
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + raw_lang
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                       return_tensors="pt").unsqueeze(0).cuda()

    img = Image.open(image_path).convert("RGB")
    # FIXED (dtype mismatch, same root cause as train_openemma_tinyvla.py's
    # preprocess_batch(): build_model.py loads the base VLM -- including
    # mm_projector, plain nn.Linear with no LoRA/PEFT dtype auto-casting -- in
    # fp16. CLIPVisionTower casts its output back to match whatever dtype the
    # input image tensor was, so it must already be fp16 here.
    image_tensor = image_processor.preprocess(img, return_tensors="pt")["pixel_values"].cuda().half()

    with torch.inference_mode():
        # eval=True triggers the diffusion "sculptor" loop instead of text generation --
        # it returns a ready-made (1, 10, 2) tensor, not a sentence.
        pred = model(input_ids=input_ids, images=image_tensor, states=state, eval=True)

    speed_curvature_pred = pred[0].cpu().numpy().tolist()   # [[speed, curv], ... x10] -- done, no regex
    return speed_curvature_pred, raw_lang


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="lp")
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--dataroot", type=str, default='datasets/NuScenes')
    parser.add_argument("--version", type=str, default='v1.0-mini')   # <-- inference split (per project requirement)
    parser.add_argument("--method", type=str, default='openemma')
    args = parser.parse_args()

    print(f"{args.model_path}")

    # BASE_PRETRAINED: the SAME base LLaVA-Pythia checkpoint path used in
    # train_openemma_tinyvla.py's PRETRAINED constant. Tokenizer/image
    # processor files live here, not in the trained-checkpoint dir below
    # (train_openemma_tinyvla.py never copies them there).
    BASE_PRETRAINED = "/kaggle/input/models/latheeshpoondla/llava-pythia/transformers/h/1/"
    # TRAINED_CHECKPOINT: one of the openemma_tinyvla_epochN directories
    # written by train_openemma_tinyvla.py's model.save_pretrained(...).
    TRAINED_CHECKPOINT = "/kaggle/working/OpenEMMApro/openemma_tinyvla_epoch0/"

    model = build_openemma_tinyvla(BASE_PRETRAINED, trained_checkpoint_path=TRAINED_CHECKPOINT).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_PRETRAINED)
    image_processor = CLIPImageProcessor.from_pretrained(BASE_PRETRAINED)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    timestamp = args.model_path + f"_results/{args.method}/" + timestamp
    os.makedirs(timestamp, exist_ok=True)

    # This block extracts and prepares a time-ordered sequence of front-camera images,
    # vehicle poses, and camera parameters from nuScenes so the driving scene can be
    # reasoned about and future motion can be predicted.
    nusc = NuScenes(version=args.version, dataroot=args.dataroot)

    scenes = nusc.scene
    print(f"Number of scenes: {len(scenes)}")

    # --- global frame counter + running average frame-processing time (across ALL scenes) ---
    global_frame_count = 0
    total_frame_time = 0.0

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

        DT = 0.5  # nuScenes keyframes are 2 Hz
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

            speed_curvature_pred, raw_lang = predict_step(
                current_image, obs_ego_velocities, obs_ego_curvatures, model, tokenizer, image_processor)
            speed_curvature_pred = speed_curvature_pred[:10]
            print(f"Got {len(speed_curvature_pred)} future actions: {speed_curvature_pred}")

            # Pred
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
            # Overlay the trajectory.
            OverlayTrajectory(img, pred_traj.tolist(), current_camera_params, current_ego_pose, color=(255, 0, 0), args=args)

            # Compute ADE.
            fut_ego_traj_world = np.array(fut_ego_traj_world)
            ade = np.mean(np.linalg.norm(fut_ego_traj_world[:pred_len] - pred_traj, axis=1))

            pred1_len = min(pred_len, 2)
            ade1s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred1_len] - pred_traj[:pred1_len], axis=1))
            ade1s_list.append(ade1s)

            pred2_len = min(pred_len, 4)
            ade2s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred2_len] - pred_traj[:pred2_len], axis=1))
            ade2s_list.append(ade2s)

            pred3_len = min(pred_len, 6)
            ade3s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred3_len] - pred_traj[:pred3_len], axis=1))
            ade3s_list.append(ade3s)

            # Write to image.
            if args.plot == True:
                cam_images_sequence.append(img.copy())
                cv2.imwrite(f"{timestamp}/{name}_{i}_front_cam.jpg", img)

                # Plot the trajectory.
                plt.plot(fut_ego_traj_world[:, 0], fut_ego_traj_world[:, 1], 'r-', label='GT')
                plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'b-', label='Pred')
                plt.legend()
                plt.title(f"Scene: {name}, Frame: {i}, ADE: {ade}")
                plt.savefig(f"{timestamp}/{name}_{i}_traj.jpg")
                plt.close()

                # Save the trajectory
                np.save(f"{timestamp}/{name}_{i}_pred_traj.npy", pred_traj)
                np.save(f"{timestamp}/{name}_{i}_pred_curvatures.npy", pred_curvatures)
                np.save(f"{timestamp}/{name}_{i}_pred_speeds.npy", pred_speeds)

                # FIXED: this used to reference undefined scene_description /
                # object_description / updated_intent (a guaranteed NameError,
                # since no separate scene/object/intent text step exists any
                # more). Logging the actual prompt used + prediction instead.
                with open(f"{timestamp}/{name}_{i}_logs.txt", 'w') as f:
                    f.write(f"Prompt: {raw_lang}\n")
                    f.write(f"Predicted speed/curvature (x10): {speed_curvature_pred}\n")
                    f.write(f"Average Displacement Error: {ade}\n")

            # --- global frame counter + per-frame time + running average ---
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            frame_time = time.time() - frame_start
            global_frame_count += 1
            total_frame_time += frame_time
            running_avg_frame_time = total_frame_time / global_frame_count
            print(f"[frame {global_frame_count}] scene={name} idx={i} "
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

        with open(f"{timestamp}/ade_results.jsonl", "a") as f:
            f.write(json.dumps(result))
            f.write("\n")

        if args.plot:
            WriteImageSequenceToVideo(cam_images_sequence, f"{timestamp}/{name}")

    print(f"Inference done. Total frames processed: {global_frame_count}, "
          f"final running average frame-processing time: {total_frame_time / max(global_frame_count, 1):.3f}s")
