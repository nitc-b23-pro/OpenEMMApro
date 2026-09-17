import argparse
import os
import re
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

from PIL import Image

from build_model import build_llava_pythia
from llava_pythia_inference import llava_pythia_generate
from transformers import AutoTokenizer, CLIPImageProcessor

OBS_LEN = 10
FUT_LEN = 10
TTL_LEN = OBS_LEN + FUT_LEN

def get_message(prompt, image):
    return [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]


def vlm_inference(text, image_path, processor, model, tokenizer, args):
    output_text = llava_pythia_generate(text, image_path, model, tokenizer, processor)
    return output_text
        
def SceneDescription(image_path, processor=None, model=None, tokenizer=None, args=None):
    prompt = f"""You are a autonomous driving labeller. You have access to these front-view camera images of a car taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Describe the driving scene according to weather, traffic lights, movements of other cars or pedestrians and lane markings."""

    result = vlm_inference(text=prompt, image_path=image_path, processor=processor, model=model, tokenizer=tokenizer, args=args)
    return result

def DescribeObjects(image_path, processor=None, model=None, tokenizer=None, args=None):

    prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. What other road users should you pay attention to in the driving scene? List two or three of them, specifying its location within the image of the driving scene and provide a short description of the that road user on what it is doing, and why it is important to you."""

    result = vlm_inference(text=prompt, image_path=image_path, processor=processor, model=model, tokenizer=tokenizer, args=args)

    return result

def DescribeOrUpdateIntent(image_path, prev_intent=None, processor=None, model=None, tokenizer=None, args=None):

    if prev_intent is None:
        prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Based on the lane markings and the movement of other cars and pedestrians, describe the desired intent of the ego car. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the current speed or slow down or speed up?"""

    else:
        prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Half a second ago your intent was to {prev_intent}. Based on the updated lane markings and the updated movement of other cars and pedestrians, do you keep your intent or do you change it? Explain your current intent: """

    result = vlm_inference(text=prompt, image_path=image_path, processor=processor, model=model, tokenizer=tokenizer, args=args)

    return result


def GenerateMotion(image_path, obs_velocities, obs_curvatures, given_intent, processor=None, model=None, tokenizer=None, args=None):
    scene_description, object_description, intent_description = None, None, None

    if args.method == "openemma":
        scene_description = SceneDescription(image_path, processor=processor, model=model, tokenizer=tokenizer, args=args)
        object_description = DescribeObjects(image_path, processor=processor, model=model, tokenizer=tokenizer, args=args)
        intent_description = DescribeOrUpdateIntent(image_path, prev_intent=given_intent, processor=processor, model=model, tokenizer=tokenizer, args=args)
        print(f'Scene Description: {scene_description}')
        print(f'Object Description: {object_description}')
        print(f'Intent Description: {intent_description}')

    obs_velocities_norm = np.linalg.norm(obs_velocities, axis=1)
    obs_curvatures = obs_curvatures * 100
    obs_speed_curvature_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in zip(obs_velocities_norm, obs_curvatures)]
    obs_speed_curvature_str = ", ".join(obs_speed_curvature_str)

    
    print(f'Observed Speed and Curvature: {obs_speed_curvature_str}')

    if args.method == "openemma":
        prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
        The scene is described as follows: {scene_description}. 
        The identified critical objects are {object_description}. 
        The car's intent is {intent_description}. 
        The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
        Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""
    else:
        prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
        The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
        Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""
    for rho in range(3):
        result = vlm_inference(text=prompt, image_path=image_path, processor=processor, model=model, tokenizer=tokenizer, args=args)
        print("Result: ", result)
        if not "unable" in result and not "sorry" in result and "[" in result:
            break
    return result, scene_description, object_description, intent_description

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="qwen")
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--dataroot", type=str, default='datasets/NuScenes')
    parser.add_argument("--version", type=str, default='v1.0-mini')
    parser.add_argument("--method", type=str, default='openemma')
    args = parser.parse_args()

    print(f"{args.model_path}")

    model = None    #Before Loading Clean state, No leftover memory, Safe fallback if a model isn’t loaded
    processor = None
    tokenizer = None
    try:
        model = build_llava_pythia("/kaggle/input/models/latheeshpoondla/llava-pythia/transformers/h/1/").cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/models/latheeshpoondla/llava-pythia/transformers/h/1/")
        processor = CLIPImageProcessor.from_pretrained("/kaggle/input/models/latheeshpoondla/llava-pythia/transformers/h/1/")
    except Exception as e:
        print("Exception:", e)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    timestamp = args.model_path + f"_results/{args.method}/" + timestamp
    os.makedirs(timestamp, exist_ok=True)


    #This block extracts and prepares a time-ordered sequence of front-camera images, vehicle poses, and camera parameters from nuScenes so the driving scene can be reasoned about and future motion can be predicted.
    # Load the dataset
    nusc = NuScenes(version=args.version, dataroot=args.dataroot)

    # Iterate the scenes
    scenes = nusc.scene
    
    print(f"Number of scenes: {len(scenes)}")
    for scene in scenes:
        token = scene['token']
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        name = scene['name']
        # Get all image and pose in this scene
        front_camera_images = []
        ego_poses = []
        camera_params = []
        curr_sample_token = first_sample_token
        while True:
            sample = nusc.get('sample', curr_sample_token)

            # Get the front camera image of the sample.
            cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            # nusc.render_sample_data(cam_front_data['token'])


            front_camera_images.append(os.path.join(nusc.dataroot, cam_front_data['filename']))

            # Get the ego pose of the sample.
            pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
            ego_poses.append(pose)

            # Get the camera parameters of the sample.
            camera_params.append(nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token']))

            # Advance the pointer.
            if curr_sample_token == last_sample_token:
                break
            curr_sample_token = sample['next']

        scene_length = len(front_camera_images)
        print(f"Scene {name} has {scene_length} frames")

        if scene_length < TTL_LEN:
            print(f"Scene {name} has less than {TTL_LEN} frames, skipping...")
            continue

        DT = 0.5  # nuScenes keyframes are 2 Hz
        ## Compute interpolated trajectory.
        # Get the velocities of the ego vehicle.
        ego_poses_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]
        ego_poses_world = np.array(ego_poses_world)
        plt.plot(ego_poses_world[:, 0], ego_poses_world[:, 1], 'r-', label='GT')

        ego_velocities = np.zeros_like(ego_poses_world)
        ego_velocities[1:] = (
            ego_poses_world[1:] - ego_poses_world[:-1]
        ) / DT
        ego_velocities[0] = ego_velocities[1]

        # Get the curvature of the ego vehicle.
        ego_curvatures = EstimateCurvatureFromTrajectory(ego_poses_world)
        ego_velocities_norm = np.linalg.norm(ego_velocities, axis=1)
        estimated_points = IntegrateCurvatureForPoints(
            ego_curvatures[1:],
            ego_velocities_norm[1:],
            ego_poses_world[0],
            atan2(ego_velocities[0][1], ego_velocities[0][0]),
            DT,
        )

        # Debug
        if args.plot:
            plt.quiver(ego_poses_world[:, 0], ego_poses_world[:, 1], ego_velocities[:, 0], ego_velocities[:, 1],
                    color='b')
            plt.plot(estimated_points[:, 0], estimated_points[:, 1], 'g-', label='Reconstruction')
            plt.legend()
            plt.savefig(f"{timestamp}/{name}_interpolation.jpg")
            plt.close()

        # Get the waypoints of the ego vehicle.
        ego_traj_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]

        prev_intent = None
        cam_images_sequence = []
        ade1s_list = []
        ade2s_list = []
        ade3s_list = []
        for i in range(scene_length - TTL_LEN):
            # Get the raw image data.
            # utils.PlotBase64Image(front_camera_images[0])
            fut_ego_traj_world = ego_traj_world[i+OBS_LEN:i+TTL_LEN]
            obs_ego_velocities = ego_velocities[i:i+OBS_LEN]
            obs_ego_curvatures = ego_curvatures[i:i+OBS_LEN]

            current_ego_position = ego_traj_world[i + OBS_LEN - 1]
            current_ego_pose = ego_poses[i + OBS_LEN - 1]
            current_camera_params = camera_params[i + OBS_LEN - 1]
            current_image = front_camera_images[i + OBS_LEN - 1]
            img = cv2.imread(current_image)
            img = yolo3d_nuScenes(img, calib=current_camera_params)[0]

            for _ in range(3):
                (prediction,
                scene_description,
                object_description,
                updated_intent) = GenerateMotion(current_image, obs_ego_velocities,
                                                obs_ego_curvatures, prev_intent, processor=processor, model=model, tokenizer=tokenizer, args=args)

                # Process the output.
                prev_intent = updated_intent  # Stateful intent
                pred_waypoints = prediction.replace("Future speeds and curvatures:", "").strip()
                coordinates = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", pred_waypoints)
                if not coordinates == []:
                    break
            if coordinates == []:
                continue
            speed_curvature_pred = [[float(v), float(k)] for v, k in coordinates]
            speed_curvature_pred = speed_curvature_pred[:10]
            print(f"Got {len(speed_curvature_pred)} future actions: {speed_curvature_pred}")

            # GT
            # OverlayTrajectory(img, fut_ego_traj_world, obs_camera_params[-1], obs_ego_poses[-1], color=(255, 0, 0))

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
            ade2s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred2_len] - pred_traj[:pred2_len] , axis=1))
            ade2s_list.append(ade2s)

            pred3_len = min(pred_len, 6)
            ade3s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred3_len] - pred_traj[:pred3_len] , axis=1))
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

                # Save the descriptions
                with open(f"{timestamp}/{name}_{i}_logs.txt", 'w') as f:
                    f.write(f"Scene Description: {scene_description}\n")
                    f.write(f"Object Description: {object_description}\n")
                    f.write(f"Intent Description: {updated_intent}\n")
                    f.write(f"Average Displacement Error: {ade}\n")

            # break  # Timestep

        mean_ade1s = np.mean(ade1s_list)
        mean_ade2s = np.mean(ade2s_list)
        mean_ade3s = np.mean(ade3s_list)
        failure_rate = 0;
        for f in ade1s_list:
            if f>10:
                failure_rate+=1
        failure_rate = (failure_rate *100)/len(ade1s_list)
                
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

        # break  # Scenes

