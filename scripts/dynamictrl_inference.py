# Copyright 2025 The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import argparse
from typing import Literal

import torch
from diffusers import CogVideoXDPMScheduler
from diffusers.utils import export_to_video, load_image, load_video

from dynamictrl.pipelines.dynamictrl_pipeline import DynamiCtrlAnimationPipeline

from utils.load_validation_control import load_control_video_inference, load_contorl_video_from_Image
from utils.pre_process import preprocess

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def generate_video(
    prompt: str,
    model_path: str,
    num_frames: int = 81,
    width: int = 1360,
    height: int = 768,
    output_path: str = "./output.mp4",
    reference_image_path: str = "",
    ori_driving_video: str = None,
    pose_video: str = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    abandon_prefix: int = None,
    seed: int = 42,
    fps: int = 8,
    pose_control_function: str = "padaln",
    re_init_noise_latent: bool = False,
):
    """
    DynamiCtrl: Generates a dynamic video based on the given human image, guided video, and prompt, and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - num_frames (int): Number of frames to generate.
    - width (int): The width of the generated video
    - height (int): The height of the generated video
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    """

    image_name = os.path.basename(reference_image_path).split(".")[0]

    # 1. Initial the DynamiCtrl inference Pipeline.
    pipe = DynamiCtrlAnimationPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    pipe.to("cuda")
    # pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # 2. Load the image.
    image = load_image(image=reference_image_path)

    # 3. Load the driving video or the pose video.
    if ori_driving_video:
        video_name = os.path.basename(ori_driving_video).split(".")[0]
        pose_images, ref_image = preprocess(ori_driving_video, reference_image_path, 
                                                          width=width, height=height, max_frame_num=num_frames - 1, sample_stride=1)
        pose_sequence_save_dir = os.path.join(output_path, f"PoseImage_{image_name}_pose_{video_name}")
        os.makedirs(pose_sequence_save_dir, exist_ok=True)
        control_poses = []
        for i in range(len(pose_images)):
            control_poses.append(pose_images[i])
            pose_images[i].save(os.path.join(pose_sequence_save_dir, f"{i}.png")) # Save aligned pose images.
        validation_control_video = load_contorl_video_from_Image(control_poses, height, width).unsqueeze(0)
    
    if pose_video:
        video_name = os.path.basename(pose_video).split(".")[0]
        validation_control_video = load_control_video_inference(pose_video, video_height=height, video_width=width, max_frames=num_frames).unsqueeze(0)

    # 4. Generate the video frames.
    video_generate = pipe(
        height=height,
        width=width,
        prompt=prompt,
        image=image,
        control_video=validation_control_video,
        pose_control_function=pose_control_function,
        num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
        num_inference_steps=num_inference_steps,  # Number of inference steps
        num_frames=num_frames,  # Number of frames to generate
        use_dynamic_cfg=False,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        re_init_noise_latent=re_init_noise_latent,
    ).frames[0]
    
    # 5. Save the videos.
    for i in range(int(len(video_generate) / num_frames)):
        name_timestamp = int(time.time())
        video_gen = video_generate[i * num_frames : (i + 1) * num_frames]
        if abandon_prefix:
            video_gen = video_gen[abandon_prefix:]
        video_output_path = os.path.join(output_path, f"DynamiCtrl_{name_timestamp}_image_{image_name}_pose_{video_name}_seed_{seed}_cfg_{guidance_scale}.mp4")
        export_to_video(video_gen, video_output_path, fps=fps)
        print(f"output_path: {video_output_path}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human image animation from an human image, a driving video, and prompt using DynamiCtrl")
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument(
        "--reference_image_path",
        type=str,
        default=None,
        help="The path of the reference human image to be used as the subject of the video",
    )
    parser.add_argument(
        "--ori_driving_video",
        type=str,
        default=None,
        help="the origin real video pth, contain the actioned human"
    )
    parser.add_argument(
        "--pose_video",
        type=str,
        default=None,
        help="the path of pose video"
    )
    parser.add_argument(
        "--model_path", type=str, default="gulucaptain/DynamiCtrl", help="Path of the pre-trained model use"
    )
    parser.add_argument("--output_path", type=str, default="./output.mp4", help="The path save generated video")
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of steps for the inference process")
    parser.add_argument("--width", type=int, default=1360, help="Number of steps for the inference process")
    parser.add_argument("--height", type=int, default=768, help="Number of steps for the inference process")
    parser.add_argument("--fps", type=int, default=16, help="Number of steps for the inference process")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--abandon_prefix", type=int, default=None, help="Save video frames range [abandon_prefix:]")
    parser.add_argument("--re_init_noise_latent", action="store_true", help="whether to resample the initial noise")
    parser.add_argument("--pose_control_function", type=str, default="padaln")
    
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    os.makedirs(args.output_path, exist_ok=True)
    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        output_path=args.output_path,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        reference_image_path=args.reference_image_path,
        ori_driving_video=args.ori_driving_video,
        pose_video=args.pose_video,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        abandon_prefix=args.abandon_prefix,
        seed=args.seed,
        fps=args.fps,
        re_init_noise_latent=args.re_init_noise_latent,
        pose_control_function=args.pose_control_function,
    )
