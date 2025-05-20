from torchvision import transforms
import torch

import decord

import numpy as np

def identity_transform(x):
    return x

def scale_transform(x):
    return x / 255.0

def load_control_video(video_path, height, width):
    control_video_height = height
    control_video_width = width
    video_transforms = transforms.Compose(
        [
            transforms.Lambda(identity_transform),
            transforms.Lambda(scale_transform),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            transforms.Resize((control_video_height, control_video_width))
        ]
    )
    validation_control_video_reader = decord.VideoReader(uri=video_path)
    frame_indices = [i for i in range(0, 36)]

    validation_control_video = validation_control_video_reader.get_batch(frame_indices).float()
    
    validation_control_video = validation_control_video.permute(0, 3, 1, 2).contiguous()
    validation_control_video = torch.cat((validation_control_video, validation_control_video[-1].unsqueeze(0)), dim=0)
    validation_control_video = torch.stack([video_transforms(frame) for frame in validation_control_video], dim=0)
    return validation_control_video

def load_control_video_inference(video_path, video_height, video_width, max_frames=None):
    video_transforms = transforms.Compose(
        [
            transforms.Lambda(identity_transform),
            transforms.Lambda(scale_transform),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            transforms.Resize((1360, 768))
        ]
    )
    validation_control_video_reader = decord.VideoReader(uri=video_path)
    if max_frames:
        frame_indices = [i for i in range(0, max_frames)]
    else:
        frame_indices = [i for i in range(len(validation_control_video_reader))]
    
    validation_control_video = validation_control_video_reader.get_batch(frame_indices).asnumpy()
    validation_control_video = torch.from_numpy(validation_control_video).float()
    
    validation_control_video = validation_control_video.permute(0, 3, 1, 2).contiguous()
    validation_control_video = torch.stack([video_transforms(frame) for frame in validation_control_video], dim=0)
    return validation_control_video

def load_contorl_video_from_Image(validation_control_images, video_height, video_width):
    video_transforms = transforms.Compose(
        [
            transforms.Lambda(identity_transform),
            transforms.Lambda(scale_transform),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            transforms.Resize((video_height, video_width))
        ]
    )
    validation_control_video_tensor = []

    for image in validation_control_images:
        validation_control_video_tensor.append(torch.from_numpy(np.array(image)).float())

    validation_control_video = torch.stack(validation_control_video_tensor, dim=0)
    validation_control_video = validation_control_video.permute(0, 3, 1, 2).contiguous()
    validation_control_video = torch.stack([video_transforms(frame) for frame in validation_control_video], dim=0)
    return validation_control_video

if __name__=="__main__":
    validation_control_video_path = "/home/user/video.mp4"
    validation_control_video = load_control_video(validation_control_video_path)