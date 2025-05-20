## DynamiCtrl: Rethinking the Basic Structure and the Role of Text for High-quality Human Image Animation

[Haoyu Zhao](https://scholar.google.com/citations?user=pCGM7jwAAAAJ&hl=zh-CN&oi=ao/), [Zhongang Qi](https://scholar.google.com/citations?user=zJvrrusAAAAJ&hl=en/), [Cong Wang](#), [Qingqing Zheng](https://scholar.google.com.hk/citations?user=l0Y7emkAAAAJ&hl=zh-CN&oi=ao/), [Guansong Lu](https://scholar.google.com.hk/citations?user=YIt8thUAAAAJ&hl=zh-CN&oi=ao), [Fei Chen](#), [Hang Xu](https://scholar.google.com.hk/citations?user=J_8TX6sAAAAJ&hl=zh-CN&oi=ao) and [Zuxuan Wu](https://scholar.google.com.hk/citations?user=7t12hVkAAAAJ&hl=zh-CN&oi=ao)

<a href='https://arxiv.org/abs/2503.21246'><img src='https://img.shields.io/badge/ArXiv-DynamiCtrl-red'></a> 
<a href='https://gulucaptain.github.io/DynamiCtrl/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://youtu.be/Mu_pNXM4PcE'><img src='https://img.shields.io/badge/YouTube-Intro-yellow'></a>
[![GitHub](https://img.shields.io/github/stars/gulucaptain/DynamiCtrl?style=social)](https://github.com/gulucaptain/DynamiCtrl)
<!-- [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](#) -->

## 🎏 Introduction
<b>TL; DR: <font color="red">DynamiCtrl</font> is the first framework to propose the "Joint-text" paradigm to the pose-guided human animation task and achieve effective pose control within the diffusion transformer (DiT) architecture.</b>

<details><summary>CLICK for the full introduction</summary>


> With diffusion transformer (DiT) excelling in video generation, its use in specific tasks has drawn increasing attention. However, adapting DiT for pose-guided human image animation faces two core challenges: (a) existing U-Net-based pose control methods may be suboptimal for the DiT backbone; and (b) removing text guidance, as in previous approaches, often leads to semantic loss and model degradation. To address these issues, we propose DynamiCtrl, a novel framework for human animation in video DiT architecture. Specifically, we use a shared VAE encoder for human images and driving poses, unifying them into a common latent space, maintaining pose fidelity, and eliminating the need for an expert pose encoder during video denoising. To integrate pose control into the DiT backbone effectively, we propose a novel Pose-adaptive Layer Norm model. It injects normalized pose features into the denoising process via conditioning on visual tokens, enabling seamless and scalable pose control across DiT blocks. Furthermore, to overcome the shortcomings of text removal, we introduce the "Joint-text" paradigm, which preserves the role of text embeddings to provide global semantic context. Through full-attention blocks, image and pose features are aligned with text features, enhancing semantic consistency, leveraging pretrained knowledge, and enabling multi-level control. Experiments verify the superiority of DynamiCtrl on benchmark and self-collected data (e.g., achieving the best LPIPS of 0.166), demonstrating strong character control and high-quality synthesis.
</details>

## 📺 Overview on YouTube

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/Mu_pNXM4PcE/0.jpg)](https://www.youtube.com/watch?v=Mu_pNXM4PcE)
Please click to watch.

## ⚔️ DynamiCtrl for High-quality Pose-guided Human Image Animation

We first refocus on the role of text for this task and find that fine-grained textual information helps improve video quality. In particular, we can achieve <font color="green">fine-grained local controllability</font> using different prompts.

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/58aed8d1-9dce-416c-9cf4-57af9dbf2d19" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/ec40ac72-6b46-487b-b78a-a28b123e1041" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/1d7d323b-dd12-4023-a709-1774801725cd" width="100%" controls autoplay loop></video>
      </td>
  </tr>
</table>

<details><summary>CLICK to check the prompts used for generation in the above three cases.</summary>

> Prompt (left): “The image depicts a stylized, animated character standing amidst a chaotic and dynamic background. The character is dressed in a blue suit with a red cape, featuring a prominent "S" emblem on the chest. The suit has a belt with pouches and a utility belt. The character has spiky hair and is standing on a pile of debris and rubble, suggesting a scene of destruction or battle. The background is filled with glowing, fiery elements and a sense of motion, adding to the dramatic and intense atmosphere of the scene."

> Prompt (mid): “The person in the image is a woman with long, blonde hair styled in loose waves. She is wearing a form-fitting, sleeveless top with a high neckline and a small cutout at the chest. The top is beige and has a strap across her chest. She is also wearing a black belt with a pouch attached to it. Around her neck, she has a turquoise pendant necklace. The background appears to be a dimly lit, urban environment with a warm, golden glow."

> Prompt (right): “The person in the image is wearing a black, form-fitting one-piece outfit and a pair of VR goggles. They are walking down a busy street with numerous people and colorful neon signs in the background. The street appears to be a bustling urban area, possibly in a city known for its vibrant nightlife and entertainment. The lighting and signage suggest a lively atmosphere, typical of a cityscape at night."
</details>

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/c37ed889-5212-4180-9f63-f6c7064bccef" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/05d24850-52d3-45b8-98f3-d2ec37846cb7" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/6cd60d42-6c9d-4049-95b3-2b226786315f" width="100%" controls autoplay loop></video>
      </td>
  </tr>
</table>

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/549614b4-200e-4561-a88c-e89e1a61ff01" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/38f3ce0c-1602-4ace-b6a1-16834ac44597" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/0dbc1303-9d22-43de-a4f9-7d9c87189898" width="100%" controls autoplay loop></video>
      </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/ce9cc50c-0366-4f96-ba0e-c49e282ac599" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/44e9990e-edae-4857-b557-14e04ce91e75" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/dd6afcc1-65c5-4dcc-80af-587546f601cc" width="100%" controls autoplay loop></video>
      </td>
  </tr>
</table>

### Fine-grained video control

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/0df11a8b-8c08-4ca3-bed6-eb27f260a9af" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/8e0d16f6-1842-429e-9e9f-ffac93d12cbc" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/358b3eb2-bcda-4581-baf1-ba6fc89cb935" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/43cd395d-f4c5-42fb-9d82-d213e04b52de" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/6828c6a8-a51f-45a6-8dfb-f4b0eaba8742" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/a8d89138-2a69-4c51-bdf6-1341014d5542" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/22d5ebc1-18da-4d27-a2e3-348e44173502" width="100%" controls autoplay loop></video>
      </td>
  </tr>
</table>

<details><summary>CLICK to check the prompts used for generation in the above background-control cases.</summary>

> Scene 1: The person in the image is wearing a white, knee-length dress with short sleeves and a square neckline. The dress features lace detailing and a ruffled hem. The person is also wearing clear, open-toed sandals. The background shows a bustling futuristic city at night, with neon lights reflecting off the wet streets and flying cars zooming above.

> Scene 2: The person in the image is wearing a white, knee-length dress with short sleeves and a square neckline. The dress features lace detailing and a ruffled hem. The person is also wearing clear, open-toed sandals. The background shows a vibrant market street in a Middle Eastern bazaar, filled with colorful fabrics, exotic spices, and merchants calling out to customers.

> Scene 3: The person in the image is wearing a white, knee-length dress with short sleeves and a square neckline. The dress features lace detailing and a ruffled hem. The person is also wearing clear, open-toed sandals. The background shows a sunny beach with golden sand, gentle ocean waves rolling onto the shore, and palm trees swaying in the breeze.

> Scene 4: The person in the image is wearing a white, knee-length dress with short sleeves and a square neckline. The dress features lace detailing and a ruffled hem. The person is also wearing clear, open-toed sandals. The background shows a high-tech research lab with sleek metallic walls, glowing holographic screens, and robotic arms assembling futuristic devices.

> Scene 5: The person in the image is wearing a white, knee-length dress with short sleeves and a square neckline. The dress features lace detailing and a ruffled hem. The person is also wearing clear, open-toed sandals. The background shows a mystical ancient temple hidden deep in the jungle, covered in vines, with glowing runes carved into the stone walls.

> Scene 6: The person in the image is wearing a white, knee-length dress with short sleeves and a square neckline. The dress features lace detailing and a ruffled hem. The person is also wearing clear, open-toed sandals. The background shows a serene snowy forest with tall pine trees, soft snowflakes falling gently, and a frozen river winding through the landscape.

> Scene 7: The person in the image is wearing a white, knee-length dress with short sleeves and a square neckline. The dress features lace detailing and a ruffled hem. The person is also wearing clear, open-toed sandals. The background shows an abandoned industrial warehouse with broken windows, scattered debris, and rusted machinery covered in dust.
</details>

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/e82cc65c-5bc9-4fac-ac99-7e52dce4a3eb" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/20733412-702b-4c71-810c-b5c79054e1a0" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/bb25c56d-c009-49ba-905b-dc78f129eb73" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/3cd01d92-de46-4df1-b55b-7da0c2552db8" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/978e46a1-3fdc-4aeb-84cb-0ad42fa2ec01" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/68754207-d4e6-4280-90ac-28caa634537d" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/737976dd-0bc5-4e2d-97a6-e00533e6f0ce" width="100%" controls autoplay loop></video>
      </td>
  </tr>
</table>



## 🚧 Todo

<details><summary>Click for Previous todos </summary>

- [&#10004;] Release the project page and demos.
- [&#10004;] Paper on Arxiv on 27 Mar 2025.
</details>

- [&#10004;] Release inference code.
- [&#10004;] Release models.
- [&#10004;] Release training code.

## 📋 Changelog
- 2025.05.20 Code and models released!
- 2025.03.30 Project page and demos released!
- 2025.03.10 Project Online!


## Installation

For usage (SFT fine-tuning, inference), you can install the dependencies with:

```bash
conda create --name dynamictrl python=3.10

source activate dynamictrl

pip install -r requirements.txt
```


## Model Zoo

We provide three grou of checkpoints:
<ol>
<li>DynamiCtrl-5B: trained with whole person image w/o mask and corresponding driving pose sequence.</li>
<li>Dynamictrl-5B-Mask_B01: trained with <strong>masked background</strong> in person image and pose sequence.</li>
<li>Dynamictrl-5B-Mask_C01: trained with <strong>masked clothes</strong> in person image and pose sequence.</li>
</ol>

| name | Details | HF weights 🤗 |
|:---|:---:|:---:|
| DynamiCtrl-5B | SFT w/ whole image | [dynamictrl-5B](https://huggingface.co/gulucaptain/DynamiCtrl) |
| Dynamictrl-5B-Mask_B01 | SFT w/ masked <font color=CornflowerBlue>B</font>ackground | [dynamictrl-5B-mask-B01](https://huggingface.co/gulucaptain/Dynamictrl-Mask_B01) |
| Dynamictrl-5B-Mask_C01 | SFT w/ masked human <font color=CornflowerBlue>C</font>lothing | [dynamictrl-5B-mask-C01](#) |

(Coming soon...) We are actively open-sourcing the Dynamictrl-5B-Mask_B01 and Dynamictrl-5B-Mask_C01 models.

[Causal VAE](https://arxiv.org/abs/2408.06072), [T5](https://arxiv.org/abs/1910.10683) is used as our latent features and text encoder, you can directly download from [Here](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V) and put it under *./checkpoints/*:

```bash
cd checkpoints

git lfs install

git clone https://huggingface.co/gulucaptain/DynamiCtrl

git clone https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V

# Replace the "transformer" folder in CogVideoX with the "transformer" folder in DynamiCtrl
cd CogVideoX1.5-5B-I2V
rm -rf transformer
cp ../DynamiCtrl/transformer ./

mv CogVideoX1.5-5B-I2V DynamiCtrl # Rename
```

Download the checkponts of [DWPose](https://github.com/IDEA-Research/DWPose) for human pose estimation:

```bash
cd checkpoints

git clone https://huggingface.co/yzd-v/DWPose

# Change the paths in ./dwpose/wholebody.py Lines 15 and 16.
```

## 👍 Quick Start

### Direct Inference w/ Driving Video

```bash
image="/home/user/code/DynamiCtrl/assets/human1.jpg"
video="/home/user/code/DynamiCtrl/assets/video1.mp4"

model_path="./checkpoints/DynamiCtrl"
output="./outputs"

CUDA_VISIBLE_DEVICES=0 python scripts/dynamictrl_inference.py \
    --prompt="Input the test prompt here." \
    --reference_image_path=$image \
    --ori_driving_video=$video \
    --model_path=$model_path \
    --generate_type="i2v" \
    --output_path=$output \
    --num_inference_steps=25 \
    --width=768 \
    --height=1360 \
    --num_frames=37 \
    --pose_control_function="padaln" \
    --guidance_scale=3.0 \
    --seed=42 \
```

<font color=Coral>Tips:</font> When using the trained DynamiCtrl model without a masked area, you should ensure that the prompt content aligns with the provided human image, including the person's appearance and the background description.

You can write the prompt by youself or we also provide a guidance to use Qwen2-VL tool to help you write the prompt corresponding to the content of image automatically, you can follow this blog [How to use Qwen2-VL](https://blog.csdn.net/zxs0222/article/details/144698753?spm=1001.2014.3001.5501).

### Inference w/ Maksed Human Image

Thanks to the proposed "Joint-text" paradigm for this task, we can achieve fine-grained control over human motion, including background and clothing areas. It is also easy to use, just provide a human image with blacked-out areas, and you can directly run the inference script for generation. Note to replace the model path. How to automatically get the mask area? You can follow this instruction: [How to get mask of subject](https://blog.csdn.net/zxs0222/article/details/147604020?spm=1001.2014.3001.5501).

```bash
image="/home/user/code/DynamiCtrl/assets/maksed_human1.jpg"
video="/home/user/code/DynamiCtrl/assets/video1.mp4"

model_path="./checkpoints/Dynamictrl-5B-Mask_B01" # or "./checkpoints/Dynamictrl-5B-Mask_C01"
output="./outputs"

CUDA_VISIBLE_DEVICES=0 python scripts/dynamictrl_inference.py \
    --prompt="Input the test prompt here." \
    --reference_image_path=$image \
    --ori_driving_video=$video \
    --model_path=$model_path \
    --generate_type="i2v" \
    --output_path=$output \
    --num_inference_steps=25 \
    --width=768 \
    --height=1360 \
    --num_frames=37 \
    --pose_control_function="padaln" \
    --guidance_scale=3.0 \
    --seed=42 \
```

<font color=Coral>Tips:</font> Although the "Dynamictrl-5B-Mask_B01" and "Dynamictrl-5B-Mask_C01" models are trained with masked human images, you can still directly test whole human images with these two models. Sometimes, they may even perform better than the basic "Dynamictrl-5B" model.

### Training

Please find the instructions on data preparation and training [here](./docs/finetune.md).

## 🔅 More Applications:

### Digital Human (contains long video performance)

Show cases: long video with 12 seconds, driving by the same audio.

<table class="center">
<tr>
  <td>
    <video src="https://github.com/user-attachments/assets/9a288ed5-2eb9-4035-9b1d-1a7bda81a81e" width="300px" height="100%" controls autoplay loop></video>
  </td>
  <td>
    <video src="https://github.com/user-attachments/assets/797c622e-8ee0-4038-912d-e5d0c9b2ea43" width="300px" height="100%" controls autoplay loop></video>
  </td>
</tr>
</table>

The identities of the digital human are generated by vivo's BlueLM model (Text to image generation).

Two steps to generate a digital human:

1. Prepare a human image and a guided pose video, and generate the video materials using our DynamiCtrl.

2. Use the output video and an audio file, and apply MuseTalk [MuseTalk](https://github.com/TMElyralab/MuseTalk) to generate the correct lip movements.



## 📍 Citation 

If you find this repository helpful, please consider citing:

```
@article{zhao2025dynamictrl,
      title={DynamiCtrl: Rethinking the Basic Structure and the Role of Text for High-quality Human Image Animation}, 
      author={Haoyu, Zhao and Zhongang, Qi and Cong, Wang and Qingping, Zheng and Guansong, Lu and Fei, Chen and Hang, Xu and Zuxuan, Wu},
      year={2025},
      journal={arXiv:2503.21246},
}
``` 

## 💗 Acknowledgements

This repository borrows heavily from [CogVideoX](https://github.com/THUDM/CogVideo). Thanks to the authors for sharing their code and models.

## 🧿 Maintenance

This is the codebase for our research work. We are still working hard to update this repo, and more details are coming in days.