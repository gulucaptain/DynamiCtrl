## DynamiCtrl: Rethinking the Basic Structure and the Role of Text for High-quality Human Image Animation

[Haoyu Zhao](https://scholar.google.com/citations?user=pCGM7jwAAAAJ&hl=zh-CN&oi=ao/), [Zhongang Qi](https://scholar.google.com/citations?user=zJvrrusAAAAJ&hl=en/), [Cong Wang](#), [Qingqing Zheng](https://scholar.google.com.hk/citations?user=l0Y7emkAAAAJ&hl=zh-CN&oi=ao/), [Guansong Lu](https://scholar.google.com.hk/citations?user=YIt8thUAAAAJ&hl=zh-CN&oi=ao), [Fei Chen](#), [Hang Xu](https://scholar.google.com.hk/citations?user=J_8TX6sAAAAJ&hl=zh-CN&oi=ao) and [Zuxuan Wu](https://scholar.google.com.hk/citations?user=7t12hVkAAAAJ&hl=zh-CN&oi=ao)

<a href='https://arxiv.org/abs/2503.21246'><img src='https://img.shields.io/badge/ArXiv-DynamiCtrl-red'></a> 
<a href='https://gulucaptain.github.io/DynamiCtrl/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://youtu.be/Mu_pNXM4PcE'><img src='https://img.shields.io/badge/YouTube-Intro-yellow'></a>
[![GitHub](https://img.shields.io/github/stars/gulucaptain/DynamiCtrl?style=social)](https://github.com/gulucaptain/DynamiCtrl)
<!-- [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](#) -->

## YouTube Overview

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/Mu_pNXM4PcE/0.jpg)](https://www.youtube.com/watch?v=Mu_pNXM4PcE)
Please click to watch.

### Generation with Image, Pose and Prompts

<table class="center">
<tr>
  <td>
    <video src="https://github.com/user-attachments/assets/ef21d87c-34b8-4cad-86b7-f23fc57e8ddb" width="768px" height="100%" controls autoplay loop></video>
  </td>
  <td>
    <video src="https://github.com/user-attachments/assets/54748782-cf2d-415c-824c-f2aa97754ab6" width="768px" height="100%" controls autoplay loop></video>
  </td>
</tr>
<tr>
<td><p style="font-size:10px; text-align:justify;">Prompt: “The person in the image is wearing a traditional outfit with intricate embroidery and embellishments. The outfit features a blue and gold color scheme with detailed floral patterns. The background is dark and blurred, which helps to highlight the person and their attire. The lighting is soft and warm, creating a serene and elegant atmosphere.”</p></td>
<td><p style="font-size:10px; text-align:justify;">Prompt: “The person in the image is a woman with long, blonde hair styled in loose waves. She is wearing a form-fitting, sleeveless top with a high neckline and a small cutout at the chest. The top is beige and has a strap across her chest. She is also wearing a black belt with a pouch attached to it. Around her neck, she has a turquoise pendant necklace. The background appears to be a dimly lit, urban environment with a warm, golden glow."</p></td>
</tr>
<tr>
  <td>
    <video src="https://github.com/user-attachments/assets/10d955dc-f00a-45c0-8632-2b2a93086281" width="768px" height="100%" controls autoplay loop></video>
  </td>
  <td>
    <video src="https://github.com/user-attachments/assets/22ab1e13-222e-4f87-bf9e-2fca53fcb76d" width="768px" height="100%" controls autoplay loop></video>
  </td>
</tr>
<tr>
<td><p style="font-size:10px; text-align:justify;">Prompt: “The person in the image is wearing a black, form-fitting one-piece outfit and a pair of VR goggles. They are walking down a busy street with numerous people and colorful neon signs in the background. The street appears to be a bustling urban area, possibly in a city known for its vibrant nightlife and entertainment. The lighting and signage suggest a lively atmosphere, typical of a cityscape at night."</p></td>
<td><p style="font-size:10px; text-align:justify;">Prompt: “The image depicts a stylized, animated character standing amidst a chaotic and dynamic background. The character is dressed in a blue suit with a red cape, featuring a prominent "S" emblem on the chest. The suit has a belt with pouches and a utility belt. The character has spiky hair and is standing on a pile of debris and rubble, suggesting a scene of destruction or battle. The background is filled with glowing, fiery elements and a sense of motion, adding to the dramatic and intense atmosphere of the scene."</p></td>
</tr>
</table>


## 🎏 Abstract
<b>TL; DR: <font color="red">DynamiCtrl</font> is the first framework to introduce text to the human image animation task and achieve pose control within the diffusion transformer (DiT) architecture.</b>

<details><summary>CLICK for the full abstract</summary>


> With diffusion transformer (DiT) excelling in video generation, its use in specific tasks has drawn increasing attention. However, adapting DiT for pose-guided human image animation faces two core challenges: (a) existing U-Net-based pose control methods may be suboptimal for the DiT backbone; and (b) removing text guidance, as in previous approaches, often leads to semantic loss and model degradation. To address these issues, we propose DynamiCtrl, a novel framework for human animation in video DiT architecture. Specifically, we use a shared VAE encoder for human images and driving poses, unifying them into a common latent space, maintaining pose fidelity, and eliminating the need for an expert pose encoder during video denoising. To integrate pose control into the DiT backbone effectively, we propose a novel Pose-adaptive Layer Norm model. It injects normalized pose features into the denoising process via conditioning on visual tokens, enabling seamless and scalable pose control across DiT blocks. Furthermore, to overcome the shortcomings of text removal, we introduce the "Joint-text" paradigm, which preserves the role of text embeddings to provide global semantic context. Through full-attention blocks, image and pose features are aligned with text features, enhancing semantic consistency, leveraging pretrained knowledge, and enabling multi-level control. Experiments verify the superiority of DynamiCtrl on benchmark and self-collected data (e.g., achieving the best LPIPS of 0.166), demonstrating strong character control and high-quality synthesis.
</details>

## 🚧 Todo

<details><summary>Click for Previous todos </summary>

- [x] Release the project page and demos
- [x] Paper on Arxiv
</details>

- [ ] Release inference code
- [ ] Release model
- [ ] Release training code

## 📋 Changelog
Code coming soon!
- 2025.03.30 Project page and demos released!
- 2025.03.10 Project Online!

## ⚔️ DynamiCtrl Human Motion Video Generation

### Background Control (contains long video performance)

We first refocus on the role of text for this task and find that fine-grained textual information helps improve video quality. In particular, we can achieve <font color="green">background controllability</font> using different prompts.

<table class="center">
<tr>
  <td>
    <img src="https://github.com/user-attachments/assets/0e603117-35af-4543-9b5b-fa86b4b44ca9" width="192px" height="100%">
  </td>
  <td>
    <video src="https://github.com/user-attachments/assets/d43a890f-62b8-4d0a-9110-da2c80067c88" width="192px" height="100%" controls autoplay loop></video>
  </td>
  <td>
    <img width="192px" height="100%" src="https://github.com/user-attachments/assets/1bb7bbfd-fb2d-4c16-913f-ed8204a40858">
  </td>
  <td>
    <video src="https://github.com/user-attachments/assets/6c586a42-51bd-4641-84f8-bb02e6883ec3" width="768px" height="100%" controls autoplay loop></video>
  </td>
</tr>
<tr>
  <td width=25% style="text-align:center;">Case (a)</td>
  <td width=25% style="text-align:center;">"->Green trees"</td>
  <td width=25% style="text-align:center;">Case (b)</td>
  <td width=25% style="text-align:center;">"->Beautiful ocean"</td>
</tr>
<tr>
  <td></td>
  <td><img src="https://github.com/user-attachments/assets/437e59f0-5f93-478e-a616-0d6d19664daa"  width="192px" height="100%"></td>
  <td>
    <video src="https://github.com/user-attachments/assets/356916f1-ff7f-466c-85b3-884fbc562893" width="768px" height="100%" controls autoplay loop></video>
  </td>
  <td></td>
</tr>
<tr>
  <td width=25% style="text-align:center;"></td>
  <td width=25% style="text-align:center;">Case (c)</td>
  <td width=25% style="text-align:center;">"Across seven different backgrounds <font color="red"> (Long video over 200 frames)</font>"</td>
  <td width=25% style="text-align:center;"></td>
</tr>
</table>

<details><summary>CLICK for the full prompts used in Case (c).</summary>


> Scene 1: The person in the image is wearing a white, knee-length dress with short sleeves and a square neckline. The dress features lace detailing and a ruffled hem. The person is also wearing clear, open-toed sandals. The background shows a bustling futuristic city at night, with neon lights reflecting off the wet streets and flying cars zooming above.

> Scene 2: The person in the image is wearing a white, knee-length dress with short sleeves and a square neckline. The dress features lace detailing and a ruffled hem. The person is also wearing clear, open-toed sandals. The background shows a vibrant market street in a Middle Eastern bazaar, filled with colorful fabrics, exotic spices, and merchants calling out to customers.

> Scene 3: The person in the image is wearing a white, knee-length dress with short sleeves and a square neckline. The dress features lace detailing and a ruffled hem. The person is also wearing clear, open-toed sandals. The background shows a mystical ancient temple hidden deep in the jungle, covered in vines, with glowing runes carved into the stone walls.

> Scene 4: The person in the image is wearing a white, knee-length dress with short sleeves and a square neckline. The dress features lace detailing and a ruffled hem. The person is also wearing clear, open-toed sandals. The background shows a sunny beach with golden sand, gentle ocean waves rolling onto the shore, and palm trees swaying in the breeze.

> Scene 5: The person in the image is wearing a white, knee-length dress with short sleeves and a square neckline. The dress features lace detailing and a ruffled hem. The person is also wearing clear, open-toed sandals. The background shows an abandoned industrial warehouse with broken windows, scattered debris, and rusted machinery covered in dust.

> Scene 6: The person in the image is wearing a white, knee-length dress with short sleeves and a square neckline. The dress features lace detailing and a ruffled hem. The person is also wearing clear, open-toed sandals. The background shows a high-tech research lab with sleek metallic walls, glowing holographic screens, and robotic arms assembling futuristic devices.

> Scene 7: The person in the image is wearing a white, knee-length dress with short sleeves and a square neckline. The dress features lace detailing and a ruffled hem. The person is also wearing clear, open-toed sandals. The background shows a serene snowy forest with tall pine trees, soft snowflakes falling gently, and a frozen river winding through the landscape.
</details>

### Cross-identity Retargetting

<table class="center">
<tr>
  <td>
    <video src="https://github.com/user-attachments/assets/4bc68bb0-1e5e-451f-ba74-706869267e51" width="800px" height="100%" controls autoplay loop></video>
  </td>
  <td>
    <video src="https://github.com/user-attachments/assets/e524ba8f-d5a4-4713-9de6-f21c12521e39" width="800px" height="100%" controls autoplay loop></video>
  </td>
</tr>
<tr>
  <td>
    <video src="https://github.com/user-attachments/assets/2683efab-4da4-458c-a61c-d3112f6f5624" width="800px" height="100%" controls autoplay loop></video>
  </td>
  <td>
    <video src="https://github.com/user-attachments/assets/67889c3b-998d-460a-87b4-762a107629f5" width="800px" height="100%" controls autoplay loop></video>
  </td>
</tr>
</table>


### Applications: Digital Human (contains long video performance)

Show cases: long video with 12 seconds, driving by the same audio.

<table class="center">
<tr>
  <td>
    <video src="https://github.com/user-attachments/assets/4475fc8d-1736-4ee1-a0a5-e16b475b5eb3" width="512px" height="100%" controls autoplay loop></video>
  </td>
  <td>
    <video src="https://github.com/user-attachments/assets/9104ffd5-3480-4ab1-9ed2-b78661132147" width="512px" height="100%" controls autoplay loop></video>
  </td>
</tr>
</table>

The identities of the digital human are generated by vivo's BlueLM model (image generation).

Two steps to generate a digital human:

1. Prepare a human image and a guided pose video, and generate the video materials using our DynamiCtrl.

2. Use the output video and an audio file, and apply MuseTalk to generate the correct lip movements.


## 📍 Citation 

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