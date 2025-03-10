## DynamiCtrl: Rethinking the Basic Structure and the Role of Text for High-quality Human Image Animation

[Haoyu Zhao](https://scholar.google.com/citations?user=pCGM7jwAAAAJ&hl=zh-CN&oi=ao/), [Zhongang Qi](https://scholar.google.com/citations?user=zJvrrusAAAAJ&hl=en/), [Cong Wang](#), [Qingqing Zheng](https://scholar.google.com.hk/citations?user=l0Y7emkAAAAJ&hl=zh-CN&oi=ao/), [Guansong Lu](https://scholar.google.com.hk/citations?user=YIt8thUAAAAJ&hl=zh-CN&oi=ao), [Fei Chen](#), [Hang Xu](https://scholar.google.com.hk/citations?user=J_8TX6sAAAAJ&hl=zh-CN&oi=ao) and [Zuxuan Wu](https://scholar.google.com.hk/citations?user=7t12hVkAAAAJ&hl=zh-CN&oi=ao)

<a href='#'><img src='https://img.shields.io/badge/ArXiv-DynamiCtrl-red'></a> 
<a href='#'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](#)
[![GitHub](https://img.shields.io/github/stars/gulucaptain/DynamiCtrl?style=social)](https://github.com/gulucaptain/DynamiCtrl)


<table class="center">
<tr>
  <td>
    <video width="768px" height="100%" controls>
    <source src="https://raw.githubusercontent.com/gulucaptain/DynamiCtrl/main/assets/videos/video1.mp4" type="video/mp4">
    </video>
  </td>
  <td>
    <video width="768px" height="100%" controls>
    <source src="https://raw.githubusercontent.com/gulucaptain/DynamiCtrl/main/assets/videos/video2.mp4" type="video/mp4">
    </video>
  </td>
</tr>
<tr>
<td><p style="font-size:12px; text-align:justify;">Prompt: “The person in the image is wearing a traditional outfit with intricate embroidery and embellishments. The outfit features a blue and gold color scheme with detailed floral patterns. The background is dark and blurred, which helps to highlight the person and their attire. The lighting is soft and warm, creating a serene and elegant atmosphere.”</p></td>
<td><p style="font-size:12px; text-align:justify;">Prompt: “The person in the image is a woman with long, blonde hair styled in loose waves. She is wearing a form-fitting, sleeveless top with a high neckline and a small cutout at the chest. The top is beige and has a strap across her chest. She is also wearing a black belt with a pouch attached to it. Around her neck, she has a turquoise pendant necklace. The background appears to be a dimly lit, urban environment with a warm, golden glow."</p></td>
</tr>
<tr>
  <td>
    <video width="768px" height="100%" controls>
    <source src="https://raw.githubusercontent.com/gulucaptain/DynamiCtrl/main/assets/videos/video3.mp4" type="video/mp4">
    </video>
  </td>
  <td>
    <video width="768px" height="100%" controls>
    <source src="https://raw.githubusercontent.com/gulucaptain/DynamiCtrl/main/assets/videos/video4.mp4" type="video/mp4">
    </video>
  </td>
</tr>
<tr>
<td><p style="font-size:12px; text-align:justify;">Prompt: “The person in the image is wearing a black, form-fitting one-piece outfit and a pair of VR goggles. They are walking down a busy street with numerous people and colorful neon signs in the background. The street appears to be a bustling urban area, possibly in a city known for its vibrant nightlife and entertainment. The lighting and signage suggest a lively atmosphere, typical of a cityscape at night."</p></td>
<td><p style="font-size:12px; text-align:justify;">Prompt: “The image depicts a stylized, animated character standing amidst a chaotic and dynamic background. The character is dressed in a blue suit with a red cape, featuring a prominent "S" emblem on the chest. The suit has a belt with pouches and a utility belt. The character has spiky hair and is standing on a pile of debris and rubble, suggesting a scene of destruction or battle. The background is filled with glowing, fiery elements and a sense of motion, adding to the dramatic and intense atmosphere of the scene."</p></td>
</tr>
</table>


## 🎏 Abstract
<b>TL; DR: <font color="red">DynamiCtrl</font> is the first framework to introduce text to the human image animation task and achieve pose control within the MM-DiT architecture.</b>

<details><summary>CLICK for the full abstract</summary>


> Human image animation has recently gained significant attention due to advancements in generative models. However, existing methods still face two major challenges: (1) architectural limitations—most models rely on U-Net, which underperforms compared to the MM-DiT; and (2) the neglect of textual information, which can enhance controllability. In this work, we introduce <font color="red">DynamiCtrl</font>, a novel framework that not only explores different pose-guided control structures in MM-DiT, but also reemphasizes the crucial role of text in this task. Specifically, we employ a Shared VAE encoder for both reference images and driving pose videos, eliminating the need for an additional pose encoder and simplifying the overall framework. To incorporate pose features into the full attention blocks, we propose Pose-adaptive Layer Norm (PadaLN), which utilizes adaptive layer normalization to encode sparse pose features. The encoded features are directly added to the visual input, preserving the spatiotemporal consistency of the backbone while effectively introducing pose control into MM-DiT. Furthermore, within the full attention mechanism, we align textual and visual features to enhance controllability. By leveraging text, we not only enable fine-grained control over the generated content, but also, for the first time, achieve simultaneous control over both background and motion. Experimental results verify the superiority of DynamiCtrl on benchmark datasets, demonstrating its strong identity preservation, heterogeneous character driving, background controllability, and high-quality synthesis. The source code will be made publicly available soon.
</details>

## 🚧 Todo

<details><summary>Click for Previous todos </summary>

- [x] Release the project page and demos
- [x] Paper on Arxiv
</details>

- [ ] Release code
- [ ] Release model

## 📋 Changelog
Code coming soon!
- 2025.03.10 Project Online!

## ⚔️ DynamiCtrl Human Motion Video Generation

### Background Control (contains long video performance)

We first refocus on the role of text for this task and find that fine-grained textual information helps improve video quality. In particular, we can achieve <font color="green">background controllability</font> using different prompts.

<table class="center">
<tr>
  <td>
    <img src="https://raw.githubusercontent.com/gulucaptain/DynamiCtrl/main/assets/videos/background_control/bg_image1.jpg" width="192px" height="100%">
  </td>
  <td>
    <video width="192px" height="100%" controls>
    <source src="https://raw.githubusercontent.com/gulucaptain/DynamiCtrl/main/assets/videos/background_control/bg_video1.mp4" type="video/mp4">
    </video>
  </td>
  <td>
    <img width="192px" height="100%" src="https://raw.githubusercontent.com/gulucaptain/DynamiCtrl/main/assets/videos/background_control/bg_image2.jpg">
  </td>
  <td>
    <video width="192px" height="100%" controls>
    <source src="https://raw.githubusercontent.com/gulucaptain/DynamiCtrl/main/assets/videos/background_control/bg_video2.mp4" type="video/mp4">
    </video>
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
  <td><img src="https://raw.githubusercontent.com/gulucaptain/DynamiCtrl/main/assets/videos/background_control/bg_image3.jpg"  width="192px" height="100%"></td>
  <td>
    <video width="192px" height="100%" controls>
    <source src="https://raw.githubusercontent.com/gulucaptain/DynamiCtrl/main/assets/videos/background_control/bg_video5.mov" type="video/mp4">
    </video>
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
    <video width="800px" height="100%" controls>
    <source src="https://raw.githubusercontent.com/gulucaptain/DynamiCtrl/main/assets/videos/re_video1.mp4" type="video/mp4">
    </video>
  </td>
  <td>
    <video width="800px" height="100%" controls>
    <source src="https://raw.githubusercontent.com/gulucaptain/DynamiCtrl/main/assets/videos/re_video2.mp4" type="video/mp4">
    </video>
  </td>
</tr>
<tr>
  <td>
    <video width="800px" height="100%" controls>
    <source src="https://raw.githubusercontent.com/gulucaptain/DynamiCtrl/main/assets/videos/re_video3.mp4" type="video/mp4">
    </video>
  </td>
  <td>
    <video width="800px" height="100%" controls>
    <source src="https://raw.githubusercontent.com/gulucaptain/DynamiCtrl/main/assets/videos/re_video4.mp4" type="video/mp4">
    </video>
  </td>
</tr>
</table>


### Applications: Digital Human (contains long video performance)

Show cases: long video with 12 seconds, driving by the same audio.

<table class="center">
<tr>
  <td>
    <video width="512px" height="100%" controls>
    <source src="https://raw.githubusercontent.com/gulucaptain/DynamiCtrl/main/assets/videos/digital_human/person1_output.mp4" type="video/mp4">
    </video>
  </td>
  <td>
    <video width="512px" height="100%" controls>
    <source src="https://raw.githubusercontent.com/gulucaptain/DynamiCtrl/main/assets/videos/digital_human/person2_output.mp4" type="video/mp4">
    </video>
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
      author={Haoyu Zhao and Zhongang Qi and Cong Wang and Qingping Zheng and Guansong Lu and Fei Chen and Hang Xu and Zuxuan Wu},
      year={2025},
      journal={arXiv:xxxx.xxxxx},
}
``` 


## 💗 Acknowledgements

This repository borrows heavily from [CogVideoX](https://github.com/THUDM/CogVideo). Thanks to the authors for sharing their code and models.

## 🧿 Maintenance

This is the codebase for our research work. We are still working hard to update this repo, and more details are coming in days.