<p align="center" width="60%">
<img src="assets/logo.png" alt="OpenEMMA" style="width: 35%; min-width: 200px; display: block; margin: auto; background-color: transparent;">
</p>

<p align="center">
    <a href="README.md"><strong>English</strong></a> | <a href="README_zh-CN.md"><strong>中文</strong></a> | <a href="README_ja-JP.md"><strong>日本語</strong></a>
</p>

<div id="top" align="center">

![Code License](https://img.shields.io/badge/Code%20License-Apache%202.0-brightgreen)
[![arXiv](https://img.shields.io/badge/arXiv-2412.15208-b31b1b.svg)](https://arxiv.org/abs/2412.15208)

</div>


# OpenEMMA: Open-Source Multimodal Model for End-to-End Autonomous Driving
**OpenEMMA** is an open-source implementation of  [Waymo's End-to-End Multimodal Model for Autonomous Driving (EMMA)](https://waymo.com/blog/2024/10/introducing-emma/), offering an end-to-end framework for motion planning in autonomous vehicles. **OpenEMMA** leverages the pretrained world knowledge of Vision Language Models  (VLMs), such as GPT-4 and LLaVA, to integrate text and front-view camera inputs, enabling precise predictions of future ego waypoints and providing decision rationales. Our goal is to provide accessible tools for researchers and developers to advance autonomous driving research and applications.

<div align="center">
  <img src="assets/EMMA-Paper-1__3_.webp" alt="EMMA diagram" width="800"/>
  <p><em>Figure 1. EMMA: Waymo's End-to-End Multimodal Model for Autonomous Driving.</em></p>
</div>

<div align="center">
  <img src="assets/openemma-pipeline.png" alt="OpenEMMA diagram" width="800"/>
  <p><em>Figure 2. OpenEMMA: Our Open-Source End-to-End Autonomous Driving Framework based on Pre-trained VLMs.</em></p>
</div>

### News
- **[2025/1/12]** 🔥**OpenEMMA** is now available as a PyPI package! You can install it using `pip install openemma`. 
- **[2024/12/19]** 🔥We released **OpenEMMA**, an open-source project for end-to-end motion planning in autonomous driving tasks. Explore our [paper](https://arxiv.org/abs/2412.15208) for more details.

### Table of Contents
- [Demos](#demos)
- [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)
- [Citation](#citation)

### Demos
![](assets/scene-0061.gif)

![](assets/scene-0103.gif)

![](assets/scene-1077.gif)

### Installation  
To get started with OpenEMMA, follow these steps to set up your environment and dependencies.

1. **Environment Setup**  
   Set up a Conda environment for OpenEMMA with Python 3.8:
   ```bash
   conda create -n openemma python=3.8
   conda activate openemma
   ```
2. **Install OpenEMMA**   
You can now install OpenEMMA with a single command using PyPI:
    ```bash
    pip install openemma
    ```
    Alternatively, follow these steps:
    - **Clone OpenEMMA Repository**   
        Clone the OpenEMMA repository and navigate to the root directory:
        ```bash
        git clone git@github.com:taco-group/OpenEMMA.git
        cd OpenEMMA
        ```
    - **Install Dependencies**  
        Ensure you have cudatoolkit installed. If not, use the following command:
        ```bash
        conda install nvidia/label/cuda-12.4.0::cuda-toolkit
        ```
        To install the core packages required for OpenEMMA, run the following command:
        ```bash
        pip install -r requirements.txt
        ```
        This will install all dependencies, including those for YOLO-3D, an external tool used for critical object detection. The weights needed to run YOLO-3D will be automatically downloaded during the first execution.

3. **Set up GPT-4 API Access**  
    To enable GPT-4’s reasoning capabilities, obtain an API key from OpenAI. You can add your API key directly in the code where prompted or set it up as an environment variable:
    ```bash
    export OPENAI_API_KEY="your_openai_api_key"
    ```
    This allows OpenEMMA to access GPT-4 for generating future waypoints and decision rationales.

### Usage  
After setting up the environment, you can start using OpenEMMA with the following instructions:

1. **Prepare Input Data**   
    Download and extract the [nuScenes dataset](https://www.nuscenes.org/nuscenes#download)
    
2. **Run OpenEMMA**  
    Use the following command to execute OpenEMMA's main script:
    - PyPI:
    ```bash
    openemma \
        --model-path qwen \
        --dataroot [dir-of-nuScenes-dataset] \
        --version [version-of-nuScenes-dataset] \
        --method openemma
    ```
    - Github Repo:
    ```bash
    python main.py \
        --model-path qwen \
        --dataroot [dir-of-nuscnse-dataset] \
        --version [version-of-nuscnse-dataset] \
        --method openemma
    ```

    Currently, we support the following models: `GPT-4o`, `LLaVA-1.6-Mistral-7B`, `Llama-3.2-11B-Vision-Instruct`, and `Qwen2-VL-7B-Instruct`. To use a specific model, simply pass `gpt`, `llava`, `llama`, and `qwen`as the argument to `--model-path`.

3. **Output Interpretation**   
    After running the model, OpenEMMA generates the following output in the `./qwen-results` location:

    - **Waypoints**: A list of future waypoints predicting the ego vehicle’s trajectory.

    - **Decision Rationales**: Text explanations of the model’s reasoning, including scene context, critical objects, and behavior decisions.

    - **Annotated Images**: Visualizations of the planned trajectory and detected critical objects overlaid on the original images.

    - **Compiled Video**: A video (e.g., `output_video.mp4`) created from the annotated images, showing the predicted path over time.

## Contact
For help or issues using this package, please submit a GitHub issue.

For personal communication related to this project, please contact Shuo Xing (shuoxing@tamu.edu).

## Citation
We are more than happy if this code is helpful to your work. 
If you use our code or extend our work, please consider citing our paper:

```bibtex

@article{openemma,
	author = {Xing, Shuo and Qian, Chengyuan and Wang, Yuping and Hua, Hongyuan and Tian, Kexin and Zhou, Yang and Tu, Zhengzhong},
	title = {OpenEMMA: Open-Source Multimodal Model for End-to-End Autonomous Driving},
	journal = {arXiv},
	year = {2024},
	month = dec,
	eprint = {2412.15208},
	doi = {10.48550/arXiv.2412.15208}
}


