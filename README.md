# Lite-HRNet

[Lite-HRNet (Yu *et al.*, 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Yu_Lite-HRNet_A_Lightweight_High-Resolution_Network_CVPR_2021_paper.html) is a state-of-the-art lightweight deep learning model for top-down pose estimation that is publicly available [here](https://github.com/HRNet/Lite-HRNet). [MMPose](https://github.com/open-mmlab/mmpose), an open-source toolbox from [OpenMMLab](https://github.com/open-mmlab) is required to build, train and evaluate Lite-HRNet models. In 2023, a new version of MMPose (version [1.x](https://mmpose.readthedocs.io/en/latest/)) was released with several major changes from the old version (version [0.x](https://mmpose.readthedocs.io/en/0.x/)). This project uses the new version of MMPose (version 1.x).

## MMPose Installation Instructions

For more information about any of the following installation steps, see the MMPose installation instructions [here](https://mmpose.readthedocs.io/en/latest/installation.html).

### 1. Clone this GitHub repository

```
git clone https://github.com/matthieuruthven/New-Lite-HRNet-Project.git
```

### 2. Create conda environment with Python 3.8 and activate it

```
conda create -n openmmlab python=3.8 -y
conda activate openmmlab
```

### 3. Install PyTorch using conda

Install PyTorch using the following command:
```
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```
For more information, see the installation instructions on the PyTorch [website](https://pytorch.org/get-started/locally/).

### 4. Install MIM using pip

Install [MIM](https://github.com/open-mmlab/mim), a package management software, using the following command:
```
pip install -U openmim
```

### 5. Install MMEngine using MIM

Install MMEngine using the following command:
```
mim install mmengine
```

### 6. Install MMCV using MIM

Install MMCV using the following command:
```
mim install "mmcv>=2.0.1"
```

### 7. Install MMPose using MIM

Install MMPose using the following command:
```
mim install "mmpose>=1.1.0"
```

## Extending MMPose for SPEED+ Dataset Loading

MMPose includes data loaders for datasets such as the [COCO](https://cocodataset.org/#home) and [MPII Human Pose](http://human-pose.mpi-inf.mpg.de/) datasets but not for the [SPEED+](https://zenodo.org/record/5588480) dataset. MMPose data loaders for the SPEED+ dataset must therefore be created. 

### 1. Add a custom Python class

For more information about any of the following steps, see the instructions [here](https://mmpose.readthedocs.io/en/latest/advanced_guides/customize_datasets.html).

1. Add the `speedplus_dataset.py` file from this GitHub repository to the `mmpose/datasets/datasets` folder

2. Update the `__init__.py` file in the `mmpose/datasets/datasets` folder

### 2. Add a custom dataset transformation

For more information, see the instructions [here](https://mmpose.readthedocs.io/en/latest/advanced_guides/customize_transforms.html).

1. Add the `my_transforms.py` file from this GitHub repository to the `mmpose/datasets/transforms` folder

2. Update the `__init__.py` file in the `mmpose/datasets/transforms` folder