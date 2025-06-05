# RecRecNet: Rectangling Rectified Wide-Angle Images by Thin-Plate Spline Model and DoF-based Curriculum Learning

## Introduction
This is the official implementation for [RecRecNet](https://arxiv.org/abs/2301.01661) (ICCV2023).

[Kang Liao](https://kangliao929.github.io/)<sup>&dagger;</sup>, [Lang Nie](https://nie-lang.github.io/)<sup>&dagger;</sup>,  [Chunyu Lin](http://faculty.bjtu.edu.cn/8549/), [Zishuo Zheng](), [Yao Zhao](http://mepro.bjtu.edu.cn/zhaoyao/e_index.htm) 


<div align="center">
  <img src="https://github.com/KangLiao929/RecRecNet/blob/main/img/pipeline.png" height="240">
</div>

> ### Problem
> Given a rectified wide-angle image, RecRecNet aims to construct a win-win representation on both image content and boundary, with the perspective of the rectangling technique.
>  ### Features
>  * First win-win representation of the large field-of-view (FoV) vision
>  * A thin-plate spline (TPS) motion module is proposed to flexibly formulate the non-linear and non-rigid rectangling transformation
>  * A DoF-based curriculum learning is designed to grasp the progressive deformation rules and relieve the burden of complex structure approximation
>  * An in-depth analysis of why the deformed image boundary can significantly influence the vision perception models

<div align="center">
  <img src="https://github.com/KangLiao929/RecRecNet/blob/main/img/result.png" height="400">
</div>

## Updates
Our recent work MOWA can solve multiple image warping tasks in a single and unified framework, including image rectangling, distortion rectification, and other practical tasks. Check out more details [here](https://kangliao929.github.io/projects/mowa/)!

## Installation
Using the virtual environment (conda) to run the code is recommended.
```
conda create -n recrecnet python=3.6
conda activate recrecnet
pip install -r requirements.txt
```

## Dataset
We constructed the first dataset for the rectified wide-angle rectangling task. The structure of the original rectified wide-angle image was first optimized by an energy function with line-preserving mesh deformation, as proposed in [He et al.](https://kaiminghe.github.io/publications/sig13pano.pdf). And then we carefully filtered all results and repeated the selection process three times. The dataset can be downloaded from: train.zip-[Google Drive](https://drive.google.com/file/d/1tUFgMMkSvdGtv7OYP1Z-0HV64suGnjUQ/view?usp=sharing), test.zip-[Google Drive](https://drive.google.com/file/d/1qpnqIYnHOYJQh4p-QdfEKGNvx1w_ekI8/view?usp=sharing).

## Pretrained Model
Download the pretrained model [Google Drive](https://drive.google.com/file/d/1y9iTfWCycS3BAFViMsClbur11IY-HgXf/view?usp=sharing) and put it into the ```.\checkpoint``` folder. The dataset and pretrained model are also available at [Baidu Netdisk](https://pan.baidu.com/s/1SZmMEsZ_egTpy38TCzlyTQ?pwd=s1vg).

## Training
### Curriculum Generation
Generate the curriculum to grasp the progressive deformation rules of rectangling. The source image can be collected from ImageNet or COCO. Please set the suitable $path1$, $path2$, and $dof$ (4 and 8) and run:
```
sh scripts/curriculum_gen.sh
```
### TPS Model Training
Customize the paths of 4-dof dataset, 8-dof dataset, and wide-angle image rectangling dataset, and run:
```
sh scripts/train.sh
```
## Testing
Customize the paths of checkpoint and test set, and run:
```
sh scripts/test.sh
```
The rectangling image and its corresponding warping mesh (formed by predicted TPS control points) can be found in the ```.\results``` folder.

<div align="center">
  <img src="https://github.com/KangLiao929/RecRecNet/blob/main/img/results.png" height="400">
</div>

## Citation
If you feel RecRecNet is helpful in your research, please consider referring to it:
```bibtex
@article{liao2023recrecnet,
  title={RecRecNet: Rectangling rectified wide-angle images by thin-plate spline model and DoF-based curriculum learning},
  author={Liao, Kang and Nie, Lang and Lin, Chunyu and Zheng, Zishuo and Zhao, Yao},
  journal={arXiv preprint arXiv:2301.01661},
  year={2023}
}
```
