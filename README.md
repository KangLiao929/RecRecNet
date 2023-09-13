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

## Installation
Using the virtual environment (conda) to run the code is recommended.
```
conda create -n recrecnet python=3.6
conda activate recrecnet
pip install -r requirements.txt
```

## Dataset
We construct the first dataset for the rectified wide-angle rectangling task. The structure of the original rectified wide-angle image is first optimized by an energy function with line-preserving mesh deformation, as proposed in [He et al.](https://kaiminghe.github.io/publications/sig13pano.pdf). And then we carefully filter all results and repeat the selection process three times. The dataset can be downloaded here: [train.zip](https://drive.google.com/file/d/1tUFgMMkSvdGtv7OYP1Z-0HV64suGnjUQ/view?usp=sharing), [test.zip](https://drive.google.com/file/d/1qpnqIYnHOYJQh4p-QdfEKGNvx1w_ekI8/view?usp=sharing).

## Citation
If you feel RecRecNet is helpful in your research, please consider referring it:
```bibtex
@article{liao2023recrecnet,
  title={RecRecNet: Rectangling rectified wide-angle images by thin-plate spline model and DoF-based curriculum learning},
  author={Liao, Kang and Nie, Lang and Lin, Chunyu and Zheng, Zishuo and Zhao, Yao},
  journal={arXiv preprint arXiv:2301.01661},
  year={2023}
}
```
