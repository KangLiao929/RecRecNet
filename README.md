# RecRecNet: Rectangling Rectified Wide-Angle Images by Thin-Plate Spline Model and DoF-based Curriculum Learning

## Introduction
This is the official implementation for [RecRecNet](https://arxiv.org/abs/2301.01661) (ICCV2023).

> ### Problem
> Given a rectified wide-angle image, RecRecNet aims to construct a win-win representation on both image content and boundary, with the perspective of the rectangling technique.
>  ### Features
>  * First win-win representation of the large filed-of-view vision
>  * A thin-plate spline (TPS) motion module is proposed to flexibly formulate the non-linear and non-rigid rectangling transformation
>  * A DoF-based curriculum learning is designed to grasp the progressive deformation rules and relieve the burden of complex structure approximation
>  * An in-depth analysis of why the deformed image boundary can significantly influence the vision perception models

Code and dataset will be released before the main conference.

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
