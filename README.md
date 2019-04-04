# Unsupervised Learning of Depth from Defocus Using a Gaussian PSF Layer
Official implementation of "Single Image Depth Estimation Trained via Depth from Defocus Cues" ([arxiv](https://arxiv.org/)).
The implementation is based on the architectures of [DeepLabV3+](https://github.com/jfzhang95/pytorch-deeplab-xception) and [Self-Attention](https://github.com/heykeetae/Self-Attention-GAN).

## Prerequisites
- Python 3.6
- Pytorch 0.4
- Numpy
- Scipy
- OpenCV
- Path
- tqdm
- h5py
### Gaussian PSF Layer
In order to build the Gaussian PSF layer on your own machine, run the following line
```
python ext/setup.py install
```

## Datasets
### [KITTI](http://www.cvlibs.net/datasets/kitti/index.php)
Download the official dataset and use [SfmLearner](https://github.com/ClementPinard/SfmLearner-Pytorch) for data preparation.

### [Make3D](http://make3d.cs.cornell.edu/data.html)
Download the official dataset and unpack into Test and Train folders.

### [NYU-V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
Download the official dataset and use [the following](https://github.com/janivanecky/Depth-Estimation/tree/master/dataset) preprocessing instructions.

### [Flowers](https://github.com/google/aperture_supervision)
Download the official dataset from [here](https://people.eecs.berkeley.edu/~pratul/) and generate all bokeh images accorfing to [Srinivasan](https://github.com/google/aperture_supervision).

### [DSLR](https://github.com/marcelampc/d3net_depth_estimation)
Download the official dataset from [here](https://github.com/marcelampc/d3net_depth_estimation/tree/master/dfd_datasets).

## Training
```
CUDA_VISIBLE_DEVICES=0 python train.py --nof-focus 2
```
