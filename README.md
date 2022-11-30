# W-Net
A source code for paper "W-Net: Structure and Texture Interaction for Image Inpainting"

<div align=center>
  <img width="750" src="https://github.com/Evergrow/W-Net/blob/main/figures/teaser.png/" hspace="10">
</div>

Compared to previous methods, our method produces perfect structure and symmetrical objects when repairing corrupted regions in the image. (a) The ground truth image with blue shadow as mask. (b) The result of GConv. (c) The result of MEDFE. (d) The result of our W-Net.

## Prerequisites
* Ubuntu 16.04
* Python 3
* NVIDIA GPU CUDA + cuDNN
* TensorFlow 1.12.0

## Usage
### Set up
* Clone this repo:
```
git clone https://github.com/Evergrow/W-Net.git
cd W-Net
```
* Setup environment: Install [TensorFlow](https://www.tensorflow.org/) and dependencies.
* Download datasets: We use [Places2](http://places2.csail.mit.edu/), [CelebA-HQ](https://github.com/switchablenorms/CelebAMask-HQ), and [Paris Street-View](https://github.com/pathak22/context-encoder) datasets. Some common inpainting datasets such as [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [ImageNet](http://www.image-net.org/) are also available.
* Collect masks: Please refer to [this script](https://github.com/Evergrow/GDN_Inpainting/blob/master/mask_processing.py) to process raw mask [QD-IMD](https://github.com/karfly/qd-imd) as the training mask. [Liu et al.](https://arxiv.org/abs/1804.07723) provides 12k [irregular masks](https://nv-adlr.github.io/publication/partialconv-inpainting) as the test mask. Note that the square mask is not a good choice for training our model, while the test mask is freestyle.

### Training
* Modify gpu id, dataset path, mask path, and checkpoint path in the [config file](https://github.com/Evergrow/GDN_Inpainting/blob/master/config.yml). Adjusting some other parameters if you like.
* Run ```python train.py``` and view training progress ```tensorboard --logdir [path to checkpoints]```

### Test
Choose the input image, mask and model to test:
```python
python test.py --image [input path] --mask [mask path] --output [output path] --checkpoint_dir [model path]
```
### Pretrained models
[Pretrained models](https://drive.google.com/drive/folders/1t0Nrd-daRDarMKc0B23cw4USgoYTJNxl?usp=sharing) are released for quick test. Download the models using Google Drive links and move them into your ./checkpoints directory.
