# GDR-DeblurGAN
[Paper Version]([https://arxiv.org/pdf/1711.07064.pdf](https://www.sciencedirect.com/science/article/pii/S1047320320301565))

Pytorch implementation of the paper Gradient-based conditional generative adversarial network for non-uniform
blind deblurring via DenseResNet.

Our network takes blurry image as an input and procude the corresponding sharp estimate, as in the example:
<img src="images/animation3.gif" width="400px"/> <img src="images/animation4.gif" width="400px"/>


The model we use is Conditional Wasserstein GAN with Gradient Penalty + Perceptual loss based on VGG-19 activations. Such architecture also gives good results on other image-to-image translation problems (super resolution, colorization, inpainting, dehazing etc.)

## How to run

### Prerequisites
- NVIDIA GPU + CUDA CuDNN (CPU untested, feedback appreciated)
- Pytorch

Note that during the inference you need to keep only Generator weights.

Put the weights into 
```bash
/.checkpoints/experiment_name
```
To test a model put your blurry images into a folder and run:
```bash
python test.py --dataroot /.path_to_your_data --model test --dataset_mode single --learn_residual
```


## Train

If you want to train the model on your data run the following command to create image pairs:
```bash
python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data
```
And then the following command to train the model

```bash
python train.py --dataroot /.path_to_your_data --learn_residual --resize_or_crop crop --fineSize CROP_SIZE (we used 256)
```



## Citation

If you find our code helpful in your research or work please cite our paper.

```
@article{GDR-DeblurGAN,
  title = {Gradient-based conditional generative adversarial network for non-uniform blind deblurring via DenseResNet},
  author = {Hongtian Zhao and Di Wu and Hang Su and Shibao Zheng and Jie Chen},
  journal = {Journal of Visual Communication and Image Representation},
  volume = {74},
  pages = {102921},
  year = {2021},
  issn = {1047-3203}
}
```

## Acknowledgments
Code borrows heavily from [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). The images were taken from GoPRO test dataset - [DeepDeblur](https://github.com/SeungjunNah/DeepDeblur_release)
[DeblurGAN](https://github.com/KupynOrest/DeblurGAN)


