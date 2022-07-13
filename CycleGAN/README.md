# CycleGAN in PyTorch

[CycleGAN Paper](https://arxiv.org/pdf/1703.10593.pdf) 

## Talks and Course
CycleGAN slides: [pptx](http://efrosgans.eecs.berkeley.edu/CVPR18_slides/CycleGAN.pptx) | [pdf](http://efrosgans.eecs.berkeley.edu/CVPR18_slides/CycleGAN.pdf)

CycleGAN course assignment [code](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip) and [handout](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-handout.pdf) designed by Prof. [Roger Grosse](http://www.cs.toronto.edu/~rgrosse/) for [CSC321](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/) "Intro to Neural Networks and Machine Learning" at University of Toronto. Please contact the instructor if you would like to adopt it in your course.

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/SelendisErised/starry-nAIght.git
cd CycleGAN
```

- Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.

### CycleGAN train/test
- Download a CycleGAN dataset (e.g. monet2photo, vangogh2photo, cezanne2photo, ukiyoe2photo):
```bash
bash ./datasets/download_cyclegan_dataset.sh monet2photo
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- To log training progress and test images to W&B dashboard, set the `--use_wandb` flag with train and test script
- Train a model:
```bash
#!./scripts/train_cyclegan.sh
python train.py --dataroot ./datasets/monet2photo --name monet2photo_cyclegan --model cycle_gan
```
To see more intermediate results, check out `./checkpoints/monet2photo/web/index.html`.
- Test the model:
```bash
#!./scripts/test_cyclegan.sh
python test.py --dataroot ./datasets/monet2photo --name monet2photo_cyclegan --model cycle_gan
```
- The test results will be saved to a html file here: `./results/cyclegan_cyclegan/latest_test/index.html`.

### Apply a pre-trained CycleGAN model
- You can download a pretrained model (e.g. style_monet, style_vangogh, style_cezanne, style_ukiyoe, monet2photo) with the following script:
```bash
bash ./scripts/download_cyclegan_model.sh style_monet
```
- The pretrained model is saved at `./checkpoints/{name}_pretrained/latest_net_G.pth`.
- If you want to use the test set in the provided dataset to test the model, you need to download the corresponding dataset, for example:
```bash
bash ./datasets/download_cyclegan_dataset.sh monet2photo
```

- Then generate the results using
```bash
python test.py --dataroot datasets/monet2photo/testA --name style_monet_pretrained --model test --no_dropout
```
- The option `--model test` is used for generating results of CycleGAN only for one side. This option will automatically set `--dataset_mode single`, which only loads the images from one set. On the contrary, using `--model cycle_gan` requires loading and generating results in both directions, which is sometimes unnecessary. The results will be saved at `./results/`. Use `--results_dir {directory_path_to_save_result}` to specify the results directory.

