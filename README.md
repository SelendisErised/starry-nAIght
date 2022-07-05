# starry-nAIght

Team Member: Zhuoyu Feng, Jinxuan Tang, Katie Jiang, Isabella Lu, Noah Hung

## Project Overview:

AI generated art can take numerous minutes to generate, primarily due to online traffic. Other AI art generators require users to pay for the entirety of the functionality or bombard the user with advertisements. Our goal is to provide users with a free, easy access art generator that’ll apply the styles of famous artists such as Picasso, Monet, and Van Gogh to their uploaded images. Methods that we used to achieve this goal includes: Style Transfer and GAN.

## Style Transfer

### Methodology

This method is first published in [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576). The style transfer algorithm used convolutional neural networks to extract the “style” from one image and apply it to another. To perform the style transfer, two images are fed through this VGG-19 network: the “content” and the “style”. A third image is created which tries to minimize the difference between the intermediate layers of the VGG-19 when both images are fed through. It provides a relatively controllable way to generate art. 

### Example

Here we provide a brief example of applying Edvard Munch's [**the Scream**](style_transfer/images/the_scream.jpg) to a photo of [Sipsey river bridge](style_transfer/images/the_scream.jpg).

![](style_transfer/results/result.png)
