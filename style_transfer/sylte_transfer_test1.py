from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import requests
from torchvision import transforms, models


def im_convert(tensor):
    """
    Display a tensor as an image
    :param tensor: input tensor
    :return: ndarray image
    """
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


def load_image(img_path, max_size=400, shape=None):
    """
    Load in and transform an image to <= 400 pixels in the x-y dims
    :param img_path: path to the image
    :param max_size: maximum size of an image
    :param shape: resize the image
    :return: image
    """
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    if shape is not None:
        size = shape
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image


def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """
    # Need the layers for the content and style representations of an image
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


def gram_matrix(tensor):
    """
    Calculate the Gram Matrix of a given tensor https://en.wikipedia.org/wiki/Gramian_matrix
    :param tensor:
    :return: gram matrix
    """
    b, d, h, w = tensor.size()
    tensor = tensor.view(b * d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


class StyleTransfer:
    def __init__(self, content_path, style_path):
        self.target = None
        self.style_grams = None
        self.style_features = None
        self.content_features = None
        self.vgg = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.style_weights = {'conv1_1': 1.,
                              'conv2_1': 0.75,
                              'conv3_1': 0.2,
                              'conv4_1': 0.2,
                              'conv5_1': 0.2}
        self.content_weight = 1  # alpha
        self.style_weight = 1e6  # beta
        self.lr = 0.003
        self.epoch = 2000
        # self.show_freq = 400
        self.content = load_image(content_path).to(self.device)
        # Resize style to match content
        self.style = load_image(style_path, shape=self.content.shape[-2:]).to(self.device)

    def model_loading(self):
        """
        Load pretrained vgg19 model and freeze all parameters
        """
        self.vgg = models.vgg19(pretrained=True).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)
        self.vgg.to(self.device)

    def feature_extraction(self):
        """
        Feature extraction and gram matrix calculation for each layer in style representation
        """
        self.content_features = get_features(self.content, self.vgg)
        self.style_features = get_features(self.style, self.vgg)
        self.style_grams = {layer: gram_matrix(self.style_features[layer]) for layer in self.style_features}
        self.target = self.content.clone().requires_grad_(True).to(self.device)

    def generate(self):
        """
        Training process (target generate)
        """
        self.model_loading()
        self.feature_extraction()
        optimizer = optim.Adam([self.target], lr=self.lr)
        for time in range(1, self.epoch + 1):
            target_features = get_features(self.target, self.vgg)
            content_loss = torch.mean((target_features['conv4_2'] - self.content_features['conv4_2']) ** 2)
            style_loss = 0
            for layer in self.style_weights:
                # get the "target" style representation for the layer
                target_feature = target_features[layer]
                target_gram = gram_matrix(target_feature)
                _, d, h, w = target_feature.shape
                # get the "style" style representation
                style_gram = self.style_grams[layer]
                # the style loss for one layer, weighted appropriately
                layer_style_loss = self.style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
                # add to the style loss
                style_loss += layer_style_loss / (d * h * w)
            total_loss = self.content_weight * content_loss + self.style_weight * style_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # if time % self.show_freq == 0:
            #     print('Total loss: ', total_loss.item())
            #     plt.imshow(im_convert(self.target))
            #     plt.show()


if __name__ == '__main__':
    content_img = 'images/sipsey_river_bridge.jpg'
    style_img = 'images/the_scream.jpg'
    style_transfer = StyleTransfer(content_img, style_img)
    style_transfer.generate()
    plt.axis('off')
    plt.imshow(im_convert(style_transfer.target))
    plt.savefig('results/result.png', bbox_inches='tight', pad_inches=0.0)
    plt.show()
