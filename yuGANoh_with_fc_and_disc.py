import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.functional import pad
import torch
import os
from PIL import Image


class Generator(nn.Module):
    """
    Generator Network, DCGAN-like architecture
    with LeakyRELU and custom kernel shapes.
    """

    def __init__(self, noise_channels, gan_features):
        """
        Initializes network. Input parameters control
        input noise size and network width.
        :param noise_channels: input noise dimension
        :param gan_features: conv channel width factor.
        """
        super(Generator, self).__init__()
        self.initial_upscaling = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(noise_channels, gan_features * 8,
                               (4, 3), 1, 0, bias=False),
            nn.BatchNorm2d(gan_features * 8),
            nn.LeakyReLU(0.2, True),
            # state size. (gan_features*8) x 4 x 3
            nn.ConvTranspose2d(gan_features * 8, gan_features * 4,
                               5, 3, 2, bias=False),
            nn.BatchNorm2d(gan_features * 4),
            nn.LeakyReLU(0.2, True),
        )

        self.card = nn.Sequential(
            # state size. (gan_features*4) x 8 x 8
            nn.ConvTranspose2d(gan_features * 4, gan_features * 2,
                               5, 3, (3, 2), bias=False),
            nn.BatchNorm2d(gan_features * 2),
            nn.LeakyReLU(0.2, True),
            # state size. (gan_features*2) x 16 x 16
            nn.ConvTranspose2d(gan_features * 2, gan_features * 2,
                               5, 3, (3, 2), bias=False),
            nn.BatchNorm2d(gan_features * 2),
            nn.LeakyReLU(0.2, True),
            # state size. (gan_features) x 32 x 32
            nn.ConvTranspose2d(gan_features * 2, gan_features,
                               5, 3, (4, 3), bias=False),
            nn.BatchNorm2d(gan_features),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(gan_features, 3, (4, 5),
                               2, (3, 2), bias=False),
            nn.Tanh()
            # state size. 3 x 64 x 64
        )
        
        # need 228x248 output.
        self.centerbox = nn.Sequential(
            # state size. (gan_features*8) x 14 x 10
            nn.ConvTranspose2d(gan_features * 4, gan_features * 4,
                               (5, 8), (3, 4), 0, bias=False, dilation=2),
            nn.BatchNorm2d(gan_features * 4),
            nn.LeakyReLU(0.2, True),
            # 36 x 39
            nn.ConvTranspose2d(gan_features * 4, gan_features * 2,
                               6, 3, 0, bias=False),
            nn.BatchNorm2d(gan_features * 2),
            nn.LeakyReLU(0.2, True),
            # 111 x 120
            nn.ConvTranspose2d(gan_features * 2, gan_features,
                               (4, 5), 2, 0, bias=False),
            nn.BatchNorm2d(gan_features),
            nn.LeakyReLU(0.2, True),
            # 224 x 243
            nn.ConvTranspose2d(gan_features, 3, (5, 6),
                               1, 0, bias=False),
            nn.Tanh()
            )

    def forward(self, noise):
        """
        Pushes noise through the network.
        The generator only needs to run through the main sequential.
        :param noise: Tensor of shape (batch_size,latent_size,1,1)
        """
        y1 = 80
        y2 = 80+228
        x1 = 40
        x2 = 40 + 248
        s = (428,321)
        padding = (x1,s[1]-x2,y1,s[0]-y2)
        
        out = self.initial_upscaling(noise)
        out1 = self.card(out)
        out2 = self.centerbox(out)
        out2 = pad(out2, padding, mode='constant', value=0)
        return out1 + out2


class Discriminator(nn.Module):
    """
    Discriminator Network, based of DCGAN, but with leakyRELU
    and a few differences.
    LeakyReLU is used, along with minibatch discrimination.
    Minibatch feature scaling is handled by similarity_features.
    """

    def __init__(self, disc_features, num_features,
                 similarity_features):
        """
        :param disc_features: number of gan features, scales network width
        :param num_features: number of features to be given to FC layer.
        :param similarity_features: number of similarity features
               for minibatch discrimination
        """
        super(Discriminator, self).__init__()
        self.num_features = num_features
        # T is similarity matrix. Needs to be learnable parameter.
        self.T = torch.nn.Parameter(
            torch.randn(
                similarity_features,
                similarity_features,
                num_features * 2),
            requires_grad=True)
        self.main = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(3, disc_features, 8, (4, 3), 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (disc_features) x 32 x 32
            nn.Conv2d(disc_features, disc_features * 2,
                      5, 3, (1, 2), bias=False),
            nn.BatchNorm2d(disc_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (disc_features*2) x 16 x 16
            nn.Conv2d(disc_features * 2, disc_features * 4,
                      5, 3, 2, bias=False),
            nn.BatchNorm2d(disc_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (disc_features*4) x 8 x 8
            nn.Conv2d(disc_features * 4, disc_features * 8,
                      7, 2, 1, bias=False),
            nn.BatchNorm2d(disc_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (disc_features*8) x 4 x 4
            nn.Conv2d(disc_features * 8, num_features * 2,
                      4, 1, 0, bias=False)
        )
        self.fc1 = nn.Linear(num_features * 2, num_features, bias=False)
        self.fc2 = nn.Linear(num_features + similarity_features, 1, bias=False)

    def forward(self, images):
        """
        Discriminator forward, minibatch discrimination implemented.
        :param images: Input image tensor. (batch_size,c,h,w)
        :return: label for CE loss and batch features for mean loss.
        """
        features = self.main(images)
        features = features.squeeze()
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        # similarity matrix
        # sf,sf,batch_size
        similarity = torch.matmul(self.T, features.transpose(0, 1))
        similarity = similarity.repeat(similarity.shape[2], 1, 1, 1)
        similarity_t = similarity.transpose(0, -1)
        # similarity distance scores
        distance = torch.exp(-torch.sum(
            torch.abs(similarity_t - similarity), dim=2))
        distance = torch.sum(distance, dim=2)
        distance = torch.cat([distance, self.fc1(features)], dim=1)
        # output label (0 or 1)
        label = torch.sigmoid(self.fc2(distance))
        return features, label
        
        
class YgoCards(Dataset):
    """
    GO Data Loader
    Reads in card images as PIL images
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = [el for el in os.listdir(self.root_dir) if '.jpg' in el]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir,
                                self.file_list[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image


def init_weights(layer):
    """
    Weight initializers.
    Pass in a network layer to be initialized.
    :param layer: reference to a layer object
    :return: None
    """
    class_name = layer.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(layer.weight.data, 0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)
    elif class_name.find('Linear') != -1:
        nn.init.xavier_uniform(layer.weight.data)
