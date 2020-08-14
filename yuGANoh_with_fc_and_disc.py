import torch.nn as nn
import torch.nn.functional as F
import torch


class Generator(nn.Module):
    def __init__(self,noise_channels,gan_features):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(noise_channels, gan_features * 8, (4, 3), 1, 0, bias=False),
            nn.BatchNorm2d(gan_features * 8),
            nn.LeakyReLU(0.2,True),
            # state size. (gan_features*8) x 4 x 3
            nn.ConvTranspose2d(gan_features * 8, gan_features * 4, 5, 3, 2, bias=False),
            nn.BatchNorm2d(gan_features * 4),
            nn.LeakyReLU(0.2,True),
            # state size. (gan_features*4) x 8 x 8
            nn.ConvTranspose2d( gan_features * 4, gan_features * 2, 5, 3, (3,2), bias=False),
            nn.BatchNorm2d(gan_features * 2),
            nn.LeakyReLU(0.2,True),
            # state size. (gan_features*2) x 16 x 16
            nn.ConvTranspose2d( gan_features * 2, gan_features * 2, 5, 3, (3,2), bias=False),
            nn.BatchNorm2d(gan_features * 2),
            nn.LeakyReLU(0.2,True),
            # state size. (gan_features) x 32 x 32
            nn.ConvTranspose2d( gan_features * 2, gan_features, 5, 3, (4,3), bias=False),
            nn.BatchNorm2d(gan_features),
            nn.LeakyReLU(0.2,True),
            nn.ConvTranspose2d( gan_features, 3, (4,5), 2, (3,2), bias=False),
            nn.Tanh()
            # state size. 3 x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
    
    
class Discriminator(nn.Module):
    def __init__(self, disc_features,num_features,similarity_features,batch_size):
        super(Discriminator,self).__init__()
        self.num_features = num_features
        self.T = torch.nn.Parameter(torch.randn(similarity_features,similarity_features,num_features*2),requires_grad=True)
        self.main = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(3, disc_features, 8, (4,3), 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (disc_features) x 32 x 32
            nn.Conv2d(disc_features, disc_features * 2, 5, 3, (1,2), bias=False),
            nn.BatchNorm2d(disc_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (disc_features*2) x 16 x 16
            nn.Conv2d(disc_features * 2, disc_features * 4, 5, 3, 2, bias=False),
            nn.BatchNorm2d(disc_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (disc_features*4) x 8 x 8
            nn.Conv2d(disc_features * 4, disc_features * 8, 7, 2, 1, bias=False),
            nn.BatchNorm2d(disc_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (disc_features*8) x 4 x 4
            nn.Conv2d(disc_features * 8, num_features*2, 4, 1, 0, bias=False)
        )
        #self.fc1 = nn.Linear(num_features*2,num_features,bias=False)
        self.fc1 = nn.Linear(num_features*2,num_features,bias=False)
        self.fc2 = nn.Linear(num_features+similarity_features,1,bias=False)

    def forward(self, input):
        features = self.main(input)
        features = features.squeeze()
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        M = torch.matmul(self.T,features.transpose(0,1)) #sf,sf,batch_size
        M = M.repeat(M.shape[2],1,1,1)
        M_i = M.transpose(0,-1)
        C = torch.exp(-torch.sum(torch.abs(M_i-M),dim=2))
        C = torch.sum(C,dim=2)
        C = torch.cat([C,self.fc1(features)],dim=1)
        label = torch.sigmoid(self.fc2(C))
        return features, label

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0,0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data,0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform(m.weight.data)