import torch.nn as nn
import torch
import torch.optim as optim

# Hyper-parameters & Variables setting
num_epoch = 200
batch_size = 32
lr = 1e-4
num_channel = 1
dir_name = "CGAN_results"

noise_size = 100

img_size = 32*32
Class_num = 3

device = torch.device('cuda:0' if torch.cuda.is_available() else 'CPU')
print(device)

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()

    self.linear1 = nn.Linear(img_size + Class_num, 1024)
    self.linear2 = nn.Linear(1024, 512)
    self.linear3 = nn.Linear(512, 256)
    self.linear4 = nn.Linear(256, 3)
    self.Leaky_relu = nn.LeakyReLU(0.2)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.Leaky_relu(self.linear1(x))
    x = self.Leaky_relu(self.linear2(x))
    x = self.Leaky_relu(self.linear3(x))
    x = self.linear4(x)
    x = self.sigmoid(x)
    return x


class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()

    self.linear1 = nn.Linear(img_size + Class_num, 1024)
    self.linear2 = nn.Linear(1024, 512)
    self.linear3 = nn.Linear(512, 256)
    self.linear4 = nn.Linear(256, img_size)
    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()

  def forwar(self, x):
    x = self.relu(self.linear1(x))
    x = self.relu(self.linear2(x))
    x = self.relu(self.linear3(x))
    x = self.linear4(x)
    x = self.tanh(x)
    return x


Gan_G = Generator()
Gan_D = Discriminator()

Gan_G = Gan_G.to(device)
Gan_D = Gan_D.to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(Gan_G.parameters(), lr=lr)
optimizer_D = optim.Adam(Gan_D.parameters(), lr=lr)