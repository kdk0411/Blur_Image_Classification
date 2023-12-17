import os
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torchvision.utils import save_image
from PIL import Image, ImageFont, ImageDraw
from GanModel import Discriminator, Generator
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)



def Train(data_loader):
    lr = 1e-4
    noise_size = 100
    batch_size = 32
    dir_name = 'Test_data_set'

    Gan_G = Generator()
    Gan_D = Discriminator()

    Gan_G = Gan_G.to(device)
    Gan_D = Gan_D.to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(Gan_G.parameters(), lr=lr)
    optimizer_D = optim.Adam(Gan_D.parameters(), lr=lr)

    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=10, eta_min=0)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=10, eta_min=0)

    result = {"train_loss" : [], "val_loss" : [], "val_acc" : []}

    train_loader = data_loader["train_loader"]
    val_loader = data_loader["val_loader"]

    epoch = 20
    for epoch in range(epoch):
        for i, (images, label) in enumerate(train_loader):

            real_label = torch.full((batch_size, 1), 1, dtype=torch.float32).to(device)
            fake_label = torch.full((batch_size, 1), 0, dtype=torch.float32).to(device)

            real_images = images.reshape(batch_size, -1).to(device)

            label_encoded = F.one_hot(label, num_classes=3).to(device)

            real_images_concat = torch.cat((real_images, label_encoded), 1)


            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            z = torch.randn(batch_size, noise_size).to(device)

            z_concat = torch.cat((z, label_encoded), 1)
            fake_images = Gan_G(z_concat)
            fake_images_concat = torch.cat((fake_images, label_encoded), 1)


            g_loss = criterion(Gan_D(fake_images_concat), real_label)

            g_loss.backward()
            optimizer_G.step()


            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            z = torch.randn(batch_size, noise_size).to(device)


            z_concat = torch.cat((z, label_encoded), 1)
            fake_images = Gan_G(z_concat)
            fake_images_concat = torch.cat((fake_images, label_encoded), 1)

            fake_loss = criterion(Gan_D(fake_images_concat), fake_label)
            real_loss = criterion(Gan_D(real_images_concat), real_label)
            d_loss = (fake_loss + real_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            d_performance = Gan_D(real_images_concat).mean()
            g_performance = Gan_D(fake_images_concat).mean()

            if (i + 1) % 150 == 0:
                print("Epoch [ {}/{} ]  Step [ {}/{} ]  d_loss : {:.5f}  g_loss : {:.5f}"
                      .format(epoch + 1, epoch, i + 1, len(train_loader), d_loss.item(), g_loss.item()))

        print(" Epock {}'s discriminator performance : {:.2f}  generator performance : {:.2f}"
              .format(epoch + 1, d_performance, g_performance))
        with torch.no_grad():
            samples = fake_images.reshape(batch_size, 1, 28, 28)
            save_image(samples, os.path.join(dir_name, 'CGAN_fake_samples{}.png'.format(epoch + 1)))
            # print("label of 'CGAN_fake_samples{}.png' is {}".format(epoch + 1, label))

            fake_sample_image = Image.open("{}/CGAN_fake_samples{}.png".format(dir_name, epoch + 1))
            font = ImageFont.truetype("arial.ttf", 17)

            label = label.tolist()
            label = label[:10]
            label = [str(l) for l in label]

            label_text = ", ".join(label)
            label_text = "Conditional GAN -\n" \
                         "first 10 labels in this image :\n" + label_text

            image_edit = ImageDraw.Draw(fake_sample_image)
            image_edit.multiline_text(xy=(15, 300),
                                      text=label_text,
                                      fill=(0, 255, 255),
                                      font=font,
                                      stroke_width=4,
                                      stroke_fill=(0, 0, 0))
            fake_sample_image.save("{}/CGAN_fake_samples{}.png".format(dir_name, epoch + 1))


# For checking CGAN's validity in final step
def check_condition(_generator):
    test_image = torch.empty(0).to(device)

    for i in range(10):
        test_label = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        test_label_encoded = F.one_hot(test_label, num_classes=10).to(device)

        # create noise(latent vector) 'z'
        _z = torch.randn(10, 100).to(device)
        _z_concat = torch.cat((_z, test_label_encoded), 1)

        test_image = torch.cat((test_image, _generator(_z_concat)), 0)

    _result = test_image.reshape(100, 1, 28, 28)
    save_image(_result, os.path.join('Test_data_set', 'CGAN_test_result.png'), nrow=10)






# with torch.no_grad():
#     check_condition(generator)