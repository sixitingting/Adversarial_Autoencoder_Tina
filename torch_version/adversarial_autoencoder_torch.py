import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from monai.utils import set_determinism
import os
import time

set_determinism(seed=42)


def form_results(z_dim, learning_rate):
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
    """
    date = time.strftime('%Y%m%d',time.localtime(time.time()))
    #把获取的时间转换成"年月日格式”
    results_path = '../TorchResults/Adversarial_Autoencoder'
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    folder_name = "/{0}_{1}_{2}_Adversarial_Autoencoder". format(date, z_dim, learning_rate)
    tensorboard_path = results_path + folder_name + '/Tensorboard'
    saved_model_path = results_path + folder_name + '/Saved_models/'
    log_path = results_path + folder_name + '/log'
    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
    return tensorboard_path, saved_model_path, log_path


class Encoder(nn.Module):
    def __init__(self, z):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, z)
        )

    def forward(self, x):
        x = x.view(-1, 784)
        z = self.encoder(x)

        return z


class Decoder(nn.Module):
    def __init__(self, z):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(z, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 784),
            nn.Sigmoid()    # 使用 sigmoid 是因为input的值范围也是[0,1]
        )

    def forward(self, x):
        x = self.decoder(x)
        out = x.view(-1, 1, 28, 28)

        return out


class Discriminator(nn.Module):
    # 这里输入的是 encoder output： latent space
    def __init__(self, z):
        super(Discriminator, self).__init__()

        self.dis = nn.Sequential(
            nn.Linear(z, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        return self.dis(x)


def train(train_dataloader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(2).to(device)
    decoder = Decoder(2).to(device)
    discriminator = Discriminator(2).to(device)

    ae_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    # 两个阶段，两个优化器。一个是重建阶段的优化 optimizer_ae， 一个是Gan的优化，optimizer_dis
    optimizer_dis = optim.Adam(discriminator.parameters(), lr=1e-3)
    optimizer_ae = optim.Adam([{'params': encoder.parameters()},
                               {'params': decoder.parameters()}], lr=1e-3)

    tensorboard_path, saved_model_path, log_path = form_results(2, 1e-3)
    writer = SummaryWriter(tensorboard_path)

    step = 0
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        discriminator.train()

        autoencoder_loss_epoch = 0.0
        discriminator_loss_epoch = 0.0
        generator_loss_epoch = 0.0

        for img, lab in train_dataloader:
            img = img.to(device)
            z_real = Variable(torch.randn(img.size(0), 2))
            # 返回一个由均值为0且方差为1的正态分布 # 2 是 latent space大小
            z_real = z_real.to(device)
            # ==========forward=========
            z = encoder(img)
            xhat = decoder(z)

            d_real = discriminator(z_real)  # real sampled gaussian
            d_fake = discriminator(z)  # fake created gaussian

            # ==========compute the loss and backpropagate=========
            encoder_decoder_loss = ae_loss(xhat, img)
            generator_loss = bce_loss(d_fake, target=torch.ones_like(d_fake))
            # encoder： d_fake值越接近1越好
            discriminator_loss = bce_loss(d_fake, target=torch.zeros_like(d_fake)) +\
                                 bce_loss(d_real, target=torch.ones_like(d_real))
            # discriminator： d_fake值越接近0越好, d_real值越接近1越好

            optimizer_dis.zero_grad()
            tot_loss = discriminator_loss + generator_loss
            tot_loss.backward(retain_graph=True)
            optimizer_dis.step()

            optimizer_ae.zero_grad()
            encoder_decoder_loss.backward(retain_graph=False)
            optimizer_ae.step()

            # ========METRICS===========
            autoencoder_loss_epoch += encoder_decoder_loss.item()
            discriminator_loss_epoch += discriminator_loss.item()
            generator_loss_epoch += generator_loss.item()

            writer.add_scalar('ae_loss', encoder_decoder_loss, step)
            writer.add_scalar('generator_loss', generator_loss, step)
            writer.add_scalar('discriminator_loss', discriminator_loss, step)

            step += 1

        epoch_autoencoder_loss = autoencoder_loss_epoch / len(train_dataloader)
        epoch_discriminator_loss = discriminator_loss_epoch / len(train_dataloader)
        epoch_generator_loss = generator_loss_epoch / len(train_dataloader)

        print("Autoencoder Loss: {}".format(epoch_autoencoder_loss))
        print("Discriminator Loss: {}".format(epoch_discriminator_loss))
        print("Generator Loss: {}".format(epoch_generator_loss))

        writer.add_images('train last batch images', img, epoch+1)
        writer.add_images('train reconstruction images', xhat, epoch+1)

        if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
            torch.save({
                'epoch': epoch + 1,
                'encoder': encoder.state_dict(),
            }, saved_model_path + f'/encoder_{epoch + 1}.pth')

            torch.save({
                'epoch': epoch + 1,
                'decoder': decoder.state_dict(),
            }, saved_model_path + f'/decoder_{epoch + 1}.pth')

            torch.save({
                'epoch': epoch + 1,
                'discriminator': discriminator.state_dict(),
            }, saved_model_path + f'/discriminator_{epoch + 1}.pth')

    return encoder, decoder, discriminator


if __name__ == '__main__':
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [transforms.ToTensor(), ])
    train_data = MNIST(root='./data', train=True, download=False, transform=transform)
    # 如果没有下载，使用download=True, 下载一次后，后面再运行代码无需下载
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128,
                                               shuffle=True, num_workers=2)
    test_data = MNIST(root='./data', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128,
                                              shuffle=False, num_workers=2)

    encoder, decoder, discriminator = train(train_loader, epochs=1000)