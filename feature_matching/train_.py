import argparse
import wandb
import sys
from pathlib import Path

sys.path.append(str(Path('.').resolve()))

import torch.nn as nn
import torch.nn.functional as F
import torch

import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.utils.spectral_norm as spectral_norm

import torchvision.utils as vutils
import torch.optim as optim

from fid import CustomTensorDataset, load_model, get_activations, compute_fid
from feature_matching.models import Generator, Discriminator, weights_init

def train_log(key, loss, example_ct, epoch):
    loss = float(loss)
    # where the magic happens
    wandb.log({"epoch": epoch, key: loss}, step=example_ct)


def train(args, netD, optimizerD, netG, optimizerG, num_epochs, dataloader, criterion, device):
    
    iters = 0
    img_list = []
    G_losses = []
    D_losses = []

    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(64, args.nz, 1, 1)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    
    example_ct = 0
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):

            example_ct += len(data)
            # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ## Train with all-real batch
            netD.zero_grad()
            
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            
            intermediate_real, output = netD(real_cpu)
            output = output.view(-1)
            intermediate_real = torch.mean(intermediate_real, dim=0).view(-1).detach()
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            
            fake = netG(noise)
            label.fill_(fake_label)
            
            intermediate_, output = netD(fake.detach())
            output = output.view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            optimizerD.step()

            # Update G network: minimize -log(D(G(z))) + alpha*||E[f(x)] - E[f(G(z))]||^2
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            
            intermediate_fake, output_ = netD(fake)
            intermediate_fake, output_ = torch.mean(intermediate_fake, dim=0).view(-1), output_.view(-1)

            err, reg = criterion(output_, label), (intermediate_real - intermediate_fake).pow(2).sum()
            alpha = (0.2 * err / reg).detach().item()
            errG = err + alpha * reg
            errG.backward()
            D_G_z2 = output_.mean().item()

            # Update G
            optimizerG.step()
            
            # Output training stats
            if i % args.log_interval == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            train_log("G_losses", errG.item(), example_ct, epoch)
            train_log("D_losses", errD.item(), example_ct, epoch)
           
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(wandb.Image(
                    vutils.make_grid(fake, padding=2, normalize=True)))

            iters += 1
    
    wandb.log({"img_list": img_list})
        
def main():
    #wandb.init()
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST DCGAN with feature matching')
    parser.add_argument('--dataroot', type=str, default="data/mnist", metavar='dataroot',
                        help='input dataroot for training dataset (default: data/mnist)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--beta', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--nc', type=int, default=1, metavar='N',
                        help='Number of channels in the training images')
    parser.add_argument('--nz', type=int, default=50, metavar='N',
                        help='Size of z latent vector (i.e. size of generator input)')
    parser.add_argument('--ngf', type=int, default=32, metavar='N',
                        help='Size of feature maps in generator (default: 32)')
    parser.add_argument('--ndf', type=int, default=32, metavar='N',
                        help='Size of feature maps in discriminator (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    #wandb.config.update(args)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
        'test': transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]),
    }

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=args.dataroot, download=True, train=True,
                            transform=data_transforms['train']), 
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs,
    )
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=args.dataroot, download=True, train=False,
                            transform=data_transforms['test']), 
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs,
    )

    dcG = Generator(args.nz, args.ngf, args.nc).to(device)
    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    dcG.apply(weights_init)

    dcD = Discriminator(args.nc, args.ndf).to(device)
    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    dcD.apply(weights_init)
    
    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Setup Adam optimizers for both G and D
    opt_dcD = optim.Adam(dcD.parameters(), lr=args.lr, betas=(args.beta, 0.999))
    opt_dcG = optim.Adam(dcG.parameters(), lr=args.lr, betas=(args.beta, 0.999))
    
    #wandb.watch(dcD)
    #wandb.watch(dcG)

    #train(args, dcD, opt_dcD, dcG, opt_dcG, args.epochs, train_loader, criterion, device)

    dcG.eval()
    with torch.no_grad():
        fake_images = dcG(torch.randn(len(test_loader)*args.batch_size, args.nz, 1, 1, device=device))

    fake_loader = torch.utils.data.DataLoader(
        CustomTensorDataset(
            tensor=fake_images,
            transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )

    m = load_model(device)

    real_pred, fake_pred = get_activations(m, args.batch_size, test_loader, fake_loader)
    fid_score = compute_fid(real_pred, fake_pred)
    
    print(f"TOTAL FID:\t{fid_score:.2f}")
    #wandb.log({"fid_score": fid_score}, step=i)


if __name__ == '__main__':
    main()