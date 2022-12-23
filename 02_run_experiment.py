from __future__ import print_function

import argparse
import os
from tempfile import gettempdir

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from clearml import Task, OutputModel
from clearml import Dataset

from experiment.args_experimenting import get_args
from experiment.model import Net


def train(model, epoch, train_loader, args, optimizer, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
            niter = epoch*len(train_loader)+batch_idx
            writer.add_scalar('Train/Loss', loss.data.item(), niter)


def test(model, test_loader, args, optimizer, writer):
    model.eval()
    test_loss = 0
    correct = 0
    for niter, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').data.item()  # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        pred = pred.eq(target.data).cpu().sum()
        writer.add_scalar('Test/Loss', pred, niter)
        correct += pred
        if niter % 100 == 0:
            writer.add_image('test', data[0, :, :, :], niter)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training params
    args = get_args()

    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    task = Task.init(project_name='Experimenting-MNIST', task_name='train pytorch model with ds', output_uri=True)  # noqa: F841
    writer = SummaryWriter('runs')
    writer.add_text('TEXT', 'This is some text', 0)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Grayscale(num_output_channels=1)])

    dataset_path = Dataset.get(
        dataset_id=args.datasets_id
    ).get_local_copy()
    print('dataset_path {}'.format(dataset_path))

    train_ds = datasets.ImageFolder(root=os.path.join(dataset_path, 'train'), transform=transform)
    test_ds = datasets.ImageFolder(root=os.path.join(dataset_path, 'test'), transform=transform)

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    
    model = Net()
    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(model, epoch, train_loader, args, optimizer, writer)

    # store in a way we can easily load into triton without having to have the model class
    torch.jit.script(model).save('serving_model.pt')
    OutputModel().update_weights('serving_model.pt')
    test(model, test_loader, args, optimizer, writer)


if __name__ == "__main__":
    main()
