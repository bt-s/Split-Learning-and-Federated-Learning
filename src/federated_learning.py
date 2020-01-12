#!/usr/bin/python3

"""federated.py Contains an implementation of federated learning with ten
                workers applied to the Fashion MNIST data set for image
                classification using a slightly modified version of the LeNet5
                CNN architecture.

For the ID2223 Scalable Machine Learning course at KTH Royal Institute of
Technology"""

__author__ = "Xenia Ioannidou and Bas Straathof"


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from sys import argv
from time import time
from argparse import ArgumentParser, Namespace

# The Pysyft Federated Learning library
import syft as sy

from models import LeNetComplete
from plotting import generate_simple_plot


def parse_args() -> Namespace:
    """Parses CL arguments

    Returns:
        Namespace object containing all arguments"""
    parser = ArgumentParser()

    parser.add_argument("-bs", "--batch_size", type=int, default=64)
    parser.add_argument("-tbs", "--test_batch_size", type=int, default=1000)
    parser.add_argument("-ls", "--log_steps", type=int, default=50)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("-e", "--epochs", type=int, default=25)

    return parser.parse_args(argv[1:])


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    """Train the model

    Args:
        args: Hyper-paramters for training
        model: The model to be trained
        device: Training device (e.g. cpu/gpu)
        train_loader: Federated data loader for training
        optimizer: Training optimizer (e.g. SGD)
        epoch: Current epoch

    Returns:
        training_time: Time it took for one epoch of training
    """
    start = time()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Ensure that thet data is send to the right worker
        model.send(data.location)

        # Write the data and targets to the compute device
        data, target = data.to(device), target.to(device)

        # Reset the optimizers gradients to zero
        optimizer.zero_grad()

        # Pass the data to the model
        output = model(data)

        # Compute the loss
        loss = F.nll_loss(output, target)

        # Execute the backpropagation step
        loss.backward()

        # Apply the optimizer
        optimizer.step()

        # Retrieve the model (PySyft feature)
        model.get()

        if batch_idx % args.log_steps == 0:
            # Retrieve the loss (PySyft feature)
            loss = loss.get()
            print(' {} - Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                data.location.id, epoch, batch_idx * args.batch_size,
                len(train_loader) * args.batch_size,
                100. * batch_idx / len(train_loader), loss.item()))

    end = time()
    training_time = end-start

    return training_time


def test(args, model, device, test_loader, epoch):
    """Evaluate the model

    Args:
        args: Hyper-parameters for testing
        model: The model to be evaluated
        device: Training device (e.g. cpu/gpu)
        test_loader: Federated data loader for testing
        epoch: Current epoch

    Returns:
        test_loss: Test loss for current epoch
        test_acc: Test accuracy for current epoch
    """
    model.eval()
    test_loss, correct = 0.0, 0

    with torch.no_grad():
        for data, target in test_loader:
            data_target = data.to(device), target.to(device)

            # Feed-forward data through the model
            output = model(data)

            # Sum up the batch loss
            test_loss += F.nll_loss(output, target, reduction="sum").item()

            # Retrieve the index of the maximum log-probability
            pred = output.argmax(1, keepdim=True)

            # Get the number of correct classifications
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Average the test_loss
    test_loss /= len(test_loader.dataset)

    # Test accuracy
    test_acc = correct / len(test_loader.dataset)

    print("\nTest set - Epoch: {} - Loss: {:.4f}, Acc: {:.2f}%\n".format(
        epoch, test_loss, 100 * test_acc))

    return test_loss, test_acc


if __name__ == "__main__":
    args = parse_args()

    # Pysyft needs to be hooked to PyTorch to enable its features
    hook = sy.TorchHook(torch)

    # Define the workers
    alfa    = sy.VirtualWorker(hook, id="alfa")
    bravo   = sy.VirtualWorker(hook, id="bravo")
    charlie = sy.VirtualWorker(hook, id="charlie")
    delta   = sy.VirtualWorker(hook, id="delta")
    echo    = sy.VirtualWorker(hook, id="echo")
    foxtrot = sy.VirtualWorker(hook, id="foxtrot")
    golf    = sy.VirtualWorker(hook, id="golf")
    hotel   = sy.VirtualWorker(hook, id="hotel")
    india   = sy.VirtualWorker(hook, id="india")
    juliet  = sy.VirtualWorker(hook, id="juliet")

    workers = (alfa, bravo, charlie, delta, echo, foxtrot, golf, hotel, india,
            juliet)

    # Check if a GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    kwargs = {'num_workers': 1, 'pin_memory': True} if device=="cuda" else {}

    # Specify required data transformation
    data_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Federated learning needs a special train loader to distribute the data
    # over the various workers
    print("Loading data...")

    train_loader = sy.FederatedDataLoader(datasets.FashionMNIST('data',
        train=True, download=True,
        transform=data_transformer).federate(workers),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=False,
        transform=data_transformer), batch_size=args.test_batch_size,
        shuffle=True, **kwargs)

    print("Data loaded...")

    # Instantiate the CNN model
    model = LeNetComplete().to(device)

    # Load the optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    # Train and evaluate for a number of epochs
    total_train_time, test_losses, test_accs = 0.0, [], []
    for epoch in range(1, args.epochs + 1):
        train_time = train(args, model, device, train_loader, optimizer, epoch)
        total_train_time += train_time

        test_loss, test_acc =  test(args, model, device, test_loader, epoch)
        test_losses.append(test_loss)
        test_accs.append(test_acc)


    # Create validation loss and accuracy plots
    epoch_list = list(range(1, args.epochs+1))
    generate_simple_plot(epoch_list, test_losses,
            "Test loss (Federated Learning)", "epoch", "loss", [0.2, 0.9],
            save=True, fname="test_loss_fl.pdf")
    generate_simple_plot(epoch_list, test_accs,
            "Test accuracy (Federated Learning)", "epoch", "accuracy",
            [0.5, 1.0], save=True, fname="test_acc_fl.pdf")

    print("Total training time: {:.2f}s".format(total_train_time))
    print("Final test accuracy: {:.4f}".format(test_acc))
    print("Final test loss: {:.4f}".format(test_loss))

