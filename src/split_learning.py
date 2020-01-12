#!/usr/bin/python3

"""split_learning.py Contains an implementation of split learning using a
                     slighlty modified version of the LeNet5 CNN architecture.

                     Split learning is here implemented for multiple workers and
                     one central server, using the Message Passing Interface
                     (MPI). This implementation has been inspired by the
                     private repository of Abishek Shing as part of the work of
                     Vepakomma et al. on split learning, please see:

                     Praneeth Vepakomma, Otkrist Gupta, Tristan Swedish, and
                        Ramesh Raskar. Split learning for health: Distributed
                        deep learning without sharing raw patient data.
                        arXiv preprint arXiv:1812.00564 , 2018

For the ID2223 Scalable Machine Learning course at KTH Royal Institute of
Technology"""

__author__ = "Xenia Ioannidou and Bas Straathof"


from mpi4py import MPI

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import os
from time import time
import pickle
import itertools
import numpy as np
from sys import argv
from argparse import ArgumentParser, Namespace

from models import LeNetClientNetwork, LeNetServerNetwork
from plotting import generate_simple_plot


def parse_args() -> Namespace:
    """Parses CL arguments

    Returns:
        Namespace object containing all arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-bs", "--batch_size", type=int, default=64)
    parser.add_argument("-nb", "--num_batches", type=int, default=938)
    parser.add_argument("-tbs", "--test_batch_size", type=int, default=1000)
    parser.add_argument("-ls", "--log_steps", type=int, default=50)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-p", "--plot", type=bool, default=True)

    return parser.parse_args(argv[1:])

args = parse_args()

# Define the communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

SERVER = 0
MAX_RANK = comm.Get_size() - 1

if MAX_RANK > 10:
    print("Aborting script since more than 10 workers were specified...")
    exit()

# Set a random seed
torch.manual_seed(0)

# Check if a GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Check if the current process is a worker or the server
if rank >= 1:
    worker, server = True, False
else:
    server, worker = True, False

worker_map = {1: "alfa", 2: "bravo", 3: "charlie", 4: "delta", 5: "echo",
        6: "foxtrot", 7: "golf", 8: "hotel", 9: "india", 10: "juliet"}


# Define the procedure for the wworkers
if worker:
    epoch = 1
    active_worker = rank
    worker_left = rank - 1
    worker_right = rank + 1

    # The server is not a worker
    if rank == 1:
        worker_left = MAX_RANK

    # Make sure that worker rank 1 is the first to start
    elif rank == MAX_RANK:
      worker_right = 1
      comm.send("you_can_start", dest=worker_right)

    # Define how the data should be transformed
    data_transformer = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

    file_path = "./data/FashionMNIST/processed/fashion_mnist.pkl"
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            train_set = pickle.load(f)
    else:
        # Let the first worker download the data
        if rank == 1:
            # Download the Fashion-MNIST training data set
            train_set = torchvision.datasets.FashionMNIST(root='./data',
                    train=True, download=True, transform=data_transformer)
            train_loader = torch.utils.data.DataLoader(train_set,
                    batch_size=args.batch_size, shuffle=True)

            # Since each worker needs its own private data set it is
            # impractical to  use the DataLoader object.
            train_set = []
            for i, l in train_loader:
                train_set.append((i, l))

            with open(file_path, 'wb') as f:
                pickle.dump(train_set, f, protocol=pickle.HIGHEST_PROTOCOL)

            print("Data downloaded. Please run the script again.")
            comm.send("data_downloaded", dest=SERVER)
        exit()

    # Make sure that each worker has its own private training data
    start = int(np.floor(len(train_set)/MAX_RANK*(rank-1)))
    stop = int(np.floor(len(train_set)/MAX_RANK*(rank)))
    train_set = train_set[start:stop]

    # Obtain the Fashion-MNIST test data set
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
        download=True, transform=data_transformer)
    test_loader = torch.utils.data.DataLoader(testset,
            batch_size=args.test_batch_size, shuffle=False)

    # Instantiate the client network
    model = LeNetClientNetwork().to(device)

    # Use Stochastic Gradient Descent as optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
            momentum=0.9, weight_decay=5e-4)

    while True:
        # Wait to receive a messag from the other worker
        msg = comm.recv(source=worker_left)

        if msg == "you_can_start":
            if rank == 1:
                print(f"\nStart epoch {epoch}:")

            start = time()
            for batch_idx, (inputs, labels) in enumerate(train_set):
                # Write the input data and lables to the current device
                inputs, labels = inputs.to(device), labels.to(device)

                # Reset the gradients to zero in the optimizer
                optimizer.zero_grad()

                # Optain the tensor output of the split layer
                split_layer_tensor = model(inputs)

                # Send tensor of the split layer and the labels to the server
                comm.send(["tensor_and_labels", [split_layer_tensor, labels]],
                    dest=SERVER)

                # Receive the gradients for backpropgation from the server
                grads = comm.recv(source=SERVER)

                # Apply the gradients to the split layer tensor
                split_layer_tensor.backward(grads)

                # Apply the optimizer
                optimizer.step()

            # Garbage collection
            del split_layer_tensor, grads, inputs, labels
            torch.cuda.empty_cache()
            end = time()

            # Send training time to server
            comm.send(["time", end-start], dest=SERVER)

            # Only let the last worker evaluate on the test set
            if rank == MAX_RANK:
                # Tell the server to start validating
                comm.send("validation", dest=SERVER)

                for batch_idx, (inputs, labels) in enumerate(test_loader):
                    # Write the input data and lables to the current device
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Optain the tensor output of the split layer
                    split_layer_tensor = model(inputs)

                    # Send tensor of the split layer and the labels to the server
                    comm.send(["tensor_and_labels", [split_layer_tensor, labels]],
                        dest=SERVER)

                # Garbage collection
                del split_layer_tensor, inputs, labels
                torch.cuda.empty_cache()

            # Signal to the other worker that it can start training
            comm.send("you_can_start", dest=worker_right)

            if epoch == args.epochs:
                msg="training_complete" if rank == MAX_RANK else "worker_done"
                comm.send(msg, dest=SERVER)
                exit()
            else:
                # Let the server know that the current epoch has finished
                msg="epoch_done" if rank == MAX_RANK else "worker_done"
                comm.send(msg, dest=SERVER)

            epoch += 1

# Define the procedure for the workers
elif server:
    # Instantiate the server network
    model = LeNetServerNetwork()
    model = model.to(device)

    # Define the loss criterion
    loss_crit = nn.CrossEntropyLoss()

    # Use Stochastic Gradient Descent with momentum and weight decay
    # as the optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
            momentum=0.9, weight_decay=5e-4)

    total_training_time = 0.0

    epoch, step, batch_idx = 1, 0, 0

    active_worker, phase = 1, "train"

    val_loss, val_losses, val_accs = 0.0, [], []
    total_n_labels_train, total_n_labels_test = 0, 0
    correct_train, correct_test = 0, 0

    while(True):
        # Wait for the message of the active worker
        msg = comm.recv(source=active_worker)

        if msg[0] == "tensor_and_labels":
            if phase == "train":
                # Reset the gradients to zero in the optimizer
                optimizer.zero_grad()

            # Dereference the input tensor and corresponding labels from the
            # message
            input_tensor, labels = msg[1]

            # Obtain logits from forward pass through server model
            logits = model(input_tensor)

            # Obtain the predictions
            _, predictions = logits.max(1)

            # Compute the loss
            loss = loss_crit(logits, labels)

            if phase == "train":
                # Add current label count to the total number of labels
                total_n_labels_train += len(labels)

                # Identify how many of the predictions were correct
                correct_train += predictions.eq(labels).sum().item()

                # Back-propagate the loss
                loss.backward()

                # Apply the optimizer
                optimizer.step()

                # Send gradients back to the active worker
                comm.send(input_tensor.grad, dest=active_worker)

                # Increment batch index
                batch_idx += 1

                if batch_idx % args.log_steps == 0:
                    # Calculate the training accuracy
                    acc = correct_train / total_n_labels_train

                    print('{} - Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                        worker_map[active_worker], epoch, int((
                        args.num_batches * args.batch_size) / MAX_RANK * (
                        active_worker-1)) + batch_idx * args.batch_size,
                        args.num_batches * args.batch_size, 100. * (((
                        args.num_batches / MAX_RANK * (active_worker-1)) + \
                        batch_idx) / args.num_batches), loss.item()))

            if phase == "validation":
                step += 1
                # Add current label count to the total number of labels
                total_n_labels_test += len(labels)

                # Identify how many of the predictions were correct
                correct_test += predictions.eq(labels).sum().item()
                val_loss += loss.item()

        elif msg[0] == "time":
            total_training_time += msg[1]

        elif msg == "worker_done":
            if active_worker == MAX_RANK:
                epoch += 1

            # Change worker and phase
            active_worker = (active_worker % MAX_RANK) + 1
            phase = "train"

            # Reset variables
            total_n_labels_train, correct_train, batch_idx = 0, 0 ,0

        elif msg == "epoch_done" or msg == "training_complete":
            # Update the validation loss
            val_loss /= step
            val_losses.append(val_loss)

            # Compute the validation accuracy
            acc = correct_test / total_n_labels_test
            val_accs.append(acc)

            print("\nTest set - Epoch: {} - Loss: {:.4f}, Acc: ({:2f}%)\n".format(
                epoch, val_loss, 100 * acc))

            if active_worker == MAX_RANK:
                epoch += 1

            # Change worker and phase
            active_worker = (active_worker % MAX_RANK) + 1
            phase = "train"

            # Reset variables
            total_n_labels_test, correct_test = 0, 0
            step, batch_idx = 0, 0

            if msg == "training_complete":
                print("Training complete.")

                # Create validation loss and accuracy plots
                epoch_list = list(range(1, args.epochs+1))
                generate_simple_plot(epoch_list, val_losses,
                        "Test loss (Split Learning)", "epoch", "loss", [0.3, 0.9],
                        save=True, fname="test_loss_sl.pdf")
                generate_simple_plot(epoch_list, val_accs,
                        "Test accuracy (Split Learning)", "epoch", "accuracy",
                        [0.65, 1.0], save=True, fname="test_acc_sl.pdf")

                print("Total training time: {:.2f}s".format(total_training_time))
                print("Final test accuracy: {:.4f}".format(acc))
                print("Final test loss: {:.4f}".format(val_loss))

                exit()

            # Only reset validation loss if training not complete
            val_loss = 0.0

        elif msg == "validation":
            # Change phase and reset variables
            phase = "validation"
            step , total_n_labels_train, correct_train = 0, 0, 0

        elif msg == "data_downloaded":
            exit()

