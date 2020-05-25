# Split Learning and Federated Learning

This repository comprises of implementations of Split Learning [Vepakomma et al. 2018] and Federated Learning [McMahan et al. 2016]. The network architecture used for both implementations is a slightly modified version of the LeNet-5 [LeCun et al., 1998] convolutional neural network (CNN) architectures. All experiments are performed on the FashionMNIST data set [Xiao et al. 2018u].

## Setup

This repository has only be tested for Python version 3.7.3. To install all dependencies, use the following command:

```
$ pip install -r requirements.txt
```

Some of the code in this repository works with the Messagge Passing Interface (MPI). To be able to run the code, ```openmpi``` or a similar program has to be set up.

## Usage
To reproduce the experiments as described in *report.pdf*, do:
```
$ ./run_all_exps.sh
```

To apply the LeNet5 CNN regular model on FashionMNIST, do:
```
$ python3.7 src/regular_learning.py
```

To apply the LeNet5 CNN federated model with 10 workers on FashionMNIST, do:
```
$ python3.7 src/federated_learning.py
```

To apply the LeNet5 CNN split learning model with 10 workers on FashionMNIST, do:

```
$ mpirun -n 11 python3.7 src/split_learning.py
```

For more information about the split learning and federated learning technologies and the experiments that can be carried out with the code in this repository, we kindly refer to the file *report.pdf*.

## References
[1] Praneeth Vepakomma, Otkrist Gupta, Tristan Swedish, and Ramesh Raskar.   Split learning for health:
Distributed deep learning without sharing raw patient data. arXiv preprint arXiv:1812.00564, 2018.

[2] H Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, et al. Communication-efficient learning
of deep networks from decentralized data. arXiv preprint arXiv:1602.05629, 2016.

[3] Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner, et al.  Gradient-based learning applied to
document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

[4] Han Xiao, Kashif Rasul, and Roland Vollgraf.  Fashion-mnist:  a novel image dataset for benchmarking
machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017
