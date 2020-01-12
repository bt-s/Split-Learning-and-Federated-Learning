# Run Split Learning on Fashion-MNIST with 10 workers
mpirun -n 11 python3.7 src/split_learning.py

# Run Federated Learning on Fashion-MNIST with 10 workers
python3.7 src/federated_learning.py

# Run Regular Learing on Fashion-MNIST
python3.7 src/regular_learning.py
