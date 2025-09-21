# Distributed Neural Network Training with MPI

## Overview

This project implements a distributed stochastic gradient descent (SGD) algorithm for training a neural network with one hidden layer, using the Message Passing Interface (MPI) for parallelization. The code is designed to efficiently handle large tabular datasets, such as the NYC Taxi 2022 dataset, and supports a variety of activation functions, batch sizes, and optimization improvements (momentum, batch normalization, early stopping).

### Key features:
* Distributed data loading and sharding for memory efficiency
* Parallel SGD with gradient averaging across processes
* Configurable neural network architecture and training parameters
* Support for multiple activation functions and batch normalization
* Comprehensive logging, result saving, and visualization

## Requirements

* Python 3.7+
* mpi4py
* numpy
* matplotlib

Install dependencies (if needed):
```bash
pip install mpi4py numpy matplotlib
```

You must also have an MPI implementation installed (e.g., OpenMPI or MPICH).

## Usage

The main script is `mpi_sgd_nn.py`. **You must run it using `mpiexec` or `mpirun` to enable distributed training.**

### Basic Command

```bash
mpiexec -n <num_processes> python mpi_sgd_nn.py --csv <path_to_csv>
```

* `<num_processes>`: Number of parallel MPI processes to use (e.g., 4, 8, 16)
* `<path_to_csv>`: Path to your dataset (e.g., nytaxi2022.csv)

### Example

```bash
mpiexec -n 6 python mpi_sgd_nn.py --csv dataset/nytaxi2022.csv --train_split 0.7 --max_rows 1000000
```

## Arguments

| **Argument** | **Type** | **Default** | **Description** |
|--------------|----------|-------------|-----------------|
| --csv | str | **(req)** | Path to the CSV dataset |
| --max_iter | int | 500 | Maximum number of SGD iterations |
| --lr | float | 0.01 | Learning rate for SGD |
| --hidden | int | 64 | Number of neurons in the hidden layer |
| --hash_buckets | int | 256 | Number of hash buckets for categorical features |
| --train_split | float | 0.7 | Fraction of data to use for training (rest for testing) |
| --max_rows | int | 0 | Maximum number of rows to process (0 = all rows) |
| --momentum | float | 0 | Momentum coefficient (set to 0 for vanilla SGD) |
| --patience | int | 1e9 | Early stopping patience (set high to disable early stopping) |
| --use_batch_norm | flag | False | Enable batch normalization (add this flag to turn it on) |

## Example Commands

### 1. Run with 4 processes, default settings:
```bash
mpiexec -n 4 python mpi_sgd_nn.py --csv nytaxi2022.csv
```

### 2. Run with 8 processes, 128 hidden neurons, and 100,000 rows:
```bash
mpiexec -n 8 python mpi_sgd_nn.py --csv nytaxi2022.csv --hidden 128 --max_rows 100000
```

### 3. Run with 6 processes, momentum and batch normalization enabled:
```bash
mpiexec -n 6 python mpi_sgd_nn.py --csv nytaxi2022.csv --momentum 0.9 --use_batch_norm
```

### 4. Run with early stopping (patience 50) and a custom learning rate:
```bash
mpiexec -n 6 python mpi_sgd_nn.py --csv nytaxi2022.csv --patience 50 --lr 0.005
```

## Output

* **Results:** Results are saved as a pickle file (e.g., `results_np6_<timestamp>.pkl`) in the current directory.
* **Plots:** Training history and performance plots are saved as `training_results.png`.
* **Console Logs:** Progress, training summaries, and experiment results are printed to the console.

## Notes

* For large datasets, increase the number of processes (`-n`) to reduce per-process memory usage.
* Use `--max_rows` to test on a subset of data before scaling up.
* The script logs progress every 10,000 rows during data loading.
* Batch normalization, momentum, and early stopping are optional improvements; disable them for baseline runs.

## Troubleshooting

* **Memory errors:** Reduce `--max_rows` or increase the number of processes.
* **Slow performance at high process counts:** Communication overhead may dominate; optimal process count depends on your hardware.

## Repository
* You can find the GitHub repository of this project here: https://github.com/ashwinkumaar/dsa5208-mpi-project