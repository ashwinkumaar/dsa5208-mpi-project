import argparse
import csv
import math
import numpy as np
import time
import pickle
from mpi4py import MPI
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt


# ---------------- Utility Functions ----------------

def mpi_print(rank: int, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


def safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def parse_datetime(dt_str: str) -> Tuple[int, int]:
    try:
        import datetime as _dt
        dt = _dt.datetime.strptime(dt_str.strip(), "%m/%d/%Y %I:%M:%S %p")
        return dt.hour, dt.weekday()
    except Exception:
        return 0, 0


def minutes_between(start: str, end: str) -> float:
    try:
        import datetime as _dt
        s = _dt.datetime.strptime(start.strip(), "%m/%d/%Y %I:%M:%S %p")
        e = _dt.datetime.strptime(end.strip(), "%m/%d/%Y %I:%M:%S %p")
        return (e - s).total_seconds() / 60.0
    except Exception:
        return 0.0


# ---------------- Parallel RMSE ----------------

def parallel_rmse(y_true_local: np.ndarray, y_pred_local: np.ndarray, comm: MPI.Intracomm) -> float:
    """Compute RMSE across all processes"""
    err_local = np.sum((y_true_local - y_pred_local) ** 2)
    n_local = len(y_true_local)

    err_global = comm.allreduce(err_local, op=MPI.SUM)
    n_global = comm.allreduce(n_local, op=MPI.SUM)

    return math.sqrt(err_global / max(n_global, 1))


def parallel_loss(y_true_local: np.ndarray, y_pred_local: np.ndarray, comm: MPI.Intracomm) -> float:
    """Compute R(θ) = 1/(2N) * Σ(f(x) - y)² across all processes"""
    err_local = np.sum((y_true_local - y_pred_local) ** 2)
    n_local = len(y_true_local)

    err_global = comm.allreduce(err_local, op=MPI.SUM)
    n_global = comm.allreduce(n_local, op=MPI.SUM)

    return err_global / (2 * max(n_global, 1))


# ---------------- Normalization ----------------

class Standardizer:
    def __init__(self, dim: int):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.std = np.ones(dim, dtype=np.float64)

    def fit_global(self, x_local: np.ndarray, comm: MPI.Intracomm):
        """Fit standardizer using statistics from all processes"""
        n_local = x_local.shape[0]
        sum_local = np.sum(x_local, axis=0)
        sq_local = np.sum(x_local * x_local, axis=0)

        sum_global = comm.allreduce(sum_local, op=MPI.SUM)
        sq_global = comm.allreduce(sq_local, op=MPI.SUM)
        n_global = comm.allreduce(n_local, op=MPI.SUM)

        self.mean = sum_global / max(n_global, 1)
        variance = sq_global / max(n_global, 1) - self.mean ** 2
        self.std = np.sqrt(np.maximum(variance, 1e-8))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        """For inverse transforming target variable if needed"""
        return y * self.std[-1] + self.mean[-1]


# ---------------- Data Preprocessing ----------------

TARGET_COL = "total_amount"


def preprocess_rows(rows, args):
    """Preprocess raw CSV rows into feature matrix and target vector"""
    X, y = [], []
    for r in rows:
        # Skip rows with missing required fields
        required_fields = [
            "tpep_pickup_datetime", "tpep_dropoff_datetime",
            "passenger_count", "trip_distance", "RatecodeID",
            "PULocationID", "DOLocationID", "payment_type",
            "extra", TARGET_COL
        ]
        if any(r.get(f, "").strip() == "" for f in required_fields):
            continue  # drop incomplete row

        # Extract datetime features
        pu = r.get("tpep_pickup_datetime", "")
        do = r.get("tpep_dropoff_datetime", "")
        hh, dow = parse_datetime(pu)
        dur = minutes_between(pu, do)

        # Extract numeric features
        pc = safe_float(r.get("passenger_count", "nan"))
        dist = safe_float(r.get("trip_distance", "nan"))
        rate = int(r.get("RatecodeID", "0"))
        puid = int(r.get("PULocationID", "0")) % args.hash_buckets
        doid = int(r.get("DOLocationID", "0")) % args.hash_buckets
        pay = int(r.get("payment_type", "0"))
        extra = safe_float(r.get("extra", "nan"))
        target = safe_float(r.get(TARGET_COL, "nan"))

        # Skip if any numeric field is NaN or target is invalid
        if any(math.isnan(v) for v in [pc, dist, extra, target]) or target <= 0:
            continue

        features = [hh, dow, pc, dist, dur, rate, puid, doid, pay, extra]
        X.append(features)
        y.append(target)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_and_split_data(csv_path: str, train_split: float, max_rows: int, comm: MPI.Intracomm, args):
    """Load data and split into train/test sets distributed across processes"""
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load all data on rank 0, then distribute
    if rank == 0:
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                rows.append(row)
                if max_rows and i >= max_rows:
                    break

        # Preprocess all data
        X_all, y_all = preprocess_rows(rows, args)
        mpi_print(rank, f"Total samples after preprocessing: {len(X_all)}")

        # Shuffle data
        indices = np.random.permutation(len(X_all))
        X_all = X_all[indices]
        y_all = y_all[indices]

        # Split train/test
        split_idx = int(len(X_all) * train_split)
        X_train_all = X_all[:split_idx]
        y_train_all = y_all[:split_idx]
        X_test_all = X_all[split_idx:]
        y_test_all = y_all[split_idx:]

        mpi_print(rank, f"Train samples: {len(X_train_all)}, Test samples: {len(X_test_all)}")
    else:
        X_train_all = y_train_all = X_test_all = y_test_all = None

    # Distribute data to processes
    X_train_all = comm.bcast(X_train_all, root=0)
    y_train_all = comm.bcast(y_train_all, root=0)
    X_test_all = comm.bcast(X_test_all, root=0)
    y_test_all = comm.bcast(y_test_all, root=0)

    # Each process gets its shard
    n_train = len(X_train_all)
    n_test = len(X_test_all)

    train_start = (n_train * rank) // size
    train_end = (n_train * (rank + 1)) // size
    test_start = (n_test * rank) // size
    test_end = (n_test * (rank + 1)) // size

    X_train_local = X_train_all[train_start:train_end]
    y_train_local = y_train_all[train_start:train_end]
    X_test_local = X_test_all[test_start:test_end]
    y_test_local = y_test_all[test_start:test_end]

    return X_train_local, y_train_local, X_test_local, y_test_local


# ---------------- Neural Network Model ----------------

class NeuralNetwork:
    """One-hidden-layer neural network following project specification"""

    def __init__(self, input_dim: int, hidden_dim: int, activation: str = "relu", learning_rate: float = 0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.activation = activation

        # Xavier/Glorot initialization
        # W_jk: weights from input k to hidden neuron j
        self.W_hidden = np.random.uniform(
            -np.sqrt(6 / (input_dim + hidden_dim)),
            np.sqrt(6 / (input_dim + hidden_dim)),
            (input_dim, hidden_dim)
        )
        # b_j: bias for hidden neuron j
        self.b_hidden = np.zeros(hidden_dim)

        # w_j: weights from hidden neuron j to output
        self.w_output = np.random.uniform(
            -np.sqrt(6 / (hidden_dim + 1)),
            np.sqrt(6 / (hidden_dim + 1)),
            hidden_dim
        )
        # output bias
        self.b_output = 0.0

    def activation_function(self, z: np.ndarray) -> np.ndarray:
        """Apply activation function"""
        if self.activation == "relu":
            return np.maximum(0, z)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to prevent overflow
        elif self.activation == "tanh":
            return np.tanh(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def activation_derivative(self, z: np.ndarray) -> np.ndarray:
        """Compute derivative of activation function"""
        if self.activation == "relu":
            return (z > 0).astype(float)
        elif self.activation == "sigmoid":
            s = self.activation_function(z)
            return s * (1 - s)
        elif self.activation == "tanh":
            return 1 - np.tanh(z) ** 2
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: f(x;θ) = Σ(wj * σ(Σ(wjk * xk + bj))) + b_output"""
        # Hidden layer: z = X @ W + b
        self.z_hidden = X @ self.W_hidden + self.b_hidden
        self.a_hidden = self.activation_function(self.z_hidden)

        # Output layer
        self.output = self.a_hidden @ self.w_output + self.b_output

        return self.output

    def compute_gradient(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute gradients ∇θf for the batch"""
        m = len(y)
        if m == 0:
            return {}

        # Forward pass
        y_pred = self.forward(X)

        # Output layer gradients
        error = y_pred - y  # Shape: (m,)

        # Gradient w.r.t. output weights and bias
        grad_w_output = (self.a_hidden.T @ error) / m  # Shape: (hidden_dim,)
        grad_b_output = np.mean(error)

        # Hidden layer gradients
        # δ_hidden = (error @ w_output^T) * σ'(z_hidden)
        delta_hidden = np.outer(error, self.w_output) * self.activation_derivative(
            self.z_hidden)  # Shape: (m, hidden_dim)

        grad_W_hidden = (X.T @ delta_hidden) / m  # Shape: (input_dim, hidden_dim)
        grad_b_hidden = np.mean(delta_hidden, axis=0)  # Shape: (hidden_dim,)

        return {
            'W_hidden': grad_W_hidden,
            'b_hidden': grad_b_hidden,
            'w_output': grad_w_output,
            'b_output': grad_b_output
        }

    def update_parameters(self, gradients: Dict[str, np.ndarray]):
        """Update parameters using gradients"""
        if not gradients:
            return

        # Gradient clipping for stability
        clip_value = 1.0

        grad_W_hidden = np.clip(gradients['W_hidden'], -clip_value, clip_value)
        grad_b_hidden = np.clip(gradients['b_hidden'], -clip_value, clip_value)
        grad_w_output = np.clip(gradients['w_output'], -clip_value, clip_value)
        grad_b_output = np.clip(gradients['b_output'], -clip_value, clip_value)

        # Parameter updates
        self.W_hidden -= self.lr * grad_W_hidden
        self.b_hidden -= self.lr * grad_b_hidden
        self.w_output -= self.lr * grad_w_output
        self.b_output -= self.lr * grad_b_output


# ---------------- SGD Training ----------------

def stochastic_gradient_descent(model: NeuralNetwork, X_train_local: np.ndarray, y_train_local: np.ndarray,
                                X_test_local: np.ndarray, y_test_local: np.ndarray,
                                batch_size: int, max_iterations: int, comm: MPI.Intracomm,
                                patience: int = 10) -> Dict[str, List]:
    """
    Implement SGD as specified in the project:
    ∇R(θ) ≈ (1/M) * Σ[f(x_ji; θ) - y_ji] * ∇f(x_ji; θ)
    """
    rank = comm.Get_rank()
    n_local = len(X_train_local)

    # Training history
    history = {
        'train_loss': [],
        'train_rmse': [],
        'test_rmse': [],
        'iterations': []
    }

    best_loss = float('inf')
    patience_counter = 0

    mpi_print(rank, f"Starting SGD training with batch_size={batch_size}")

    for iteration in range(max_iterations):
        # Randomly sample M distinct indices for stochastic gradient
        if n_local > 0:
            if batch_size >= n_local:
                batch_indices = np.arange(n_local)
            else:
                batch_indices = np.random.choice(n_local, batch_size, replace=False)

            X_batch = X_train_local[batch_indices]
            y_batch = y_train_local[batch_indices]
        else:
            X_batch = np.array([]).reshape(0, X_train_local.shape[1])
            y_batch = np.array([])

        # Compute local gradient
        local_gradients = model.compute_gradient(X_batch, y_batch)

        # Average gradients across all processes
        if local_gradients:
            for key in local_gradients:
                local_gradients[key] = comm.allreduce(local_gradients[key], op=MPI.SUM) / comm.Get_size()

        # Update parameters
        model.update_parameters(local_gradients)

        # Compute training loss and metrics
        if iteration % 5 == 0 or iteration == max_iterations - 1:
            y_train_pred = model.forward(X_train_local)
            train_loss = parallel_loss(y_train_local, y_train_pred, comm)
            train_rmse = parallel_rmse(y_train_local, y_train_pred, comm)

            y_test_pred = model.forward(X_test_local)
            test_rmse = parallel_rmse(y_test_local, y_test_pred, comm)

            history['train_loss'].append(train_loss)
            history['train_rmse'].append(train_rmse)
            history['test_rmse'].append(test_rmse)
            history['iterations'].append(iteration)

            mpi_print(rank,
                      f"Iter {iteration}: Train Loss={train_loss:.6f}, Train RMSE={train_rmse:.4f}, Test RMSE={test_rmse:.4f}")

            # Check for convergence
            if train_loss < best_loss - 1e-6:  # Improvement threshold
                best_loss = train_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                mpi_print(rank, f"Converged at iteration {iteration}")
                break

    return history


# ---------------- Experimental Framework ----------------

def run_experiments(args):
    """Run experiments with different activation functions and batch sizes"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load and split data
    mpi_print(rank, f"Loading data from {args.csv}")
    X_train_local, y_train_local, X_test_local, y_test_local = load_and_split_data(
        args.csv, args.train_split, args.max_rows, comm, args
    )

    # Normalize features
    scaler = Standardizer(X_train_local.shape[1])
    scaler.fit_global(X_train_local, comm)
    X_train_local = scaler.transform(X_train_local)
    X_test_local = scaler.transform(X_test_local)

    mpi_print(rank, f"Local data - Train: {len(X_train_local)}, Test: {len(X_test_local)}")

    # Experimental parameters
    activations = ["relu", "sigmoid", "tanh"]
    batch_sizes = [16, 32, 64, 128, 256]

    results = []

    for activation in activations:
        for batch_size in batch_sizes:
            mpi_print(rank, f"\n{'=' * 50}")
            mpi_print(rank, f"Experiment: activation={activation}, batch_size={batch_size}")
            mpi_print(rank, f"{'=' * 50}")

            start_time = time.time()

            # Create and train model
            model = NeuralNetwork(
                input_dim=X_train_local.shape[1],
                hidden_dim=args.hidden,
                activation=activation,
                learning_rate=args.lr
            )

            history = stochastic_gradient_descent(
                model, X_train_local, y_train_local, X_test_local, y_test_local,
                batch_size=batch_size, max_iterations=args.max_iter, comm=comm
            )

            training_time = time.time() - start_time

            # Final evaluation
            y_train_pred = model.forward(X_train_local)
            y_test_pred = model.forward(X_test_local)

            final_train_rmse = parallel_rmse(y_train_local, y_train_pred, comm)
            final_test_rmse = parallel_rmse(y_test_local, y_test_pred, comm)
            final_train_loss = parallel_loss(y_train_local, y_train_pred, comm)

            result = {
                'activation': activation,
                'batch_size': batch_size,
                'hidden_neurons': args.hidden,
                'learning_rate': args.lr,
                'final_train_rmse': final_train_rmse,
                'final_test_rmse': final_test_rmse,
                'final_train_loss': final_train_loss,
                'training_time': training_time,
                'history': history,
                'n_processes': size
            }

            results.append(result)

            mpi_print(rank, f"Final Results:")
            mpi_print(rank, f"  Train RMSE: {final_train_rmse:.4f}")
            mpi_print(rank, f"  Test RMSE: {final_test_rmse:.4f}")
            mpi_print(rank, f"  Training Time: {training_time:.2f}s")

    # Save results
    if rank == 0:
        with open(f'results_np{size}.pkl', 'wb') as f:
            pickle.dump(results, f)

        # Create summary plots
        plot_results(results)

        print(f"\n{'=' * 60}")
        print("EXPERIMENTAL SUMMARY")
        print(f"{'=' * 60}")
        print(f"{'Activation':<10} {'Batch':<8} {'Train RMSE':<12} {'Test RMSE':<12} {'Time(s)':<10}")
        print(f"{'-' * 60}")
        for result in results:
            print(f"{result['activation']:<10} {result['batch_size']:<8} "
                  f"{result['final_train_rmse']:<12.4f} {result['final_test_rmse']:<12.4f} "
                  f"{result['training_time']:<10.2f}")

    return results


def plot_results(results):
    """Create training history plots"""
    plt.figure(figsize=(15, 10))

    # Plot 1: Training loss vs iteration for different activations
    plt.subplot(2, 3, 1)
    for activation in ["relu", "sigmoid", "tanh"]:
        result = next(r for r in results if r['activation'] == activation and r['batch_size'] == 64)
        plt.plot(result['history']['iterations'], result['history']['train_loss'],
                 label=f"{activation}", marker='o', markersize=3)
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss R(θ)')
    plt.title('Training Loss vs Iteration (Batch=64)')
    plt.legend()
    plt.grid(True)

    # Plot 2: Training loss vs iteration for different batch sizes
    plt.subplot(2, 3, 2)
    for batch_size in [16, 64, 256]:
        result = next(r for r in results if r['activation'] == 'relu' and r['batch_size'] == batch_size)
        plt.plot(result['history']['iterations'], result['history']['train_loss'],
                 label=f"Batch={batch_size}", marker='o', markersize=3)
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss R(θ)')
    plt.title('Training Loss vs Iteration (ReLU)')
    plt.legend()
    plt.grid(True)

    # Plot 3: RMSE comparison
    plt.subplot(2, 3, 3)
    activations = [r['activation'] for r in results if r['batch_size'] == 64]
    train_rmse = [r['final_train_rmse'] for r in results if r['batch_size'] == 64]
    test_rmse = [r['final_test_rmse'] for r in results if r['batch_size'] == 64]

    x = np.arange(len(activations))
    width = 0.35
    plt.bar(x - width / 2, train_rmse, width, label='Train RMSE')
    plt.bar(x + width / 2, test_rmse, width, label='Test RMSE')
    plt.xlabel('Activation Function')
    plt.ylabel('RMSE')
    plt.title('RMSE Comparison (Batch=64)')
    plt.xticks(x, activations)
    plt.legend()
    plt.grid(True)

    # Plot 4: Training time vs batch size
    plt.subplot(2, 3, 4)
    batch_sizes = [r['batch_size'] for r in results if r['activation'] == 'relu']
    times = [r['training_time'] for r in results if r['activation'] == 'relu']
    plt.plot(batch_sizes, times, 'o-')
    plt.xlabel('Batch Size')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time vs Batch Size (ReLU)')
    plt.grid(True)

    # Plot 5: Convergence comparison
    plt.subplot(2, 3, 5)
    for activation in ["relu", "sigmoid", "tanh"]:
        result = next(r for r in results if r['activation'] == activation and r['batch_size'] == 64)
        plt.plot(result['history']['iterations'], result['history']['test_rmse'],
                 label=f"{activation}", marker='o', markersize=3)
    plt.xlabel('Iteration')
    plt.ylabel('Test RMSE')
    plt.title('Test RMSE vs Iteration (Batch=64)')
    plt.legend()
    plt.grid(True)

    # Plot 6: Batch size effect on final performance
    plt.subplot(2, 3, 6)
    batch_sizes = [r['batch_size'] for r in results if r['activation'] == 'relu']
    test_rmse_batch = [r['final_test_rmse'] for r in results if r['activation'] == 'relu']
    plt.plot(batch_sizes, test_rmse_batch, 'o-')
    plt.xlabel('Batch Size')
    plt.ylabel('Test RMSE')
    plt.title('Test RMSE vs Batch Size (ReLU)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()


# ---------------- Main ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MPI-based SGD for Neural Networks')
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--max_iter", type=int, default=500, help="Maximum iterations")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=32, help="Hidden layer neurons")
    parser.add_argument("--hash_buckets", type=int, default=128, help="Hash buckets for categorical features")
    parser.add_argument("--train_split", type=float, default=0.7, help="Train/test split ratio")
    parser.add_argument("--max_rows", type=int, default=0, help="Max rows to process (0 for all)")

    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(5208)

    results = run_experiments(args)