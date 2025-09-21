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


# ---------------- Parallel RMSE and Loss ----------------

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


# ---------------- Batch Normalization ----------------

class BatchNorm:
    def __init__(self, dim: int, eps: float = 1e-8):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)
        self.momentum = 0.9

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if x.shape[0] == 0:
            return x

        if training and x.shape[0] > 1:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            # Normalize
            x_norm = (x - mean) / np.sqrt(var + self.eps)
        else:
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.gamma * x_norm + self.beta


# ---------------- Data Preprocessing ----------------

TARGET_COL = "total_amount"


def preprocess_rows(rows, args):
    """Enhanced preprocessing with feature engineering"""
    X, y = [], []
    skipped = 0
    for r in rows:
        # Skip incomplete rows
        required_fields = [
            "tpep_pickup_datetime", "tpep_dropoff_datetime",
            "passenger_count", "trip_distance", "RatecodeID",
            "PULocationID", "DOLocationID", "payment_type",
            "extra", TARGET_COL
        ]
        if any(r.get(f, "").strip() == "" or '/uFFFD' in r.get(f, "") for f in required_fields):
            skipped += 1
            continue

        # Extract and engineer features
        pu = r.get("tpep_pickup_datetime", "")
        do = r.get("tpep_dropoff_datetime", "")
        hh, dow = parse_datetime(pu)
        dur = minutes_between(pu, do)

        # Additional time features
        is_weekend = 1 if dow >= 5 else 0
        is_rush_hour = 1 if (7 <= hh <= 9) or (17 <= hh <= 19) else 0
        is_night = 1 if hh >= 22 or hh <= 5 else 0

        # Numeric features with outlier handling
        pc = safe_float(r.get("passenger_count", "nan"))
        dist = safe_float(r.get("trip_distance", "nan"))
        rate = int(r.get("RatecodeID", "0"))
        puid = int(r.get("PULocationID", "0")) % args.hash_buckets
        doid = int(r.get("DOLocationID", "0")) % args.hash_buckets
        pay = int(r.get("payment_type", "0"))
        extra = safe_float(r.get("extra", "nan"))
        target = safe_float(r.get(TARGET_COL, "nan"))

        # Outlier filtering
        if (any(math.isnan(v) for v in [pc, dist, extra, target]) or
                target <= 0 or target > 1000 or  # reasonable fare range
                dist > 500 or dist < 0 or  # reasonable distance range
                dur > 300 or dur < 0 or  # reasonable duration range
                pc <= 0 or pc > 8):  # reasonable passenger count
            continue

        # Speed feature (with safety check)
        speed = dist / max(dur / 60, 0.1) if dur > 0 else 0  # miles per hour

        # Log transform for skewed features
        log_dist = math.log(dist + 1)
        log_dur = math.log(dur + 1)

        features = [hh, dow, pc, dist, dur, rate, puid, doid, pay, extra,
                    is_weekend, is_rush_hour, is_night, speed, log_dist, log_dur]
        X.append(features)
        y.append(target)

    print(f"Skipped {skipped} rows during preprocessing")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_and_split_data(csv_path: str, train_split: float, max_rows: int, comm: MPI.Intracomm, args):
    """Load data and split into train/test sets distributed across processes"""
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load all data on rank 0, then distribute
    if rank == 0:
        rows = []
        with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
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


# ---------------- Improved Neural Network Model ----------------

class NeuralNetwork:
    """Enhanced one-hidden-layer neural network with momentum and batch normalization"""

    def __init__(self, input_dim: int, hidden_dim: int, activation: str = "relu",
                 learning_rate: float = 0.01, momentum: float = 0.9, use_batch_norm: bool = True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.initial_lr = learning_rate
        self.momentum = momentum
        self.activation = activation
        self.use_batch_norm = use_batch_norm

        # Xavier/Glorot initialization with improved scaling
        xavier_hidden = np.sqrt(2.0 / (input_dim + hidden_dim))
        xavier_output = np.sqrt(2.0 / (hidden_dim + 1))

        # W_jk: weights from input k to hidden neuron j
        self.W_hidden = np.random.normal(0, xavier_hidden, (input_dim, hidden_dim))
        # b_j: bias for hidden neuron j
        self.b_hidden = np.zeros(hidden_dim)

        # w_j: weights from hidden neuron j to output
        self.w_output = np.random.normal(0, xavier_output, hidden_dim)
        # output bias
        self.b_output = 0.0

        # Momentum terms
        self.v_W_hidden = np.zeros_like(self.W_hidden)
        self.v_b_hidden = np.zeros_like(self.b_hidden)
        self.v_w_output = np.zeros_like(self.w_output)
        self.v_b_output = 0.0

        # Batch normalization
        if self.use_batch_norm:
            self.batch_norm = BatchNorm(hidden_dim)

        # Learning rate scheduling
        self.decay_rate = 0.98
        self.decay_steps = 25

    def activation_function(self, z: np.ndarray) -> np.ndarray:
        """Apply activation function with improved numerical stability"""
        if self.activation == "relu":
            return np.maximum(0, z)
        elif self.activation == "leaky_relu":
            return np.where(z > 0, z, 0.01 * z)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == "tanh":
            return np.tanh(np.clip(z, -500, 500))
        elif self.activation == "elu":
            return np.where(z > 0, z, np.exp(np.clip(z, -500, 500)) - 1)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def activation_derivative(self, z: np.ndarray) -> np.ndarray:
        """Compute derivative of activation function"""
        if self.activation == "relu":
            return (z > 0).astype(float)
        elif self.activation == "leaky_relu":
            return np.where(z > 0, 1.0, 0.01)
        elif self.activation == "sigmoid":
            s = self.activation_function(z)
            return s * (1 - s)
        elif self.activation == "tanh":
            return 1 - np.tanh(z) ** 2
        elif self.activation == "elu":
            return np.where(z > 0, 1.0, np.exp(z))
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass with batch normalization"""
        if len(X) == 0:
            return np.array([])

        # Hidden layer: z = X @ W + b
        self.z_hidden = X @ self.W_hidden + self.b_hidden

        # Apply batch normalization if enabled
        if self.use_batch_norm:
            self.z_hidden_norm = self.batch_norm.forward(self.z_hidden, training)
            self.a_hidden = self.activation_function(self.z_hidden_norm)
        else:
            self.a_hidden = self.activation_function(self.z_hidden)

        # Output layer
        self.output = self.a_hidden @ self.w_output + self.b_output

        return self.output

    def compute_gradient(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute gradients with improved numerical stability"""
        m = len(y)
        if m == 0:
            return {}

        # Forward pass
        y_pred = self.forward(X, training=True)

        # Output layer gradients
        error = y_pred - y  # Shape: (m,)

        # Gradient w.r.t. output weights and bias
        grad_w_output = (self.a_hidden.T @ error) / m
        grad_b_output = np.mean(error)

        # Hidden layer gradients
        if self.use_batch_norm:
            delta_hidden = np.outer(error, self.w_output) * self.activation_derivative(self.z_hidden_norm)
        else:
            delta_hidden = np.outer(error, self.w_output) * self.activation_derivative(self.z_hidden)

        grad_W_hidden = (X.T @ delta_hidden) / m
        grad_b_hidden = np.mean(delta_hidden, axis=0)

        return {
            'W_hidden': grad_W_hidden,
            'b_hidden': grad_b_hidden,
            'w_output': grad_w_output,
            'b_output': grad_b_output
        }

    def update_parameters(self, gradients: Dict[str, np.ndarray], iteration: int = 0):
        """Update parameters with momentum and learning rate decay"""
        if not gradients:
            return

        # Learning rate decay
        current_lr = self.initial_lr * (self.decay_rate ** (iteration // self.decay_steps))

        # Adaptive gradient clipping based on gradient norm
        total_norm = 0
        for key in gradients:
            total_norm += np.sum(gradients[key] ** 2)
        total_norm = np.sqrt(total_norm)

        clip_value = 5.0
        if total_norm > clip_value:
            for key in gradients:
                gradients[key] = gradients[key] * clip_value / total_norm

        # Momentum updates
        self.v_W_hidden = self.momentum * self.v_W_hidden + current_lr * gradients['W_hidden']
        self.v_b_hidden = self.momentum * self.v_b_hidden + current_lr * gradients['b_hidden']
        self.v_w_output = self.momentum * self.v_w_output + current_lr * gradients['w_output']
        self.v_b_output = self.momentum * self.v_b_output + current_lr * gradients['b_output']

        # Parameter updates
        self.W_hidden -= self.v_W_hidden
        self.b_hidden -= self.v_b_hidden
        self.w_output -= self.v_w_output
        self.b_output -= self.v_b_output


# ---------------- Distributed SGD Training ----------------

def distributed_stochastic_gradient_descent(model: NeuralNetwork, X_train_local: np.ndarray,
                                            y_train_local: np.ndarray,
                                            X_test_local: np.ndarray, y_test_local: np.ndarray,
                                            batch_size: int, max_iterations: int, comm: MPI.Intracomm,
                                            patience: int = 15, min_improvement: float = 1e-6) -> Dict[str, List]:
    """
    Distributed SGD where each process samples from its local data
    This is a practical implementation that respects the data locality constraint
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_local = len(X_train_local)

    # Training history
    history = {
        'train_loss': [],
        'train_rmse': [],
        'test_rmse': [],
        'iterations': [],
        'learning_rates': []
    }

    best_loss = float('inf')
    patience_counter = 0

    mpi_print(rank, f"Starting Distributed SGD with batch_size={batch_size}, local_samples={n_local}")
    mpi_print(rank, f"Model: {model.activation}, hidden={model.hidden_dim}, lr={model.lr}, momentum={model.momentum}")

    for iteration in range(max_iterations):
        # Each process samples from its local data
        if n_local > 0:
            if batch_size >= n_local:
                batch_indices = np.arange(n_local)
            else:
                batch_indices = np.random.choice(n_local, batch_size, replace=False)

            X_batch = X_train_local[batch_indices]
            y_batch = y_train_local[batch_indices]
        else:
            X_batch = np.array([]).reshape(0, X_train_local.shape[1] if len(X_train_local.shape) > 1 else 0)
            y_batch = np.array([])

        # Compute local gradient
        local_gradients = model.compute_gradient(X_batch, y_batch)

        # Average gradients across all processes
        if local_gradients:
            for key in local_gradients:
                # Sum gradients from all processes and average
                local_gradients[key] = comm.allreduce(local_gradients[key], op=MPI.SUM) / size

        # Update parameters
        model.update_parameters(local_gradients, iteration)

        # Evaluation every 5 iterations
        if iteration % 5 == 0 or iteration == max_iterations - 1:
            y_train_pred = model.forward(X_train_local, training=False)
            train_loss = parallel_loss(y_train_local, y_train_pred, comm)
            train_rmse = parallel_rmse(y_train_local, y_train_pred, comm)

            y_test_pred = model.forward(X_test_local, training=False)
            test_rmse = parallel_rmse(y_test_local, y_test_pred, comm)

            # Current learning rate
            current_lr = model.initial_lr * (model.decay_rate ** (iteration // model.decay_steps))

            history['train_loss'].append(train_loss)
            history['train_rmse'].append(train_rmse)
            history['test_rmse'].append(test_rmse)
            history['iterations'].append(iteration)
            history['learning_rates'].append(current_lr)

            mpi_print(rank,
                      f"Iter {iteration:4d}: Loss={train_loss:.6f}, Train RMSE={train_rmse:.4f}, "
                      f"Test RMSE={test_rmse:.4f}, LR={current_lr:.6f}")

            # Convergence check with improved criteria
            if train_loss < best_loss - min_improvement:
                best_loss = train_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                mpi_print(rank, f"Converged at iteration {iteration} (no improvement for {patience} checks)")
                break

            # Additional stopping criterion: if loss becomes too small
            if train_loss < 1e-8:
                mpi_print(rank, f"Stopped at iteration {iteration} (loss converged to near zero)")
                break

    return history


# ---------------- Experimental Framework ----------------

def run_experiments(args):
    """Run experiments with different activation functions and batch sizes"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load and split data
    mpi_print(rank, f"Loading data from {args.csv} with {size} processes")
    start_load_time = time.time()

    X_train_local, y_train_local, X_test_local, y_test_local = load_and_split_data(
        args.csv, args.train_split, args.max_rows, comm, args
    )

    load_time = time.time() - start_load_time
    mpi_print(rank, f"Data loading completed in {load_time:.2f} seconds")

    # Normalize features
    mpi_print(rank, "Normalizing features...")
    scaler = Standardizer(X_train_local.shape[1])
    scaler.fit_global(X_train_local, comm)
    X_train_local = scaler.transform(X_train_local)
    X_test_local = scaler.transform(X_test_local)

    mpi_print(rank, f"Local data - Train: {len(X_train_local)}, Test: {len(X_test_local)}")
    mpi_print(rank, f"Feature dimensions: {X_train_local.shape[1]}")

    # Experimental parameters
    activations = ["relu", "leaky_relu", "tanh", "sigmoid", "elu"]
    batch_sizes = [32, 64, 128, 256, 512]

    results = []
    total_experiments = len(activations) * len(batch_sizes)
    experiment_count = 0

    for activation in activations:
        for batch_size in batch_sizes:
            experiment_count += 1
            mpi_print(rank, f"\n{'=' * 70}")
            mpi_print(rank,
                      f"Experiment {experiment_count}/{total_experiments}: activation={activation}, batch_size={batch_size}")
            mpi_print(rank, f"{'=' * 70}")

            start_time = time.time()

            # Create and train model
            model = NeuralNetwork(
                input_dim=X_train_local.shape[1],
                hidden_dim=args.hidden,
                activation=activation,
                learning_rate=args.lr,
                momentum=args.momentum,
                use_batch_norm=args.use_batch_norm
            )

            # Initial evaluation
            y_train_pred = model.forward(X_train_local, training=False)
            y_test_pred = model.forward(X_test_local, training=False)

            initial_train_rmse = parallel_rmse(y_train_local, y_train_pred, comm)
            initial_test_rmse = parallel_rmse(y_test_local, y_test_pred, comm)
            initial_train_loss = parallel_loss(y_train_local, y_train_pred, comm)

            history = distributed_stochastic_gradient_descent(
                model, X_train_local, y_train_local, X_test_local, y_test_local,
                batch_size=batch_size, max_iterations=args.max_iter, comm=comm,
                patience=args.patience
            )

            training_time = time.time() - start_time

            # Final evaluation
            y_train_pred = model.forward(X_train_local, training=False)
            y_test_pred = model.forward(X_test_local, training=False)

            final_train_rmse = parallel_rmse(y_train_local, y_train_pred, comm)
            final_test_rmse = parallel_rmse(y_test_local, y_test_pred, comm)
            final_train_loss = parallel_loss(y_train_local, y_train_pred, comm)

            result = {
                'activation': activation,
                'batch_size': batch_size,
                'hidden_neurons': args.hidden,
                'learning_rate': args.lr,
                'momentum': args.momentum,
                'use_batch_norm': args.use_batch_norm,
                'initial_train_rmse': final_train_rmse,
                'initial_test_rmse': final_test_rmse,
                'initial_train_loss': final_train_loss,
                'final_train_rmse': final_train_rmse,
                'final_test_rmse': final_test_rmse,
                'final_train_loss': final_train_loss,
                'training_time': training_time,
                'history': history,
                'n_processes': size,
                'feature_dim': X_train_local.shape[1]
            }

            results.append(result)

            mpi_print(rank, f"Final Results:")
            mpi_print(rank, f"  Train RMSE: {final_train_rmse:.4f}")
            mpi_print(rank, f"  Test RMSE: {final_test_rmse:.4f}")
            mpi_print(rank, f"  Final Loss: {final_train_loss:.6f}")
            mpi_print(rank, f"  Training Time: {training_time:.2f}s")
            mpi_print(rank, f"  Iterations: {len(history['train_loss']) * 5}")

    # Save results
    if rank == 0:
        timestamp = int(time.time())
        results_filename = f'improved_results_np{size}_{timestamp}.pkl'
        with open(results_filename, 'wb') as f:
            pickle.dump(results, f)
        mpi_print(rank, f"Results saved to {results_filename}")

        # Create summary plots
        plot_results(results)

        # Print comprehensive summary
        print_experiment_summary(results)

    return results


def plot_results(results):
    """Create comprehensive training history plots"""
    plt.figure(figsize=(20, 15))

    # Plot 1: Training loss comparison for different activations
    plt.subplot(3, 4, 1)
    for activation in ["relu", "leaky_relu", "tanh", "sigmoid", "elu"]:
        result = next((r for r in results if r['activation'] == activation and r['batch_size'] == 128), None)
        if result:
            plt.plot(result['history']['iterations'], result['history']['train_loss'],
                     label=f"{activation}", marker='o', markersize=2)
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss R(θ)')
    plt.title('Training Loss vs Iteration (Batch=128)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Plot 2: Learning rate schedule
    plt.subplot(3, 4, 2)
    result = next((r for r in results if r['activation'] == 'relu' and r['batch_size'] == 128), None)
    if result and 'learning_rates' in result['history']:
        plt.plot(result['history']['iterations'], result['history']['learning_rates'], 'r-')
        plt.xlabel('Iteration')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)

    # Plot 3: RMSE comparison by activation
    plt.subplot(3, 4, 3)
    activations = list(set([r['activation'] for r in results]))
    train_rmse_by_act = []
    test_rmse_by_act = []

    for act in activations:
        results_act = [r for r in results if r['activation'] == act and r['batch_size'] == 128]
        if results_act:
            train_rmse_by_act.append(results_act[0]['final_train_rmse'])
            test_rmse_by_act.append(results_act[0]['final_test_rmse'])

    x = np.arange(len(activations))
    width = 0.35
    plt.bar(x - width / 2, train_rmse_by_act, width, label='Train RMSE', alpha=0.8)
    plt.bar(x + width / 2, test_rmse_by_act, width, label='Test RMSE', alpha=0.8)
    plt.xlabel('Activation Function')
    plt.ylabel('RMSE')
    plt.title('RMSE by Activation (Batch=128)')
    plt.xticks(x, activations, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4: Training time vs batch size
    plt.subplot(3, 4, 4)
    batch_sizes = sorted(list(set([r['batch_size'] for r in results])))
    times_by_batch = []

    for bs in batch_sizes:
        results_batch = [r for r in results if r['batch_size'] == bs and r['activation'] == 'relu']
        if results_batch:
            times_by_batch.append(results_batch[0]['training_time'])

    plt.plot(batch_sizes, times_by_batch, 'o-', color='green')
    plt.xlabel('Batch Size')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time vs Batch Size (ReLU)')
    plt.grid(True, alpha=0.3)

    # Plot 5: Convergence comparison
    plt.subplot(3, 4, 5)
    for activation in ["relu", "leaky_relu", "tanh", "sigmoid", "elu"]:
        result = next((r for r in results if r['activation'] == activation and r['batch_size'] == 128), None)
        if result:
            plt.plot(result['history']['iterations'], result['history']['test_rmse'],
                     label=f"{activation}", marker='o', markersize=2)
    plt.xlabel('Iteration')
    plt.ylabel('Test RMSE')
    plt.title('Test RMSE Convergence (Batch=128)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 6: Batch size effect on performance
    plt.subplot(3, 4, 6)
    test_rmse_by_batch = []
    train_rmse_by_batch = []

    for bs in batch_sizes:
        results_batch = [r for r in results if r['batch_size'] == bs and r['activation'] == 'relu']
        if results_batch:
            test_rmse_by_batch.append(results_batch[0]['final_test_rmse'])
            train_rmse_by_batch.append(results_batch[0]['final_train_rmse'])

    plt.plot(batch_sizes, train_rmse_by_batch, 'o-', label='Train RMSE', color='blue')
    plt.plot(batch_sizes, test_rmse_by_batch, 'o-', label='Test RMSE', color='red')
    plt.xlabel('Batch Size')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Batch Size (ReLU)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 7: Training loss over time for different batch sizes
    plt.subplot(3, 4, 7)
    for batch_size in [32, 128, 512]:
        result = next((r for r in results if r['activation'] == 'relu' and r['batch_size'] == batch_size), None)
        if result:
            plt.plot(result['history']['iterations'], result['history']['train_loss'],
                     label=f"Batch={batch_size}", marker='o', markersize=2)
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.title('Loss vs Iteration (ReLU)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Plot 8: Overfitting analysis (Train vs Test RMSE)
    plt.subplot(3, 4, 8)
    for activation in ["relu", "leaky_relu", "tanh", "sigmoid", "elu"]:
        results_act = [r for r in results if r['activation'] == activation]
        if results_act:
            train_rmse = [r['final_train_rmse'] for r in results_act]
            test_rmse = [r['final_test_rmse'] for r in results_act]
            plt.scatter(train_rmse, test_rmse, label=activation, alpha=0.7, s=50)

    # Diagonal line for reference
    min_rmse = min([r['final_train_rmse'] for r in results] + [r['final_test_rmse'] for r in results])
    max_rmse = max([r['final_train_rmse'] for r in results] + [r['final_test_rmse'] for r in results])
    plt.plot([min_rmse, max_rmse], [min_rmse, max_rmse], 'k--', alpha=0.5)

    plt.xlabel('Train RMSE')
    plt.ylabel('Test RMSE')
    plt.title('Overfitting Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 9: Performance heatmap
    plt.subplot(3, 4, 9)
    activations_sorted = sorted(list(set([r['activation'] for r in results])))
    batch_sizes_sorted = sorted(list(set([r['batch_size'] for r in results])))

    heatmap_data = np.zeros((len(activations_sorted), len(batch_sizes_sorted)))

    for i, act in enumerate(activations_sorted):
        for j, bs in enumerate(batch_sizes_sorted):
            result = next((r for r in results if r['activation'] == act and r['batch_size'] == bs), None)
            if result:
                heatmap_data[i, j] = result['final_test_rmse']

    im = plt.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')
    plt.colorbar(im)
    plt.yticks(range(len(activations_sorted)), activations_sorted)
    plt.xticks(range(len(batch_sizes_sorted)), batch_sizes_sorted)
    plt.xlabel('Batch Size')
    plt.ylabel('Activation')
    plt.title('Test RMSE Heatmap')

    # Plot 10: Training efficiency (RMSE improvement per second)
    plt.subplot(3, 4, 10)
    efficiency_scores = []
    labels = []

    for result in results:
        if result['training_time'] > 0:
            # Calculate efficiency as RMSE improvement per second
            initial_rmse = result['history']['test_rmse'][0]
            improvement = max(0, initial_rmse - result['final_test_rmse'])
            efficiency = improvement / result['training_time']
            efficiency_scores.append(efficiency)
            labels.append(f"{result['activation'][:4]}-{result['batch_size']}")

    # Sort by efficiency
    sorted_indices = np.argsort(efficiency_scores)[:10]  # Top 10
    top_scores = [efficiency_scores[i] for i in sorted_indices]
    top_labels = [labels[i] for i in sorted_indices]

    plt.barh(range(len(top_scores)), top_scores)
    plt.yticks(range(len(top_scores)), top_labels)
    plt.xlabel('Efficiency (RMSE Improvement/Second)')
    plt.title('Training Efficiency (Top 10)')
    plt.grid(True, alpha=0.3)

    # Plot 11: Loss convergence comparison
    plt.subplot(3, 4, 11)
    best_results = {}
    for result in results:
        key = result['activation']
        if key not in best_results or result['final_test_rmse'] < best_results[key]['final_test_rmse']:
            best_results[key] = result

    for activation, result in best_results.items():
        plt.plot(result['history']['iterations'], result['history']['train_loss'],
                 label=f"{activation} (bs={result['batch_size']})", marker='o', markersize=2)
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.title('Best Loss Convergence per Activation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Plot 12: Final performance summary
    plt.subplot(3, 4, 12)
    final_results = []
    final_labels = []

    for result in results:
        final_results.append(result['final_test_rmse'])
        final_labels.append(f"{result['activation'][:4]}-{result['batch_size']}")

    # Sort by performance
    sorted_indices = np.argsort(final_results)[:8]  # Best 8
    best_rmse = [final_results[i] for i in sorted_indices]
    best_labels = [final_labels[i] for i in sorted_indices]

    plt.barh(range(len(best_rmse)), best_rmse, color='lightcoral')
    plt.yticks(range(len(best_rmse)), best_labels)
    plt.xlabel('Test RMSE')
    plt.title('Best Configurations (Test RMSE)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('improved_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_experiment_summary(results):
    """Print comprehensive experimental summary"""
    print(f"\n{'=' * 100}")
    print("COMPREHENSIVE EXPERIMENTAL SUMMARY")
    print(f"{'=' * 100}")

    # Overall statistics
    total_experiments = len(results)
    total_time = sum(r['training_time'] for r in results)
    avg_time = total_time / total_experiments if total_experiments > 0 else 0

    print(f"Total Experiments: {total_experiments}")
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Average Training Time: {avg_time:.2f} seconds")
    print(f"Number of Processes: {results[0]['n_processes'] if results else 'N/A'}")
    print(f"Feature Dimensions: {results[0]['feature_dim'] if results else 'N/A'}")

    # Best overall performance
    best_result = min(results, key=lambda x: x['final_test_rmse'])
    print(f"\nBEST OVERALL PERFORMANCE:")
    print(f"  Configuration: {best_result['activation']} activation, batch_size={best_result['batch_size']}")
    print(f"  Test RMSE: {best_result['final_test_rmse']:.4f}")
    print(f"  Train RMSE: {best_result['final_train_rmse']:.4f}")
    print(f"  Training Time: {best_result['training_time']:.2f}s")

    # Performance by activation function
    print(f"\nPERFORMANCE BY ACTIVATION FUNCTION:")
    print(f"{'Activation':<12} {'Best RMSE':<12} {'Avg RMSE':<12} {'Std RMSE':<12} {'Best Batch':<12}")
    print("-" * 70)

    activations = list(set([r['activation'] for r in results]))
    for activation in sorted(activations):
        act_results = [r for r in results if r['activation'] == activation]
        test_rmses = [r['final_test_rmse'] for r in act_results]
        best_rmse = min(test_rmses)
        avg_rmse = np.mean(test_rmses)
        std_rmse = np.std(test_rmses)
        best_batch = next(r['batch_size'] for r in act_results if r['final_test_rmse'] == best_rmse)

        print(f"{activation:<12} {best_rmse:<12.4f} {avg_rmse:<12.4f} {std_rmse:<12.4f} {best_batch:<12}")

    # Performance by batch size
    print(f"\nPERFORMANCE BY BATCH SIZE:")
    print(f"{'Batch Size':<12} {'Best RMSE':<12} {'Avg RMSE':<12} {'Avg Time':<12}")
    print("-" * 50)

    batch_sizes = sorted(list(set([r['batch_size'] for r in results])))
    for batch_size in batch_sizes:
        batch_results = [r for r in results if r['batch_size'] == batch_size]
        test_rmses = [r['final_test_rmse'] for r in batch_results]
        times = [r['training_time'] for r in batch_results]
        best_rmse = min(test_rmses)
        avg_rmse = np.mean(test_rmses)
        avg_time = np.mean(times)

        print(f"{batch_size:<12} {best_rmse:<12.4f} {avg_rmse:<12.4f} {avg_time:<12.2f}")

    # Top 5 configurations
    print(f"\nTOP 5 CONFIGURATIONS:")
    print(f"{'Rank':<6} {'Activation':<12} {'Batch':<8} {'Test RMSE':<12} {'Train RMSE':<12} {'Time(s)':<10}")
    print("-" * 70)

    sorted_results = sorted(results, key=lambda x: x['final_test_rmse'])
    for i, result in enumerate(sorted_results[:5]):
        print(f"{i + 1:<6} {result['activation']:<12} {result['batch_size']:<8} "
              f"{result['final_test_rmse']:<12.4f} {result['final_train_rmse']:<12.4f} "
              f"{result['training_time']:<10.2f}")

    # Training efficiency analysis
    print(f"\nTRAINING EFFICIENCY ANALYSIS:")
    print(f"{'Configuration':<20} {'RMSE/Time Ratio':<15} {'Total Time':<12}")
    print("-" * 50)

    efficiency_results = []
    for result in results:
        efficiency = result['final_test_rmse'] / max(result['training_time'], 0.1)  # Lower is better
        efficiency_results.append((result, efficiency))

    # Sort by efficiency (lower RMSE/time is better)
    efficiency_results.sort(key=lambda x: x[1])

    for result, efficiency in efficiency_results[:5]:
        config = f"{result['activation'][:4]}-{result['batch_size']}"
        print(f"{config:<20} {efficiency:<15.6f} {result['training_time']:<12.2f}")


# ---------------- Main Function ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Improved MPI-based SGD for Neural Networks')
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--max_iter", type=int, default=500, help="Maximum iterations")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden layer neurons")
    parser.add_argument("--hash_buckets", type=int, default=256, help="Hash buckets for categorical features")
    parser.add_argument("--train_split", type=float, default=0.7, help="Train/test split ratio")
    parser.add_argument("--max_rows", type=int, default=0, help="Max rows to process (0 for all)")
    # Improvements
    parser.add_argument("--momentum", type=float, default=0, help="Momentum coefficient")  # default - no momentum by setting it to 0
    parser.add_argument("--patience", type=int, default=1e9, help="Early stopping patience") # default - turn off patience by setting it to a large value > max_iter
    parser.add_argument("--use_batch_norm", action='store_true', default=False, help="Use batch normalization") # default - turn it off by setting it to False

    args = parser.parse_args()

    # Set random seed for reproducibility across processes
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Use rank-dependent seed for proper randomness in distributed setting
    np.random.seed(5208 + rank * 1000)

    mpi_print(rank, f"Starting improved neural network training with:")
    mpi_print(rank, f"  Learning rate: {args.lr}")
    mpi_print(rank, f"  Momentum: {args.momentum}")
    mpi_print(rank, f"  Hidden neurons: {args.hidden}")
    mpi_print(rank, f"  Batch normalization: {args.use_batch_norm}")
    mpi_print(rank, f"  Max iterations: {args.max_iter}")
    mpi_print(rank, f"  Patience: {args.patience}")

    results = run_experiments(args)

    if rank == 0:
        mpi_print(rank, "\nExperiment completed successfully!")
        mpi_print(rank, "Check 'improved_training_results.png' for visualization plots.")
        mpi_print(rank, "Results saved in pickle format for further analysis.")