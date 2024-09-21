import os
import torch
import torch.nn as nn
import sys
import tqdm.auto
import itertools
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import Tensor
from typing import List, NamedTuple
from typing import Any, Callable, Optional


class MLP(nn.Module):
    def __init__(self, n_features:int, n_targets:int, hidden_dim: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        layers = []
        
        for _ in range(n_layers):
            in_dim = n_features if not len(layers) else hidden_dim
            
            layers += [
                nn.Linear(in_dim, hidden_dim), 
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
                nn.ReLU()
            ]
        
        layers += [
            nn.Linear(hidden_dim, n_targets),
            nn.Softmax(dim=1)
        ]
        
        self.sequencial = nn.Sequential(*layers)
    
    def forward(self, x: Tensor):
        x = x.view(x.size(0), -1)
        return self.sequencial(x)


class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch: the loss
    and number of correct classifications.
    """
    loss: float
    num_correct: int

class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch: the loss per batch
    and accuracy on the dataset (train or test).
    """
    losses: List[float]
    accuracy: float

class FitResult(NamedTuple):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch and the accuracies are per epoch.
    """
    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]
    best_loss: float
    

class Trainer():
    """
    A class for training a pytorch model.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: loss function to compare true labels and predictions.
        :param optimizer: optimizer to optimize model weights.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

        if self.device:
            model.to(self.device)

    def fit(
        self,
        dl_train: DataLoader,
        dl_test: DataLoader,
        save_model: bool,
        num_epochs: int,
        early_stopping: int = None,
        print_every: int = 1,
        **kw,
    ) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param save_model: if set to true then saves the model as .pt file.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        epochs_without_improvement = 0

        train_losses, train_acc, test_losses, test_acc = [], [], [], []
        best_loss = np.inf

        for epoch in range(num_epochs):
            verbose = False
            if print_every > 0 and (epoch % print_every == 0 or epoch == num_epochs - 1):
                verbose = True
                
            self._print(f"--- EPOCH {epoch+1}/{num_epochs} ---", verbose)

            train_result = self.train_epoch(dl_train, **kw)
            test_result = self.test_epoch(dl_test, **kw)
            
            train_losses.append(np.mean(train_result.losses))
            train_acc.append(train_result.accuracy)
            
            test_loss = np.mean(test_result.losses)
            test_losses.append(test_loss)
            test_acc.append(test_result.accuracy)
            
            if test_loss < best_loss:
                best_loss = test_loss
                if save_model:
                    self.save_model()
            else:
                epochs_without_improvement += 1
                
                if epochs_without_improvement == early_stopping:
                    break

        return FitResult(actual_num_epochs, train_losses, train_acc, test_losses, test_acc, best_loss)

    def save_model(self):
        """
        Saves the model in it's current state to a file with the given name (treated
        as a relative path).
        :param checkpoint_filename: File name or relative path to save to.
        """
        model_script = torch.jit.script(self.model)
        model_script.save('results/best_model_script.pt')
        print(f"\n*** Saved best model")

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train()
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.eval()
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and updates weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        X, y = batch
        
        X = X.to(self.device)
        y = y.to(self.device)
        
        self.optimizer.zero_grad()
        
        y_pred_scores = self.model(X)
        loss = self.loss_fn(y_pred_scores, y)
        loss.backward()
        self.optimizer.step()
        
        y_pred = torch.argmax(y_pred_scores, dim=1)
        num_correct = torch.sum(y == y_pred)
        
        return BatchResult(loss.item(), num_correct.item())

    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        X, y = batch
        
        X = X.to(self.device)
        y = y.to(self.device)
        
        with torch.no_grad():
            y_pred_scores = self.model(X)
            loss = self.loss_fn(y_pred_scores, y)
            
            y_pred = torch.argmax(y_pred_scores, dim=1)
            num_correct = torch.sum(y == y_pred)
            
        return BatchResult(loss.item(), num_correct.item())

    @staticmethod
    def _print(message, verbose=True):
        """Simple wrapper around print to make it conditional"""
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
        dl: DataLoader,
        forward_fn: Callable[[Any], BatchResult],
        verbose=True,
        max_batches=None,
    ) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_fn = tqdm.auto.tqdm
            pbar_file = sys.stdout
        else:
            pbar_fn = tqdm.tqdm
            pbar_file = open(os.devnull, "w")

        pbar_name = forward_fn.__name__
        with pbar_fn(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f"{pbar_name} ({batch_res.loss:.3f})")
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            accuracy = 100.0 * num_correct / num_samples
            pbar.set_description(
                f"{pbar_name} "
                f"(Avg. Loss {avg_loss:.3f}, "
                f"Accuracy {accuracy:.1f})"
            )

        if not verbose:
            pbar_file.close()

        return EpochResult(losses=losses, accuracy=accuracy)
    
    
def plot_fit(
    fit_res: FitResult,
    fig=None,
    log_loss=False,
    legend=None,
    train_test_overlay: bool = False,
):
    """
    Plots a FitResult object.
    Creates four plots: train loss, test loss, train acc, test acc.
    :param fit_res: The fit result to plot.
    :param fig: A figure previously returned from this function. If not None,
        plots will the added to this figure.
    :param log_loss: Whether to plot the losses in log scale.
    :param legend: What to call this FitResult in the legend.
    :param train_test_overlay: Whether to overlay train/test plots on the same axis.
    :return: The figure.
    """
    if fig is None:
        nrows = 1 if train_test_overlay else 2
        ncols = 2
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(8 * ncols, 5 * nrows),
            sharex="col",
            sharey=False,
            squeeze=False,
        )
        axes = axes.reshape(-1)
    else:
        axes = fig.axes

    for ax in axes:
        for line in ax.lines:
            if line.get_label() == legend:
                line.remove()

    p = itertools.product(enumerate(["train", "test"]), enumerate(["loss", "acc"]))
    for (i, train_test), (j, loss_acc) in p:

        ax = axes[j if train_test_overlay else i * 2 + j]

        attr = f"{train_test}_{loss_acc}"
        data = getattr(fit_res, attr)
        label = train_test if train_test_overlay else legend
        h = ax.plot(np.arange(1, len(data) + 1), data, label=label)
        ax.set_title(loss_acc)

        if loss_acc == "loss":
            ax.set_xlabel("Iteration #")
            ax.set_ylabel("Loss")
            if log_loss:
                ax.set_yscale("log")
                ax.set_ylabel("Loss (log)")
        else:
            ax.set_xlabel("Epoch #")
            ax.set_ylabel("Accuracy (%)")

        if legend or train_test_overlay:
            ax.legend()
        ax.grid(True)

    return fig, axes