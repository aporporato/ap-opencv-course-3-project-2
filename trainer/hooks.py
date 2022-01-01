# # <font style="color:blue">Hooks for Trainer Class</font>
#
# This module implements several hooks (helper functions) for the Trainer class. 
#
# This module has the following method implemented:
#
# - `train_hook_default`
#
#
# - `test_hook_default`
#
#
# - `end_epoch_hook_classification`
#

"""Implementation of several hooks that used in a Trainer class."""
from operator import itemgetter

import torch
from tqdm.auto import tqdm

from .utils import AverageMeter


# Max-Norm Regularization for Dropout
#  (from https://discuss.pytorch.org/t/how-to-correctly-implement-in-place-max-norm-constraint/96769
#    and https://github.com/kevinzakka/pytorch-goodies#max-norm-constraint )
def _max_norm(model, max_val=1.0, eps=1e-8):
    with torch.no_grad():
        for param in model.parameters():
            norm = param.norm(2, dim=0, keepdim=True).clamp(min=eps)
            desired = torch.clamp(norm, max=max_val)
            param *= (desired / norm)


# ## <font style="color:green">train_hook_default</font>
#
# Default train loop function for single epoch. 
#
# **Parameters:**
#
# - `model` (`nn.Module`): torch model which will be train.
#
#
# - `loader` (`torch.utils.DataLoader`): dataset loader.
#
#
# - `loss_fn` (`callable`): loss function.
#
#
# - `optimizer` (`torch.optim.Optimizer`): Optimizer.
#
#
# - `device` (`str`): Specifies device at which samples will be uploaded.
#
#
# - `use_max_norm_normalization` (`bool`): should be True if the model has Dropout layers.
#
#
# - `data_getter` (`Callable`): function object to extract input data from the sample prepared by dataloader.
#
#
# - `target_getter` (`Callable`): function object to extract target data from the sample prepared by dataloader.
#
#
# - `iterator_type` (`iterator`): type of the iterator. e.g. tqdm
#
#
# - `prefix` (`string`): prefix which will be add to the description string of progress bar.
#
#
# - `stage_progress` (`bool`): if True then progress bar will be show.
#

def train_hook_default(
        model,
        loader,
        loss_fn,
        optimizer,
        device,
        use_max_norm_normalization=False,
        data_getter=itemgetter("image"),
        target_getter=itemgetter("mask"),
        iterator_type=tqdm,
        prefix="",
        stage_progress=False
):
    """ Default train loop function.

    Arguments:
        model (nn.Module): torch model which will be train.
        loader (torch.utils.DataLoader): dataset loader.
        loss_fn (callable): loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (str): Specifies device at which samples will be uploaded.
        use_max_norm_normalization (bool): should be True if the model has Dropout layers.
        data_getter (Callable): function object to extract input data from the sample prepared by dataloader.
        target_getter (Callable): function object to extract target data from the sample prepared by dataloader.
        iterator_type (iterator): type of the iterator.
        prefix (string): prefix which will be add to the description string.
        stage_progress (bool): if True then progress bar will be show.

    Returns:
        Dictionary of output metrics with keys:
            loss: average loss.
    """
    model = model.train()
    iterator = iterator_type(loader, disable=not stage_progress, dynamic_ncols=True)
    loss_avg = AverageMeter()
    for i, sample in enumerate(iterator):
        if use_max_norm_normalization:
            _max_norm(model)
        optimizer.zero_grad()
        inputs = data_getter(sample).to(device)
        targets = target_getter(sample).to(device)
        predicts = model(inputs)
        loss = loss_fn(predicts, targets)
        loss.backward()
        optimizer.step()
        loss_avg.update(loss.item())
        status = "{0}[Train][{1}] Loss_avg: {2:.5}, Loss: {3:.5}, LR: {4:.5}".format(
            prefix, i, loss_avg.avg, loss_avg.val, optimizer.param_groups[0]["lr"]
        )
        iterator.set_description(status)
    return {"loss": loss_avg.avg}


# ## <font style="color:green">test_hook_default</font>
#
# Default test loop function for single epoch. 
#
# **Parameters:**
#
# - `model` (`nn.Module`): torch model which will be train.
#
#
# - `loader` (`torch.utils.DataLoader`): dataset loader.
#
#
# - `device` (`str`): Specifies device at which samples will be uploaded.
#
#
# - `data_getter` (`Callable`): function object to extract input data from the sample prepared by dataloader.
#
#
# - `target_getter` (`Callable`): function object to extract target data from the sample prepared by dataloader.
#
#
# - `iterator_type` (`iterator`): type of the iterator. e.g. tqdm
#
#
# - `prefix` (`string`): prefix which will be add to the description string of progress bar.
#
#
# - `stage_progress` (`bool`): if True then progress bar will be show.
#

def test_hook_default(
        model,
        loader,
        loss_fn,
        metric_fn,
        device,
        data_getter=itemgetter("image"),
        target_getter=itemgetter("mask"),
        iterator_type=tqdm,
        prefix="",
        stage_progress=False,
        get_key_metric=itemgetter("accuracy")
):
    """ Default test loop function.

    Arguments:
        model (nn.Module): torch model which will be train.
        loader (torch.utils.DataLoader): dataset loader.
        loss_fn (callable): loss function.
        metric_fn (callable): evaluation metric function.
        device (str): Specifies device at which samples will be uploaded.
        data_getter (Callable): function object to extract input data from the sample prepared by dataloader.
        target_getter (Callable): function object to extract target data from the sample prepared by dataloader.
        iterator_type (iterator): type of the iterator.
        prefix (string): prefix which will be add to the description string.
        stage_progress (bool): if True then progress bar will be show.

    Returns:
        Dictionary of output metrics with keys:
            metric: output metric.
            loss: average loss.
    """
    model = model.eval()
    iterator = iterator_type(loader, disable=not stage_progress, dynamic_ncols=True)
    loss_avg = AverageMeter()
    metric_fn.reset()
    for i, sample in enumerate(iterator):
        inputs = data_getter(sample).to(device)
        targets = target_getter(sample).to(device)
        with torch.no_grad():
            predict = model(inputs)
            loss = loss_fn(predict, targets)
        loss_avg.update(loss.item())
        predict = predict.softmax(dim=1).detach()
        metric_fn.update_value(predict, targets)
        status = "{0}[Test][{1}] Loss_avg: {2:.5}".format(prefix, i, loss_avg.avg)
        if get_key_metric is not None:
            status = status + ", Metric_avg: {0:.5}".format(get_key_metric(metric_fn.get_metric_value()))
        iterator.set_description(status)
    output = {"metric": metric_fn.get_metric_value(), "loss": loss_avg.avg}
    return output


# ## <font style="color:green">end_epoch_hook_classification</font>
#
# To show end epoch progress bar.
#
# **Parameters:**
#
# - `iterator` (`iter`): iterator.
#
#
# - `epoch` (`int`): number of epoch to store.
#
#
# - `output_train` (`dict`): description of the train stage.
#
#
# - `output_test` (`dict`): description of the test stage.

def end_epoch_hook_classification(iterator, epoch, output_train, output_test):
    """ Default end_epoch_hook for classification tasks.
    Arguments:
        iterator (iter): iterator.
        epoch (int): number of epoch to store.
        output_train (dict): description of the train stage.
        output_test (dict): description of the test stage.
    """
    if hasattr(iterator, "set_description"):
        iterator.set_description(
            "epoch: {0}, test_top1: {1:.5}, train_loss: {2:.5}, test_loss: {3:.5}".format(
                epoch, output_test["metric"]["top1"], output_train["loss"], output_test["loss"]
            )
        )


# ## <font style="color:green">early_stop_hook_classification</font>
#
# Check for training early stop conditions.
#
# **Parameters:**
#
# - `epochs` (`int`): total training epochs number.
#
#
# - `train_history` (`list[float]`): history of loss on training set.
#
#
# - `current_loss` (`Number`): current epoch training loss.
#
#
# - `last_best_loss` (`int`): best training loss up until the current epoch loss.
#
#
# - `epochs_from_last_best_loss` (`int`): number of epoch from the one with the best loss.

def early_stop_hook_classification(epochs,
                                   train_history,
                                   current_loss,
                                   last_best_loss,
                                   epochs_from_last_best_loss):
    """ Default end_epoch_hook for classification tasks.
    Arguments:
        epochs (int): training epochs number.
        train_history (list[float]): history of loss on training set.
        current_loss (float): current epoch training loss.
        last_best_loss (int): best training loss up until the current epoch loss.
        epochs_from_last_best_loss (int): number of epoch from the one with the best loss.
    """
    stop = False
    if len(train_history) >= epochs // 3:
        # If validation loss do not decrease for a while, stop (the net is overfitting)
        if current_loss < last_best_loss:
            epochs_from_last_best_loss = 0
            last_best_loss = current_loss
        else:
            epochs_from_last_best_loss = epochs_from_last_best_loss + 1
        if epochs_from_last_best_loss >= epochs // 10:
            print("Early Stopping at epoch {}/{}: best epoch was {} epochs ago (loss was {})"
                  .format(len(train_history), epochs, epochs_from_last_best_loss, last_best_loss))
            stop = True
    else:
        if last_best_loss > current_loss:
            last_best_loss = current_loss
    train_history.append(current_loss)
    return stop, train_history, last_best_loss, epochs_from_last_best_loss
