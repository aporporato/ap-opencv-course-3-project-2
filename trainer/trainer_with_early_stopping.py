# # <font style="color:blue">Trainer Class</font>
#
# **This is a generic class for training loop.**
#
# Trainer class is equivalent to the `main` method. 
#
# In the main method, we were passing configurations, the model, optimizer, learning rate scheduler, and the number of epochs.  It was calling the method to get the train and test data loader. Using these, it is training and validating the model. During training and validation, it was also sending logs to TensorBoard and saving the model.
#
# The trainer class is doing the same in a more modular way so that we can experiment with different loss functions, different visualizers, different types of targets, etc. 
#

"""Unified class to make training pipeline for deep neural networks."""
import datetime
import os
from operator import itemgetter
from pathlib import Path
from typing import Union, Callable

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from .hooks import test_hook_default, train_hook_default
from .trainer import Trainer
from .visualizer import Visualizer


# ## <font style="color:blue">Trainer Class Methods and its Parameters</font>
#
# ### <font style="color:green">  \_\_init\_\_ </font>
#
# Setting different attributes.
#
# **Parameters:**
#
# - `model` : `nn.Module` - torch model to train
#
#         
# - `loader_train` : `torch.utils.DataLoader` - train dataset loader.
#
#     
# - `loader_test` : `torch.utils.DataLoader` - test dataset loader
#
#        
# - `loss_fn` : `callable` - loss function. In the main function, the cross-entropy loss was being used; here, we can pass the loss we want to use. For example, if we are solving a regression problem, we can not use cross-entropy loss. It is better to use RMS-loss.
#
#
#         
# - `metric_fn` : `callable` - evaluation metric function. In the main function, we had loss and accuracy as our evaluation metric. Here we can pass any evaluation metric. For example, in a detection problem, we need a precision-recall metric instead of accuracy.
#
#         
# - `optimizer` : `torch.optim.Optimizer` - Optimizer.
#
#         
# - `lr_scheduler` : `torch.optim.LrScheduler` - Learning Rate scheduler.
#
#         
# - `configuration` : `TrainerConfiguration` - a set of training process parameters.
#
# Here, we need a data iterator and target iterator separately, because we are writing a general trainer class. For example, for the detection problem for a single image, we might have `n`-number of objects and their coordinates. 
#
#         
# - `data_getter` : `Callable` - function object to extract input data from the sample prepared by dataloader.
#
#         
# - `target_getter` : `Callable` - function object to extract target data from the sample prepared by dataloader.
#
#         
# - `visualizer` : `Visualizer` - optional, shows metrics values (various backends are possible). We can pass the visualizer of our choice. For example, Matplotlib based visualizer, TensorBoard based, etc.
#
# It is also calling its method `_register_default_hooks` what this method does we will see next. In short, this is making sure that training and validation function is registered at the time of trainer class object initiation. 
#
#
# ### <font style="color:green"> _register_default_hooks </font>
#
# It is calling the another method `register_hook` to register training (`train_hook_default`) and validation (`test_hook_default`) functions. `train_hook_default` and `test_hook_default` are defined in the `hook`-module.  We will go in details in the module.
#
#
# ### <font style="color:green"> register_hook </font>
#
# It is updating the key-value pair of a dictionary, where the key is string and value is a callable function.
#
# **Parameters:**
#
# - `hook_type`: `string` - hook type. For example, wether the function will be used for train or test.
#
#
# - `hook_fn`: `callable` - hook function.
#
#
#
#
# ### <font style="color:green"> fit </font>
#
# Taking the number of epochs and training and validating the model. It is also adding logs to the visualizer. 
#
# **Parameters:**
#
# - `epochs`: `int` - number of epochs to train model.
#

class TrainerWithEarlyStopping(Trainer):  # pylint: disable=too-many-instance-attributes
    """ Generic class for training loop.

    Parameters
    ----------
    model : nn.Module
        torch model to train
    loader_train : torch.utils.DataLoader
        train dataset loader.
    loader_test : torch.utils.DataLoader
        test dataset loader
    loss_fn : callable
        loss function
    metric_fn : callable
        evaluation metric function
    optimizer : torch.optim.Optimizer
        Optimizer
    lr_scheduler : torch.optim.LrScheduler
        Learning Rate scheduler
    configuration : TrainerConfiguration
        a set of training process parameters
    data_getter : Callable
        function object to extract input data from the sample prepared by dataloader.
    target_getter : Callable
        function object to extract target data from the sample prepared by dataloader.
    visualizer : Visualizer, optional
        shows metrics values (various backends are possible)
    # """

    def __init__(  # pylint: disable=too-many-arguments
            self,
            model: torch.nn.Module,
            loader_train: torch.utils.data.DataLoader,
            loader_test: torch.utils.data.DataLoader,
            loss_fn: Callable,
            metric_fn: Callable,
            optimizer: torch.optim.Optimizer,
            lr_scheduler: Callable,
            device: Union[torch.device, str] = "cuda",
            model_saving_frequency: int = 1,
            save_dir: Union[str, Path] = "checkpoints",
            model_name_prefix: str = "model",
            data_getter: Callable = itemgetter("image"),
            target_getter: Callable = itemgetter("target"),
            stage_progress: bool = True,
            visualizer: Union[Visualizer, None] = None,
            get_key_metric: Callable = itemgetter("top1"),
    ):
        super(self.__class__, self).__init__(model,
                                             loader_train,
                                             loader_test,
                                             loss_fn,
                                             metric_fn,
                                             optimizer,
                                             lr_scheduler,
                                             device,
                                             model_saving_frequency,
                                             save_dir,
                                             model_name_prefix,
                                             data_getter,
                                             target_getter,
                                             stage_progress,
                                             visualizer,
                                             get_key_metric)

    def fit(self, epochs):
        """ Fit model method.

        Arguments:
            epochs (int): number of epochs to train model.
        """
        iterator = tqdm(range(epochs), dynamic_ncols=True)
        train_history = []
        last_best_loss = torch.tensor(np.inf)
        epochs_from_last_best_loss = 0
        for epoch in iterator:
            output_train = self.hooks["train"](
                self.model,
                self.loader_train,
                self.loss_fn,
                self.optimizer,
                self.device,
                self.use_max_norm_normalization,
                prefix="[{}/{}]".format(epoch + 1, epochs),
                stage_progress=self.stage_progress,
                data_getter=self.data_getter,
                target_getter=self.target_getter
            )
            output_test = self.hooks["test"](
                self.model,
                self.loader_test,
                self.loss_fn,
                self.metric_fn,
                self.device,
                prefix="[{}/{}]".format(epoch + 1, epochs),
                stage_progress=self.stage_progress,
                data_getter=self.data_getter,
                target_getter=self.target_getter,
                get_key_metric=self.get_key_metric
            )
            if self.visualizer:
                self.visualizer.update_charts(
                    None, output_train['loss'], output_test['metric'], output_test['loss'],
                    self.optimizer.param_groups[0]['lr'], epoch
                )

            self.metrics['epoch'].append(epoch)
            self.metrics['train_loss'].append(output_train['loss'])
            self.metrics['test_loss'].append(output_test['loss'])
            self.metrics['test_metric'].append(output_test['metric'])

            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(output_train['loss'])
                else:
                    self.lr_scheduler.step()

            if self.hooks["end_epoch"] is not None:
                self.hooks["end_epoch"](iterator, epoch, output_train, output_test)

            if (epoch + 1) % self.model_saving_frequency == 0:
                os.makedirs(self.save_dir, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_dir, self.model_name_prefix) + str(datetime.datetime.now())
                )

            if self.hooks["early_stop"] is not None:
                stop, train_history, last_best_loss, epochs_from_last_best_loss = \
                    self.hooks["early_stop"](epochs, train_history, output_train['loss'], last_best_loss,
                                             epochs_from_last_best_loss)
                if stop:
                    return self.metrics

        return self.metrics

    def _register_default_hooks(self):
        self.register_hook("train", train_hook_default)
        self.register_hook("test", test_hook_default)
        self.register_hook("end_epoch", None)
        self.register_hook("early_stop", None)