import time
import os
import json
import sys
import wandb
from collections import defaultdict, OrderedDict
from omegaconf import OmegaConf as om
from omegaconf import DictConfig
import jax.numpy as jnp
import jax.random
from jaxtyping import Array
from typing import Callable, Dict, Optional, Iterator, Any
from flax.training.train_state import TrainState
from flax.training import checkpoints
from flax import jax_utils
from functools import partial


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, total_iters, meter_names, prefix=""):
        self.iter_fmtstr = self._get_iter_fmtstr(total_iters)
        self.meters = OrderedDict({mn: AverageMeter(mn, ':6.3f')
                                   for mn in meter_names})
        self.prefix = prefix

    def update(self, n=1, **kwargs):
        for k, v in kwargs.items():
            self.meters[k].update(v, n=n)

    def display(self, iteration):
        entries = [self.prefix + self.iter_fmtstr.format(iteration)]
        entries += [str(meter) for meter in self.meters.values()]
        print('\t'.join(entries))

    def _get_iter_fmtstr(self, total_iters):
        num_digits = len(str(total_iters // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(total_iters) + ']'


@partial(jax.jit, static_argnums=(1,))
def reshape_batch_per_device(x, num_devices):
    return jax.tree_util.tree_map(partial(reshape_array_per_device, num_devices=num_devices), x)


def reshape_array_per_device(x, num_devices):
    batch_size_per_device, ragged = divmod(x.shape[0], num_devices)
    if ragged:
        msg = "batch size must be divisible by device count, got {} and {}."
        raise ValueError(msg.format(x.shape[0], num_devices))
    return x.reshape((num_devices, batch_size_per_device, ) + (x.shape[1:]))


class TrainerModule:
    def __init__(
            self,
            train_state: TrainState,
            training_step_fn: Callable,
            evaluation_step_fn: Callable,
            world_size: int,
            config: DictConfig,
            debug: bool = False,
    ):
        """
        A basic Trainer module summarizing most common training functionalities
        like logging, model initialization, training loop, etc.

        Atributes:
          model_class: The class of the model that should be trained.
          model_hparams: A dictionary of all hyperparameters of the model. Is
            used as input to the model when created.
          optimizer_hparams: A dictionary of all hyperparameters of the optimizer.
            Used during initialization of the optimizer.
          exmp_input: Input to the model for initialization and tabulate.
          seed: Seed to initialize PRNG.
          logger_params: A dictionary containing the specification of the logger.
          enable_progress_bar: If False, no progress bar is shown.
          debug: If True, no jitting is applied. Can be helpful for debugging.
        """
        super().__init__()
        self.train_state = train_state
        self.train_step = training_step_fn
        self.eval_step = evaluation_step_fn

        self.world_size = world_size
        self.log_config = config.logging
        self.debug = debug
        self.epoch_idx = 0
        self.best_eval_metrics = {}

        # logger details
        self.log_dir = os.path.join(self.log_config.log_dir)
        print('[*] Logging to', self.log_dir)

        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.isdir(os.path.join(self.log_dir, 'metrics')):
            os.makedirs(os.path.join(self.log_dir, 'metrics'))
        if not os.path.isdir(os.path.join(self.log_dir, 'checkpoints')):
            os.makedirs(os.path.join(self.log_dir, 'checkpoints'))

        num_parameters = int(sum(
            [arr.size for arr in jax.tree_flatten(self.train_state.params)[0]
             if isinstance(arr, Array)]
        ) / self.world_size)
        print("[*] Number of model parameters:", num_parameters)

        if self.log_config.wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                dir=self.log_config.log_dir,
                project=self.log_config.project,
                entity=self.log_config.entity,
                config=om.to_container(config, resolve=True))
            wandb.config.update({'SLURM_JOB_ID': os.getenv('SLURM_JOB_ID')})

            # log number of parameters
            wandb.run.summary['Num parameters'] = num_parameters
            wandb.define_metric(self.log_config.summary_metric, summary='max')

    def train_model(
            self,
            train_loader: Iterator,
            val_loader: Iterator,
            num_epochs: int,
            dropout_key: Array,
            test_loader: Optional[Iterator] = None,
    ) -> Dict[str, Any]:
        """
        Starts a training loop for the given number of epochs.

        Args:
          train_loader: Data loader of the training set.
          val_loader: Data loader of the validation set.
          test_loader: If given, best model will be evaluated on the test set.
          num_epochs: Number of epochs for which to train the model.

        Returns:
          A dictionary of the train, validation and evt. test metrics for the
          best model on the validation set.
        """

        # Prepare training loop
        self.on_training_start()

        for epoch_idx in range(1, num_epochs+1):
            self.epoch_idx = epoch_idx

            # run training step for this epoch
            train_metrics = self.train_epoch(train_loader, dropout_key)

            self.on_training_epoch_end(train_metrics)

            # Validation every N epochs
            eval_metrics = self.eval_model(
                val_loader,
                log_prefix='Performance/Validation',
            )

            self.on_validation_epoch_end(eval_metrics)

            if self.log_config.wandb:
                from optax import MultiStepsState
                wandb_metrics = {'Performance/epoch': epoch_idx}
                wandb_metrics.update(train_metrics)
                wandb_metrics.update(eval_metrics)
                if isinstance(self.train_state.opt_state, MultiStepsState):
                    lr = self.train_state.opt_state.inner_opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate'].item()
                else:
                    lr = self.train_state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate'].item()
                wandb_metrics['learning rate'] = lr
                wandb.log(wandb_metrics)

        # Test best model if possible
        if test_loader is not None:
            self.load_model()
            test_metrics = self.eval_model(
                test_loader,
                log_prefix='Performance/Test',
            )
            self.save_metrics('test', test_metrics)
            self.best_eval_metrics.update(test_metrics)

            if self.log_config.wandb:
                 wandb.log(test_metrics)

            print('-' * 89)
            print('| End of Training |')
            print('| Test  Metrics |',
                  ' | '.join([f"{k.split('/')[1].replace('Test','')}: {v:5.2f}" for k, v in test_metrics.items() if 'Test' in k]))
            print('-' * 89)

        return self.best_eval_metrics

    def train_epoch(self, train_loader: Iterator, dropout_key) -> Dict[str, Any]:
        """
        Trains a model for one epoch.

        Args:
          train_loader: Data loader of the training set.

        Returns:
          A dictionary of the average training metrics over all batches
          for logging.
        """

        # Train model for one epoch, and log avg loss and accuracy
        metrics = defaultdict(float)
        running_metrics = defaultdict(float)
        num_batches = 0
        num_train_batches = len(train_loader)
        start_time = time.time()
        epoch_start_time = start_time

        # set up intra epoch logging
        log_interval = self.log_config.interval

        for i, batch in enumerate(train_loader):
            # skip batches with empty sequences which might randomly occur due to data augmentation
            _, _, _, lengths = batch
            if jnp.any(lengths == 0):
                continue

            num_batches += 1
            if self.world_size > 1:
                step_key, dropout_key = jax.vmap(jax.random.split, in_axes=0, out_axes=1)(dropout_key)
                step_key = jax.vmap(jax.random.fold_in)(step_key, jnp.arange(self.world_size))
                batch = reshape_batch_per_device(batch, self.world_size)
            else:
                step_key, dropout_key = jax.random.split(dropout_key)

            self.train_state, step_metrics = self.train_step(self.train_state, batch, step_key)

            # exit from training if loss is nan
            if jnp.isnan(step_metrics['loss']).any():
                print("EXITING TRAINING DUE TO NAN LOSS")
                break

            # record metrics
            for key in step_metrics:
                metrics['Performance/Training ' + key] += step_metrics[key]
                running_metrics['Performance/Training ' + key] += step_metrics[key]

            # print metrics to terminal
            if (i + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                start_time = time.time()
                print(f'| epoch {self.epoch_idx} | {i + 1}/{num_train_batches} batches | ms/batch {elapsed * 1000 / log_interval:5.2f} |',
                      ' | '.join([f'{k}: {jnp.mean(v).item() / log_interval:5.2f}' for k, v in running_metrics.items()]))
                for key in step_metrics:
                    running_metrics['Performance/Training ' + key] = 0

        metrics = {key: jnp.mean(metrics[key] / num_batches).item() for key in metrics}
        metrics['epoch_time'] = time.time() - epoch_start_time
        return metrics

    def eval_model(
            self,
            data_loader: Iterator,
            log_prefix: Optional[str] = '',
    ) -> Dict[str, Any]:
        """
        Evaluates the model on a dataset.

        Args:
          data_loader: Data loader of the dataset to evaluate on.
          init_hidden_fn: function that initializes the hidden state
          log_prefix: Prefix to add to all metrics (e.g. 'val/' or 'test/')

        Returns:
          A dictionary of the evaluation metrics, averaged over data points
          in the dataset.
        """

        # Test model on all images of a data loader and return avg loss
        metrics = defaultdict(float)
        num_batches = 0

        for i, batch in enumerate(iter(data_loader)):

            if self.world_size > 1:
                batch = reshape_batch_per_device(batch, self.world_size)

            self.train_state, step_metrics = self.eval_step(self.train_state, batch)

            for key in step_metrics:
                metrics[key] += step_metrics[key]
            num_batches += 1

        prefix = log_prefix + ' ' if log_prefix else ''
        metrics = {(prefix + key): jnp.mean(metrics[key] / num_batches).item() for key in metrics}
        return metrics

    def is_new_model_better(self, new_metrics: Dict[str, Any], old_metrics: Dict[str, Any]) -> bool:
        """
        Compares two sets of evaluation metrics to decide whether the
        new model is better than the previous ones or not.

        Args:
          new_metrics: A dictionary of the evaluation metrics of the new model.
          old_metrics: A dictionary of the evaluation metrics of the previously
            best model, i.e. the one to compare to.

        Returns:
          True if the new model is better than the old one, and False otherwise.
        """
        if len(old_metrics) == 0:
            return True
        for key, is_larger in [('val/val_metric', False), ('Performance/Validation accuracy', True), ('Performance/Validation loss', False)]:
            if key in new_metrics:
                if is_larger:
                    return new_metrics[key] > old_metrics[key]
                else:
                    return new_metrics[key] < old_metrics[key]
        assert False, f'No known metrics to log on: {new_metrics}'

    def save_metrics(self, filename: str, metrics: Dict[str, Any]):
        """
        Saves a dictionary of metrics to file. Can be used as a textual
        representation of the validation performance for checking in the terminal.

        Args:
          filename: Name of the metrics file without folders and postfix.
          metrics: A dictionary of metrics to save in the file.
        """
        with open(os.path.join(self.log_dir, f'metrics/{filename}.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

    def save_model(self):
        if self.world_size > 1:
            state = jax_utils.unreplicate(self.train_state)
        else:
            state = self.train_state
        checkpoints.save_checkpoint(
            ckpt_dir=os.path.join(self.log_dir, 'checkpoints'),
            target=state,
            step=state.step,
            overwrite=True,
            keep=1
        )
        del state

    def load_model(self):
        if self.world_size > 1:
            state = jax_utils.unreplicate(self.train_state)
            raw_restored = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(self.log_dir, 'checkpoints'), target=state)
            self.train_state = jax_utils.replicate(raw_restored)
            del state
        else:
            self.train_state = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(self.log_dir, 'checkpoints'), target=self.train_state)

    def on_training_start(self):
        """
        Method called before training is started. Can be used for additional
        initialization operations etc.
        """
        pass

    def on_training_epoch_end(self, train_metrics):
        """
        Method called at the end of each training epoch. Can be used for additional
        logging or similar.
        """
        print('-' * 89)
        print(f"| end of epoch {self.epoch_idx:3d} | time per epoch: {train_metrics['epoch_time']:5.2f}s |")
        print('| Train Metrics |', ' | '.join(
            [f"{k.split('/')[1].replace('Training ', '')}: {v:5.2f}" for k, v in train_metrics.items() if
             'Train' in k]))

        # check metrics for nan values and possibly exit training
        if jnp.isnan(train_metrics['Performance/Training loss']).item():
            print("EXITING TRAINING DUE TO NAN LOSS")
            sys.exit(1)

    def on_validation_epoch_end(self, eval_metrics: Dict[str, Any]):
        """
        Method called at the end of each validation epoch. Can be used for additional
        logging and evaluation.

        Args:
          eval_metrics: A dictionary of the validation metrics. New metrics added to
            this dictionary will be logged as well.
        """
        print('| Eval  Metrics |', ' | '.join(
            [f"{k.split('/')[1].replace('Validation ', '')}: {v:5.2f}" for k, v in eval_metrics.items() if
             'Validation' in k]))
        print('-' * 89)

        self.save_metrics(f'eval_epoch_{str(self.epoch_idx).zfill(3)}', eval_metrics)

        # Save best model
        if self.is_new_model_better(eval_metrics, self.best_eval_metrics):
            best_eval_metrics = eval_metrics
            self.save_model()
            self.save_metrics('best_eval', eval_metrics)
