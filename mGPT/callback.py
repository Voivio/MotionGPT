import os
from pathlib import Path
from typing import Union

import wandb
from .render.matplot.plot_3d_global import draw_to_batch
from .data.humanml.scripts.motion_process import recover_from_ric

import imageio.v2 as imageio
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import Callback, RichProgressBar, ModelCheckpoint


def build_callbacks(cfg, logger=None, phase='test', **kwargs):
    callbacks = []
    logger = logger

    # Rich Progress Bar
    callbacks.append(progressBar())

    # Checkpoint Callback
    if phase == 'train':
        callbacks.extend(getCheckpointCallback(cfg, logger=logger, **kwargs))

    if cfg.TRAIN.STAGE == 'vae':
        datamodule = kwargs.get('datamodule', None)
        if datamodule is None:
            raise ValueError("A Data module is required for VAE visualization callback.")

        callbacks.append(VisualizationCallback(
            save_dir=Path(cfg.FOLDER_EXP) / "gifs",
            datamodule=datamodule,
            interval=cfg.LOGGER.VAL_EVERY_STEPS * 10,
            n_batches_to_viz=10,
            logger_log=True
        ))

    return callbacks


def getCheckpointCallback(cfg, logger=None, **kwargs):
    callbacks = []
    # Logging
    metric_monitor = {
        "loss_total": "total/train",
        "Train_jf": "recons/text2jfeats/train",
        "Val_jf": "recons/text2jfeats/val",
        "Train_rf": "recons/text2rfeats/train",
        "Val_rf": "recons/text2rfeats/val",
        "APE root": "Metrics/APE_root",
        "APE mean pose": "Metrics/APE_mean_pose",
        "AVE root": "Metrics/AVE_root",
        "AVE mean pose": "Metrics/AVE_mean_pose",
        "R_TOP_1": "Metrics/R_precision_top_1",
        "R_TOP_2": "Metrics/R_precision_top_2",
        "R_TOP_3": "Metrics/R_precision_top_3",
        "gt_R_TOP_3": "Metrics/gt_R_precision_top_3",
        "FID": "Metrics/FID",
        "gt_FID": "Metrics/gt_FID",
        "Diversity": "Metrics/Diversity",
        "MM dist": "Metrics/Matching_score",
        "Accuracy": "Metrics/accuracy",
    }
    callbacks.append(
        progressLogger(logger, metric_monitor=metric_monitor, log_every_n_steps=1))

    # Save 10 latest checkpoints
    checkpointParams = {
        'dirpath': os.path.join(cfg.FOLDER_EXP, "checkpoints"),
        'filename': "{epoch}",
        'monitor': "step",
        'mode': "max",
        'every_n_epochs': cfg.LOGGER.VAL_EVERY_STEPS,
        'save_top_k': 8,
        'save_last': True,
        'save_on_train_epoch_end': True
    }
    callbacks.append(ModelCheckpoint(**checkpointParams))

    # Save checkpoint every n*10 epochs
    checkpointParams.update({
        'every_n_epochs':
            cfg.LOGGER.VAL_EVERY_STEPS * 10,
        'save_top_k':
            -1,
        'save_last':
            False
    })
    callbacks.append(ModelCheckpoint(**checkpointParams))

    metrics = cfg.METRIC.TYPE
    metric_monitor_map = {
        'TemosMetric': {
            'Metrics/APE_root': {
                'abbr': 'APEroot',
                'mode': 'min'
            },
        },
        'TM2TMetrics': {
            'Metrics/FID': {
                'abbr': 'FID',
                'mode': 'min'
            },
            'Metrics/R_precision_top_3': {
                'abbr': 'R3',
                'mode': 'max'
            }
        },
        'MRMetrics': {
            'Metrics/MPJPE': {
                'abbr': 'MPJPE',
                'mode': 'min'
            }
        },
        'HUMANACTMetrics': {
            'Metrics/Accuracy': {
                'abbr': 'Accuracy',
                'mode': 'max'
            }
        },
        'UESTCMetrics': {
            'Metrics/Accuracy': {
                'abbr': 'Accuracy',
                'mode': 'max'
            }
        },
        'UncondMetrics': {
            'Metrics/FID': {
                'abbr': 'FID',
                'mode': 'min'
            }
        }
    }

    checkpointParams.update({
        'every_n_epochs': cfg.LOGGER.VAL_EVERY_STEPS,
        'save_top_k': 1,
    })

    for metric in metrics:
        if metric in metric_monitor_map.keys():
            metric_monitors = metric_monitor_map[metric]

            # Delete R3 if training VAE
            if cfg.TRAIN.STAGE == 'vae' and metric == 'TM2TMetrics':
                del metric_monitors['Metrics/R_precision_top_3']

            for metric_monitor in metric_monitors:
                checkpointParams.update({
                    'filename':
                        metric_monitor_map[metric][metric_monitor]['mode']
                        + "-" +
                        metric_monitor_map[metric][metric_monitor]['abbr']
                        + "{ep}",
                    'monitor':
                        metric_monitor,
                    'mode':
                        metric_monitor_map[metric][metric_monitor]['mode'],
                })
                callbacks.append(
                    ModelCheckpoint(**checkpointParams))
    return callbacks


class progressBar(RichProgressBar):
    def __init__(self, ):
        super().__init__()

    def get_metrics(self, trainer, model):
        # Don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


class progressLogger(Callback):
    def __init__(self,
                 logger,
                 metric_monitor: dict,
                 precision: int = 3,
                 log_every_n_steps: int = 1):
        # Metric to monitor
        self.logger = logger
        self.metric_monitor = metric_monitor
        self.precision = precision
        self.log_every_n_steps = log_every_n_steps

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule,
                       **kwargs) -> None:
        self.logger.info("Training started")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule,
                     **kwargs) -> None:
        self.logger.info("Training done")

    def on_validation_epoch_end(self, trainer: Trainer,
                                pl_module: LightningModule, **kwargs) -> None:
        if trainer.sanity_checking:
            self.logger.info("Sanity checking ok.")

    def on_train_epoch_end(self,
                           trainer: Trainer,
                           pl_module: LightningModule,
                           padding=False,
                           **kwargs) -> None:
        metric_format = f"{{:.{self.precision}e}}"
        line = f"Epoch {trainer.current_epoch}"
        if padding:
            line = f"{line:>{len('Epoch xxxx')}}"  # Right padding

        if trainer.current_epoch % self.log_every_n_steps == 0:
            metrics_str = []

            losses_dict = trainer.callback_metrics
            for metric_name, dico_name in self.metric_monitor.items():
                if dico_name in losses_dict:
                    metric = losses_dict[dico_name].item()
                    metric = metric_format.format(metric)
                    metric = f"{metric_name} {metric}"
                    metrics_str.append(metric)

            line = line + ": " + "   ".join(metrics_str)

        self.logger.info(line)


class VisualizationCallback(Callback):
    def __init__(self, save_dir: Union[str, Path], datamodule, interval=1, n_batches_to_viz=2, logger_log=True):
        """
        Args:
            save_dir (Union[str, Path]): Directory to save the GIFs.
            datamodule (LightningDataModule): The data module used for training.
            interval (int): Save every `interval` epochs.
            n_batches_to_viz (int): Number of batches to visualize. Use a negative number to visualize all batches. Each batch contains one sequence.
            logger_log (bool): Whether to log the GIFs to the logger (e.g., TensorBoard/W&B).
        """
        super().__init__()
        self.save_dir = Path(save_dir)

        self.datamodule = datamodule
        # of type: mGPT.data.humanml.dataset_t2m_eval.Text2MotionDatasetEval
        # we visualize data from the sample dataset
        self.sample_dataset = datamodule.get_sample_set(overrides={"split": "test", "tiny": True})

        self.interval = interval
        self.n_batches_to_viz = n_batches_to_viz
        self.logger_log = logger_log
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called at the end of the training epoch. The main entry point.
        
        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: The Lightning Module currently being trained
        """
        # Only create visualizations if the epoch matches the interval
        if (trainer.current_epoch + 1) % self.interval == 0:
            self._generate_visualization(trainer, pl_module)

    def _generate_visualization(self, trainer, pl_module):
        """
        Generates 3D motion visualizations based on the model's predictions.

        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module (LightningModule): The model being trained.
        """
        # Store the current mode
        was_training = pl_module.training

        # Switch to eval mode for inference
        pl_module.eval()

        try:
            with torch.no_grad():
                all_out_names = []
                for batch_idx, batch in enumerate(self.sample_dataset):
                    if 0 <= self.n_batches_to_viz <= batch_idx:
                        break

                    # batch is a tuple
                    # semantics:: (caption, motion, m_length, ...)
                    caption = batch[0]
                    inputs = batch[1]  # Shape: [T, D]
                    inputs = inputs[None]  # Add batch dimension
                    inputs = torch.tensor(inputs).float().to(pl_module.device)

                    # Forward pass through VAE to get reconstructions
                    result_set = pl_module.val_vae_forward(
                        batch={
                            "motion": inputs,
                            "length": [batch[2]],
                        }
                    )

                    # reconstruct the global positions
                    reconstructed = result_set["m_rst"]
                    denormed = self.datamodule.denormalize(reconstructed)
                    positions_in_global_frame = recover_from_ric(
                        denormed, joints_num=22
                    )

                    # Save visualizations
                    out_names = self._save_gif_batch_local(
                        outputs=positions_in_global_frame,
                        epoch=trainer.current_epoch + 1,
                        batch_idx=batch_idx,
                        caption=caption
                    )

                    all_out_names.extend(out_names)
        finally:
            # Restore original training state
            if was_training:
                pl_module.train()

        if self.logger_log and trainer.logger is not None:
            self._log_to_logger(
                epoch=trainer.current_epoch + 1,
                logger=trainer.logger,
                out_names=all_out_names
            )

    def _save_gif_batch_local(self, outputs, epoch, batch_idx, caption=""):
        """
        Creates and saves GIF visualizations for the batch.

        Args:
            outputs (torch.Tensor): Model reconstructions for the batch.
            epoch (int): Current epoch number.
            batch_idx (int): Batch index in the data loader.
            caption (str): Caption for the GIFs.
        """
        # Convert to numpy arrays
        outputs = outputs.cpu().numpy()  # Shape: [B, T, D]
        n_seqs = outputs.shape[0]

        # Generate filenames for each sequence
        out_names = [
            self.save_dir / f"epoch_{epoch}_batch_{batch_idx}_seq_{i}.gif"
            for i in range(n_seqs)
        ]

        # Generate titles
        titles = [f"{caption} batch_{batch_idx}_seq_{i}" for i in range(n_seqs)]

        # Generate and save GIFs
        draw_to_batch(
            smpl_joints_batch=outputs,
            title_batch=titles,
            outname=out_names,
        )

        return out_names

    @staticmethod
    def _log_to_logger(epoch, logger, out_names):
        """
        Logs the GIF visualizations to the logger.

        Args:
            epoch (int): Current epoch number.
            logger: The logger instance.
            out_names (List[Path]): List of paths to the GIFs.
        """
        # Log to logger if enabled
        if isinstance(logger, TensorBoardLogger):
            for i, gif_path in enumerate(out_names):
                gif_data = imageio.imread(gif_path)

                logger.experiment.add_image(
                    f"recon_visualization/motion_seq_{i}",
                    gif_data,
                    epoch,
                )
        elif isinstance(logger, WandbLogger):
            images = {
                f"recon_visualization/motion_seq_{i}": wandb.Image(gif_path.as_posix())
                for i, gif_path in enumerate(out_names)
            }
            logger.experiment.log(images, step=epoch)
