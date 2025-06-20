"""Main file for LSM Training."""

from absl import flags
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic import app
from scenic.projects.baselines import simple_cnn
from scenic.projects.baselines import vit
import tensorflow as tf

from google3.experimental.largesensormodels.scenic.datasets import get_dataset
from google3.experimental.largesensormodels.scenic.models import lsm_resnet
from google3.experimental.largesensormodels.scenic.models import lsm_vit
from google3.experimental.largesensormodels.scenic.trainers import lsm_embedding_dump
from google3.experimental.largesensormodels.scenic.trainers import lsm_generative_task_standalone_eval as lsm_generative_evaluator
from google3.experimental.largesensormodels.scenic.trainers import lsm_linear_probe_standalone_eval as lsm_linear_probe_evaluator
from google3.experimental.largesensormodels.scenic.trainers import lsm_linear_probe_standalone_trainer as lsm_linear_probe_trainer
from google3.experimental.largesensormodels.scenic.trainers import lsm_mae_trainer
from google3.experimental.largesensormodels.scenic.trainers import lsm_supervised_standalone_eval as lsm_supervised_evaluator
from google3.experimental.largesensormodels.scenic.trainers import lsm_supervised_trainer


FLAGS = flags.FLAGS


def get_model_cls(model_name: str):
  """Get the model class for the Multimask project."""

  # LSM ViT MAE Models.
  if model_name == 'lsm_vit_mae':
    return lsm_vit.ViTMAESingleChannelModel

  # Supervised Baseline Models.
  elif model_name == 'simplecnn':
    return simple_cnn.SimpleCNNClassificationModel
  elif model_name == 'resnet':
    return lsm_resnet.ResNetClassificationModel
  elif model_name == 'metadata_encoded_resnet':
    return lsm_resnet.MetadataEncodedResNetClassificationModel
  elif model_name == 'vit':
    return vit.ViTMultiLabelClassificationModel
  else:
    raise ValueError(f'Unrecognized model: {model_name}.')


def get_train_fn(trainer_name):
  """Get the trainer function."""

  # Trainers + Evaluators.
  if trainer_name == 'lsm_mae_trainer':
    return lsm_mae_trainer.train
  elif trainer_name == 'lsm_supervised_trainer':
    return lsm_supervised_trainer.train
  elif trainer_name == 'lsm_linear_probe_trainer':
    return lsm_linear_probe_trainer.linear_probe_train

  # Standalone Evaluators.
  elif (
      trainer_name == 'lsm_generative_evaluator'
      or trainer_name == 'lsm_v2_generative_evaluator'
  ):
    return lsm_generative_evaluator.evaluate
  elif trainer_name == 'lsm_linear_probe_evaluator':
    return lsm_linear_probe_evaluator.linear_probe_evaluate
  elif trainer_name == 'lsm_supervised_evaluator':
    return lsm_supervised_evaluator.supervised_evaluate

  # Embedding Dumpers.
  elif trainer_name == 'lsm_embedding_dump':
    return lsm_embedding_dump.dump_embeddings

  else:
    raise ValueError(f'Unrecognized trainer: {trainer_name}.')


def main(
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    workdir: str,
    writer: metric_writers.MetricWriter,
):
  """Main function for the PlainViT project."""
  # Set seeds and enable tf op determinism.
  # Needed for the dataset.
  tf.keras.utils.set_random_seed(config.rng_seed)
  np.random.seed(config.rng_seed)
  tf.config.experimental.enable_op_determinism()
  jax.config.update('jax_enable_compilation_cache', False)

  # Build the loss_fn, metrics, and flax_model.
  model_cls = get_model_cls(config.model_name)
  data_rng, rng = jax.random.split(rng)

  # Get the training / eval dataset.
  dataset = get_dataset.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address
  )

  # Get and launch the trainer.
  get_train_fn(config.trainer_name)(
      rng=rng,
      config=config,
      model_cls=model_cls,
      dataset=dataset,
      workdir=workdir,
      writer=writer,
  )


if __name__ == '__main__':
  app.run(main=main)
