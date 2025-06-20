"""Methods for plotting a confusion matrix and converting it to a tensor.

This is branched from:
google3/fitbit/research/sensor_algorithms/training/logging/confusion_matrix_logging.py

The tensor can be shown in tensorboard via tf.summary.image for instance.
"""

import io
import itertools
import os
import pickle
from typing import Any, Dict, Optional, Sequence, Union

from clu import metric_writers
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skmetrics
import sklearn.preprocessing as skpreprocessing
import tensorflow as tf

from google3.pyglib import gfile


Array = Union[np.ndarray, jnp.ndarray, tf.Tensor]


def plot_to_tensor(figure: plt.Figure) -> tf.Tensor:
  """Converts the given matplotlib figure to an image tensor.

  Forked from:
  google3/fitbit/research/sensor_algorithms/training/logging/
  confusion_matrix_logging.py

  Args:
    figure: A matplotlib figure to convert to an tensor.

  Returns:
    A tensor of the figure image.
  """
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


def confusion_matrix_fig(
    confusion_matrix: tf.Tensor, labels: Sequence[str], scale: float = 0.8
) -> plt.Figure:
  """Returns a matplotlib plot of the given confusion matrix.

  Forked from:
  google3/fitbit/research/sensor_algorithms/training/logging/
  confusion_matrix_logging.py

  Args:
      confusion_matrix: Confusion matrix as 2D numpy array.
      labels: List of class names, will be used as axis labels.
      scale: Scale for the image size.
  """
  label_totals = np.sum(confusion_matrix, axis=1, keepdims=True)
  prediction_totals = np.sum(confusion_matrix, axis=0, keepdims=True)

  cm_normalized = np.nan_to_num(confusion_matrix / label_totals)

  num_labels = len(labels)
  longest_label = max([len(label) for label in labels])

  # Guesstimating an appropriate size.
  image_size = scale * (num_labels + (longest_label / 8.0))

  fig = plt.figure(
      figsize=(image_size, image_size), facecolor='w', edgecolor='k'
  )
  ax = fig.add_subplot(1, 1, 1)
  ax.imshow(cm_normalized, cmap='Blues')

  tick_marks = np.arange(num_labels)

  ax.set_xlabel('Predicted')
  ax.set_xticks(tick_marks)
  x_labels = (
      f'{label} ({int(count):,})'
      for label, count in zip(labels, prediction_totals[0, :])
  )
  ax.set_xticklabels(x_labels, rotation=-45, ha='center')
  ax.xaxis.set_label_position('bottom')
  ax.xaxis.tick_bottom()

  ax.set_ylabel('True Label')
  ax.set_yticks(tick_marks)
  y_labels = (
      f'{label} ({int(count):,})'
      for label, count in zip(labels, label_totals[:, 0])
  )
  ax.set_yticklabels(y_labels, va='center')
  ax.yaxis.set_label_position('left')
  ax.yaxis.tick_left()

  for row_idx, col_idx in itertools.product(
      range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])
  ):
    text_color = 'white' if cm_normalized[row_idx, col_idx] >= 0.5 else 'black'
    if confusion_matrix[row_idx, col_idx] == 0:
      text_str = '.'
    else:
      text_str = (
          f'{cm_normalized[row_idx,col_idx]:2.0%}\n'
          f'({int(confusion_matrix[row_idx, col_idx]):,})'
      )
    ax.text(
        col_idx,
        row_idx,
        text_str,
        horizontalalignment='center',
        verticalalignment='center',
        color=text_color,
    )

  fig.set_tight_layout(True)  # pytype: disable=attribute-error
  return fig


def compute_mean_avg_precision(
    targets: Sequence[int],
    logits: Sequence[int],
    n_classes: int,
    return_per_class_ap=True,
):
  """Computes mean average precision for multi-class classification.

  Forked from: google3/third_party/py/scenic/projects/av_mae/evaluation_lib.py

  Args:
    targets: List of length num_examples - classes indexed.
    logits: List of length num_examples - classes indexed.
    n_classes: Int number of classes
    return_per_class_ap: If True, return results for each class in the summary.

  Returns:
    summary: Dictionary containing the mean average precision, and maybe the
      average precision per class.
  """
  # 0. Setup
  targets = np.array(targets)
  logits = np.array(logits)
  if logits.shape[0] != targets.shape[0]:
    raise ValueError(
        'Predictions and targets have different leading shape\n'
        f'Preds: {logits.shape}\nTargets: {targets.shape}'
    )

  # 1.a. Multi-class classification.
  if n_classes > 2:
    # OHE encode target labels.
    labels = skpreprocessing.label_binarize(
        targets, classes=np.arange(n_classes)
    )
    # Get average precision across all classes.
    average_precisions = []
    summary = {}
    for i in range(n_classes):
      avg_precision = skmetrics.average_precision_score(
          labels[:, i], logits[:, i]
      )
      if return_per_class_ap:
        summary_key = f'class_{i}_AP'
        summary[summary_key] = avg_precision
      average_precisions.append(avg_precision)

    # Update and return metrics.
    summary['nanmean_AP'] = np.nanmean(average_precisions)
    summary['mAP'] = np.mean(average_precisions)

  # 1.b. Binary classification.
  else:
    avg_precision = skmetrics.average_precision_score(targets, logits[:, 1])
    summary = {'AP': avg_precision}

  # 2. Return.
  return summary


# pylint: disable=dangerous-default-value
def classification_metrics(
    targets: Sequence[int],
    preds: Sequence[int],
    logits: Optional[Array],
    label_names: Sequence[str],
    step: int,
    figure_scale: float = 0.8,
    metrics: Sequence[str] = [
        'confusion_matrix',
        'mean_average_precision',
        'balanced_accuracy',
        'f1_score',
    ],
    writer: Optional[metric_writers.MetricWriter] = None,
    write_out_files: bool = False,
    workdir: Optional[str] = None,
    prefix: Optional[str] = None,
    seperator: str = '_',
):
  """Calculates classification metrics from confusion matrix.

  Args:
    targets: A list of target classses
    preds: A list of predictions classes
    logits: A list of prediction logits
    label_names: List of class names, will be used as axis labels.
    step: The training / evaluation step.
    figure_scale: Scale for the image size.
    metrics: A list of metrics to calculate.
    writer: an optional metrics writer to write to.
    write_out_files: Whether or not to write out files.
    workdir: The experiment working directory, to dump files.
    prefix: Prefix to the written metric name
    seperator: A character seperator between the prefix and the metric name.

  Returns:
    A dictionary of metrics calculated from the confusion matrix.
  """
  metrics_dict = dict()
  img_dict = dict()
  file_dump_dict = dict()
  n_classes = len(label_names)
  if prefix is not None:
    prefix = prefix + seperator
  else:
    prefix = ''

  for m in metrics:
    # Confusion Matrix - returns image
    if m == 'confusion_matrix':
      confusion_matrix = tf.math.confusion_matrix(
          targets, preds, num_classes=n_classes
      )
      fig = confusion_matrix_fig(
          confusion_matrix, label_names, scale=figure_scale
      )
      cm_img_tensor = plot_to_tensor(fig)
      cm_img = cm_img_tensor.numpy()
      img_dict['confusion_matrix'] = cm_img
      file_dump_dict['confusion_matrix'] = confusion_matrix
      file_dump_dict['confusion_matrix_labels'] = label_names

    # Mean Average Precision.
    elif m == 'mean_average_precision':
      map_metrics = compute_mean_avg_precision(targets, logits, n_classes)
      for k, v in map_metrics.items():
        metrics_dict[k] = v

    # Balanced Accuracy.
    elif m == 'balanced_accuracy':
      balanced_accuracy = skmetrics.balanced_accuracy_score(targets, preds)
      metrics_dict['balanced_accuracy'] = balanced_accuracy

    # F1 Score.
    elif m == 'f1_score':
      f1_score = skmetrics.f1_score(targets, preds, average='macro')
      metrics_dict['f1_score'] = f1_score

    else:
      raise ValueError(f'Metric {m} is not supported.')

  # Write metrics and images out with writer.
  if writer is not None:
    # Metrics.
    for k, v in metrics_dict.items():
      writer.write_scalars(step, {prefix + k: v},)
    # Images.
    for k, v in img_dict.items():
      writer.write_images(step, {prefix + k: v},)
    writer.flush()
    # Dump Files to CNS.
    if workdir is not None and write_out_files:
      for k, v in file_dump_dict.items():
        if k in ['confusion_matrix', 'confusion_matrix_labels']:
          # dump confusion matrix to CNS
          fpath = os.path.join(workdir, f'{prefix + k}_{step}.npy')
          with gfile.Open(fpath, 'wb') as f:
            np.save(f, np.array(v))

  return metrics_dict


# TODO(girishvn): Move this to a more general util file, and combined with
# implementation from lsm_generative_eval.py
def save_dict_to_pickle(
    data_dict: Dict[str, Any],
    file_dir: str,
    file_name: str
) -> None:
  """Saves a dictionary containing NumPy arrays to a pickle file, excluding keys with NoneType values.

  Args:
      data_dict: The dictionary to save.
      file_dir: The directory to save the file to.
      file_name: The name of the file to save (without extension)
  """

  file_name = f'{file_name}.pickle'
  filepath = os.path.join(file_dir, file_name)  # Combine directory and filename
  with gfile.Open(filepath, 'wb') as f:
    pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def dump_classification_outputs(
    targets: Sequence[int],
    preds: Sequence[int],
    logits: Optional[Array],
    label_names: Sequence[str],
    step: int,
    writer: Optional[metric_writers.MetricWriter] = None,  # pylint: disable=unused-argument
    write_out_files: bool = False,
    workdir: Optional[str] = None,
    prefix: Optional[str] = None,
    seperator: str = '_',  # pylint: disable=unused-argument
) -> None:
  """Dumps classification outputs to CNS."""

  output_dict = {
      'targets': np.array(targets),
      'preds': np.array(preds),
      'logits': np.array(logits),
      'label_names': label_names,
      'step': step,
  }

  # Dump Files to CNS.
  if workdir is not None and write_out_files:
    # dump confusion matrix to CNS
    fdir = workdir
    fname = f'{prefix}_{step}_classification_outputs'
    save_dict_to_pickle(output_dict, fdir, fname)

  return

