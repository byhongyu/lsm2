"""Common utilities for XManager."""

import os

from absl import logging
import ml_collections

import google3.learning.deepmind.xmanager2.client.google as xm
import google3.learning.deepmind.xmanager2.client.xmanager_api as xm_api
from google3.pyglib import gfile


def get_info_from_xmanager(xid, wid):
  """Given the xid and wid, returns the configs and checkpoint directory."""
  xm_client = xm_api.XManagerApi(
      xm_deployment_env='alphabet', force_remote=True
  )
  xp = xm_client.get_experiment(xid)
  wu = xp.get_work_unit(wid)
  logging.info('Loading configurations of %s.', xp.display_name)
  configs = ml_collections.ConfigDict(wu.configuration['config'])
  checkpoint_path = [
      a.artifact
      for a in xp.get_artifacts()
      if (a.description.lower() in ['workdir on cns', 'workdir'])
  ][0]

  # Replace placeholder strings in the workdir artifact:
  checkpoint_path = checkpoint_path.replace('%vars.experiment_id%', str(xid))
  checkpoint_path = checkpoint_path.replace('/%vars.work_unit_id%', '')
  checkpoint_path = os.path.join(checkpoint_path, str(wid))

  # By default we use r=3 for the experiment dir, so check if it exists.
  checkpoint_path_r3 = os.path.join(checkpoint_path, 'r=3')
  try:
    if gfile.Exists(checkpoint_path_r3):
      checkpoint_path = checkpoint_path_r3
  except gfile.GOSError as e:
    # gfile.Exists() will fail if the user does not have access to the
    # directory. Ignore this since the caller may be overriding the checkpoint
    # path anyway.
    logging.warning(
        'Failed to check r=3 directory %s: %s', checkpoint_path_r3, e
    )
    pass
  return configs, checkpoint_path


def get_xm_note_writer(pool=None):
  """Returns a function to write XManager notes."""
  xm.setup_work_unit()
  xm_wu = xm_api.XManagerApi().get_current_work_unit()
  if pool:

    def write_note(note):
      pool.apply_async(lambda note=note: xm_wu.set_notes(note))

  else:
    write_note = xm_wu.set_notes
  return write_note
