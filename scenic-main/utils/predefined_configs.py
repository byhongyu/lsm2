"""Some pre-defined configs parts that will be borrowed throughout configs."""

from google3.experimental.largesensormodels.scenic.models.lsm_vit_utils import model_constants
from google3.experimental.largesensormodels.scenic.models.lsm_vit_utils import patcher_config
from google3.experimental.largesensormodels.scenic.trainers.masking import masker_config
from google3.experimental.largesensormodels.scenic.utils import base_config

# TODO(girishvn, xumax): This system needs to be re-architeced. Different
# configs should be in different predefined dictionaries.
# TODO(xliucs, dmcduff) Push this file into the lsm codebase or vice versa.
# xmanager jobs break with this import
# from google3.medical.waveforms.modelling.lsm.datasets.lsm import experiment_constants  # pylint: disable=line-too-long

LSM_PREDEFINED_CONFIGS = {}

###### MASKER CONFIGS ######
LSM_PREDEFINED_CONFIGS['downstream_inherited_attn_masking'] = (
    masker_config.Masker_Config(
        maskstrategy_list=[
            masker_config.MaskStrategy_Config(
                strategy='random',
                mask_probability=0.0,
                weight=1,
                mask_dim='time',
                inherited_depend=False,
            ),
        ],
        on_cpu=True,  # CPU-bound masking
        inherited=True,  # inherit imputation missingness mask
        strictmaskperc=0.0  # ALL tokens sent to encoder
    )
)

LSM_PREDEFINED_CONFIGS['downstream_noninherited_masking'] = (
    masker_config.Masker_Config(
        maskstrategy_list=[
            masker_config.MaskStrategy_Config(
                strategy='random',
                mask_probability=0.8,
                weight=1,
                mask_dim='time',
            ),
        ],
        on_cpu=False,
        inherited=False,
    )
)

###### PATCHER CONFIGS ######
for key in model_constants.HIDDEN_SIZES:
  LSM_PREDEFINED_CONFIGS[f'{key}__patcher_config__10by1_sharedpatch'] = (
      patcher_config.Patcher_Config(
          hidden_size=model_constants.HIDDEN_SIZES[key],
          kernel_size=(10, 1),
          groups=1,
          mode='2d',
      )
  )

  LSM_PREDEFINED_CONFIGS[f'{key}__patcher_config__10by2_sharedpatch'] = (
      patcher_config.Patcher_Config(
          hidden_size=model_constants.HIDDEN_SIZES[key],
          kernel_size=(10, 2),
          groups=1,
          mode='2d',
      )
  )

  LSM_PREDEFINED_CONFIGS[f'{key}__patcher_config__10by1_permodalitypatch'] = (
      patcher_config.Patcher_Config(
          hidden_size=model_constants.HIDDEN_SIZES[key] * 26,
          kernel_size=(10),
          groups=26,
          mode='1d',
      )
  )

###### GENERATIVE TASK CONFIGS ######
# Generative Task Masker Configs.
# TODO(xumax, xliucs) redefine these configs as specialized configs with strict
# keys, not using base
LSM_PREDEFINED_CONFIGS['eval_fore_1day'] = base_config.Base_Config(
    {'horizons': [0.00695, 0.02084, 0.04167, .125]}
)
LSM_PREDEFINED_CONFIGS['eval_imp_1day'] = base_config.Base_Config(
    {'horizons': [0.00695, 0.02084, 0.04167, .125]}
)
LSM_PREDEFINED_CONFIGS['eval_randimp_1day'] = base_config.Base_Config(
    # {"random_imputation_ratios": [0.2, 0.5, 0.8]}
    # # temporarily change this because full mask is superset of existing imputation mask
    # # but existing imputation can go up to 80%
    {'random_imputation_ratios': [0.8]}

)

LSM_PREDEFINED_CONFIGS['eval_modimp_1day'] = base_config.Base_Config(
    {'horizons': [2 / 26, 6 / 26, 12 / 26, 20/26]}
)