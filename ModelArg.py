import io
import os
import math
import torch
import warnings
from itertools import chain
from dataclasses import dataclass
from tqdm.notebook import tqdm
from collections.abc import Mapping
from datasets import load_dataset
from torch.utils.data.dataset import Dataset
from transformers.data.data_collator import DataCollatorMixin
from transformers import (CONFIG_MAPPING,
                          MODEL_FOR_MASKED_LM_MAPPING,
                          PreTrainedTokenizer,
                          TrainingArguments,
                          AutoConfig,
                          AutoTokenizer,
                          AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling,
                          DataCollatorForWholeWordMask,
                          PretrainedConfig,
                          Trainer,
                          set_seed)
from config import output_model_dir, model_type, dataset_name, vocab_size, max_seq_length, mlm_probability, whole_word_mask, line_by_line, pad_to_max_length
# Set seed for reproducibility,
set_seed(69)
#chay lai notebook cung ket qua

# Look for gpu to use. Will use `cpu` by default if no gpu found.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class ModelDataArguments(object):
    r"""Arguments pertaining to which model/config/tokenizer/data we are going to fine-tune, or train.

    Eve though all arguments are optional, there still needs to be a certain
    number of arguments that require values attributed.

    Raises:

            ValueError: If `CONFIG_MAPPING` is not loaded in global variables.

            ValueError: If `model_type` is not present in `CONFIG_MAPPING.keys()`.

            ValueError: If `model_type`, `model_config_name` and
            `model_name_or_path` variables are all `None`. At least one of them
            needs to be set.

            warnings: If `model_config_name` and `model_name_or_path` are both
            `None`, the model will be trained from scratch.

    """

    def __init__(self,
               model_type=None,
               config_name=None,
               tokenizer_name=None,
               model_name_or_path=None,
               dataset_name=None,
               dataset_config_name=None,
               cache_dir=None,
               preprocessing_num_workers=None,
               line_by_line=False,
               whole_word_mask=False,
               mlm_probability=0.15,
               max_seq_length=-1,
               pad_to_max_length=False,
               overwrite_cache=False,
               validation_split_percentage=5,
               ):

    # Make sure CONFIG_MAPPING is imported from transformers module.
    ## Đảm bảo rằng CONFIG_MAPPING được nhập từ mô-đun mtransformers.

        if 'CONFIG_MAPPING' not in globals():
            raise ValueError('Could not find CONFIG_MAPPING imported! ' \
                            'Make sure to import it from `transformers` module!')

        # Make sure model_type is valid.
        if (model_type is not None) and (model_type not in CONFIG_MAPPING.keys()):
            raise ValueError('Invalid `model_type`! Use one of the following: %s' %
                            (str(list(CONFIG_MAPPING.keys()))))

        # Make sure that model_type, config_name and model_name_or_path
        # variables are not all `None`.
        if not any([model_type, config_name, model_name_or_path]):
            raise ValueError('You can`t have all `model_type`, `config_name`,' \
                            ' `model_name_or_path` be `None`! You need to have' \
                            'at least one of them set!')

        # Check if a new model will be loaded from scratch.
        if not any([config_name, model_name_or_path]):
        # Setup warning to show pretty. This is an overkill
            warnings.formatwarning = lambda message,category,*args,**kwargs: \
                                    '%s: %s\n' % (category.__name__, message)
            # Display warning.
            warnings.warn('You are planning to train a model from scratch! 🙀')

            # Set all data related arguments.
            self.dataset_name = dataset_name
            self.dataset_config_name = dataset_config_name
            self.preprocessing_num_workers = preprocessing_num_workers
            self.line_by_line = line_by_line
            self.whole_word_mask = whole_word_mask
            self.mlm_probability = mlm_probability
            self.max_seq_length = max_seq_length
            self.pad_to_max_length = pad_to_max_length
            self.overwrite_cache = overwrite_cache
            self.validation_split_percentage = validation_split_percentage

            # Set all model and tokenizer arguments.
            self.model_type = model_type
            self.config_name = config_name
            self.tokenizer_name = tokenizer_name
            self.model_name_or_path = model_name_or_path
            self.cache_dir = cache_dir
    
    
    def get_model_config(self, override_config):
        r"""
        Get model configuration.

        Using the ModelDataArguments return the model configuration.

        Arguments:

            args (:obj:`ModelDataArguments`):
            Model and data configuration arguments needed to perform pretraining.

            override_config (:obj:`Config`):
            Configuration to replace the old one.

        Returns:

            :obj:`PretrainedConfig`: Model transformers configuration.

        """

        # Check model configuration.
        if self.config_name is not None:
            # Use model configure name if defined.
            model_config = AutoConfig.from_pretrained(self.config_name,
                                            cache_dir=self.cache_dir)

        elif self.model_name_or_path is not None:
            # Use model name or path if defined.
            model_config = AutoConfig.from_pretrained(self.model_name_or_path,
                                            cache_dir=self.cache_dir)

        else:
            # Use config mapping if building model from scratch.
            model_config = CONFIG_MAPPING[self.model_type]()

        if override_config:
            model_config.update(override_config)

        return model_config
    def get_model(self, model_config):
        """
        Get model.

        Using the ModelDataArguments return the actual model.

        Arguments:
            model_config (:obj:`PretrainedConfig`): Model transformers configuration.

        Returns:
            :obj:`torch.nn.Module`: PyTorch model.
        """

        if 'MODEL_FOR_MASKED_LM_MAPPING' not in globals():
            raise ValueError('Could not find MODEL_FOR_MASKED_LM_MAPPING is imported!')

        if self.model_name_or_path:
            if type(model_config) in MODEL_FOR_MASKED_LM_MAPPING.keys():
                return AutoModelForMaskedLM.from_pretrained(
                    self.model_name_or_path,
                    from_tf=bool(".ckpt" in self.model_name_or_path),
                    config=model_config,
                    cache_dir=self.cache_dir,
                )
            else:
                raise ValueError(
                    'Invalid `model_name_or_path`! It should be in %s!' % str(MODEL_FOR_MASKED_LM_MAPPING.keys())
                )

        else:
            print("Training new model from scratch!")
            return AutoModelForMaskedLM.from_config(model_config)
model_data_args = ModelDataArguments(
    dataset_name=dataset_name,
    line_by_line=line_by_line,
    whole_word_mask=whole_word_mask,
    mlm_probability=mlm_probability,
    max_seq_length=max_seq_length,
    pad_to_max_length=pad_to_max_length,
    overwrite_cache=False,
    model_type=model_type,
    cache_dir=None
)
