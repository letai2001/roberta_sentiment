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
from ModelArg import ModelDataArguments
from DataCollator_custom import CustomDataCollatorForLanguageModeling
# Set seed for reproducibility,
set_seed(69)
#chay lai notebook cung ket qua

# Look for gpu to use. Will use `cpu` by default if no gpu found.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DataCollatorCreator:
    def __init__(self, args: ModelDataArguments):
        self.args = args

    def get_collator(self, tokenizer: PreTrainedTokenizer):
        """
        Get appropriate collator function.

        Arguments:
            tokenizer (:obj:`PreTrainedTokenizer`): Model transformers tokenizer.

        Returns:
            :obj:`data_collator`: Transformers specific data collator.
        """
        if self.args.whole_word_mask:
            return DataCollatorForWholeWordMask(
                tokenizer=tokenizer,
                mlm_probability=self.args.mlm_probability,
            )
        else:
            return CustomDataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=self.args.mlm_probability,
            )
