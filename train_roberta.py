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
from ModelArg import ModelDataArguments , model_data_args
from DataCollator_custom import CustomDataCollatorForLanguageModeling
from config import local_path_tokenize
# Set seed for reproducibility,
set_seed(69)
#chay lai notebook cung ket qua

# Look for gpu to use. Will use `cpu` by default if no gpu found.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Loading model configuration...')
override_config = {
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "intermediate_size": 2048,
    "hidden_size": 512,
    "max_position_embeddings": 130 # 128+2 for special tokens
}
config = model_data_args.get_model_config(override_config)

# Load model tokenizer.
print('Loading model`s tokenizer...')
tokenizer = model_data_args.get_tokenizer(local_path=local_path_tokenize, config=config)

# Loading model.
print('Loading actual model...')
model =model_data_args.get_model(model_data_args, config)

# Resize model to fit all tokens in tokenizer.
model.resize_token_embeddings(len(tokenizer))

# Number of model parameters
print("Number of model parameters:", model.num_parameters())


