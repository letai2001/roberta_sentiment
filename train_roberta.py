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
from Dataset import DataProcessor

from ModelArg import ModelDataArguments , model_data_args
from DataCollator_custom import CustomDataCollatorForLanguageModeling
from config import output_model_dir , local_path_tokenize , training_args
from DataCollator import DataCollatorCreator
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
tokenizer = model_data_args.get_tokenizer(local_path=output_model_dir, config=config)

# Loading model.
print('Loading actual model...')
model =model_data_args.get_model(config)

# Resize model to fit all tokens in tokenizer.
model.resize_token_embeddings(len(tokenizer))

# Number of model parameters
print("Number of model parameters:", model.num_parameters())

print('Preprocessing datasets...')
data_pro = DataProcessor(model_data_args , training_args)
datasets = data_pro.get_dataset()
text_datasets = datasets.remove_columns('label')
tokenized_datasets = data_pro.preprocess_data(text_datasets, tokenizer)

# Split train/eval datasets
train_dataset, eval_dataset = tokenized_datasets['train'], tokenized_datasets['validation']
print('Training set:', len(train_dataset))
print('Validation set:', len(eval_dataset))

data_collator_creator = DataCollatorCreator(model_data_args)
# Get data collator to modify data format depending on type of model used.
data_collator = data_collator_creator.get_collator(model_data_args, tokenizer)

# Check how many logging prints you'll have. This is to avoid overflowing the
# notebook with a lot of prints. Display warning to user if the logging steps
# that will be displayed is larger than 100.
if (len(train_dataset) // training_args.per_device_train_batch_size \
    // training_args.logging_steps * training_args.num_train_epochs) > 100:
  # Display warning.
  warnings.warn('Your `logging_steps` value will will do a lot of printing!' \
                ' Consider increasing `logging_steps` to avoid overflowing' \
                ' the notebook with a lot of prints!')
