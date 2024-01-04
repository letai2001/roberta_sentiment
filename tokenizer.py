# tokenizer.py
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

from tokenizers import models, pre_tokenizers, trainers, Tokenizer, ByteLevelBPETokenizer
from Dataset import DataProcessor
from ModelArg import model_data_args
def train_tokenizer(dataset_path, vocab_size, output_dir):
    # Load dataset
    data_pro = DataProcessor(model_data_args , None)
    datasets = data_pro.get_dataset()

    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())   # or use ByteLevelBPETokenizer()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) 

    # Define trainer
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=[
        "<s>",  # [CLS]
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    # Train tokenizer
    tokenizer.train_from_iterator(iterator=datasets['train']['text'], trainer=trainer)

    # Save tokenizer
    tokenizer.save(f"{output_dir}/tokenizer.json")

if __name__ == "__main__":
    # Define parameters
    dataset_path = 'sepidmnorozy/Vietnamese_sentiment'  # Change this to your dataset path
    vocab_size = 4000
    output_dir = "./tokenizer_output"

    # Train tokenizer
    train_tokenizer(dataset_path, vocab_size, output_dir)