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

# Set seed for reproducibility,
set_seed(69)
#chay lai notebook cung ket qua

# Look for gpu to use. Will use `cpu` by default if no gpu found.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DataProcessor:
    def __init__(self, model_args: ModelDataArguments, train_args: TrainingArguments):
        self.model_args = model_args
        self.train_args = train_args

    def get_dataset(self):
        raw_datasets = load_dataset(
            self.model_args.dataset_name,
            self.model_args.dataset_config_name,
            cache_dir=self.model_args.cache_dir
        )
        # Splitting the dataset into train and validation set if need.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                self.model_args.dataset_name,
                self.model_args.dataset_config_name,
                split=f"train[:{self.model_args.validation_split_percentage}%]",
                cache_dir=self.model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                self.model_args.dataset_name,
                self.model_args.dataset_config_name,
                split=f"train[{self.model_args.validation_split_percentage}%:]",
                cache_dir=self.model_args.cache_dir,
            )
        save_path = "C:\\Users\\Admin\\Downloads\\roberta\\data"  # Thay đổi đường dẫn này theo nhu cầu của bạn
        raw_datasets.save_to_disk(save_path)

        return raw_datasets

        # Code từ hàm get_dataset ở trên
        # ...

    def preprocess_data(self, dataset, tokenizer: PreTrainedTokenizer):
        # Code từ hàm preprocess_data ở trên
        # ...
        r"""
        Preprocess and tokenize the dataset.

        This function can tokenize each nonempty line in the dataset or group chunks together
        after tokenizing every text.

        Arguments:

            args (:obj:`ModelDataArguments`):
            Model and data configuration arguments needed to perform pretraining.

            train_args (:obj:`TrainingArguments`):
            Training arguments needed to perform pretraining.

            dataset (:obj:`Dataset`):
            Raw dataset that needs to be preprocessed.

            tokenizer (:obj:`PreTrainedTokenizer`):
            Model transformers tokenizer.

        Returns:

            :obj:`Dataset`: PyTorch Dataset that contains file's data.

        """

        if self.train_args.do_train:
            column_names = list(dataset["train"].features)
        else:
            column_names = list(dataset["validation"].features)
        text_column_name = "text" if "text" in column_names else column_names[0]

        # Preprocessing the datasets.
        # First we tokenize all the texts.
        if self.model_args.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if self.model_args.pad_to_max_length else False

            def tokenize_function(examples):
                # Remove empty lines
                examples[text_column_name] = [
                    line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
                ]
                return tokenizer(
                    examples[text_column_name],
                    padding=padding,
                    truncation=True,
                    max_length=self.model_args.max_seq_length,
                    # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                    # receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
                )

            with self.train_args.main_process_first(desc="dataset map tokenization"):
                tokenized_datasets = dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=self.model_args.preprocessing_num_workers,
                    remove_columns=[text_column_name],
                    load_from_cache_file=not self.model_args.overwrite_cache,
                    desc="Running tokenizer on dataset line_by_line",
                )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
            # efficient when it receives the `special_tokens_mask`.
            def tokenize_function(examples):
                return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

            with self.train_args.main_process_first(desc="dataset map tokenization"):
                tokenized_datasets = dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=self.model_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not self.model_args.overwrite_cache,
                    desc="Running tokenizer on every text in dataset",
                )

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                if total_length >= self.model_args.max_seq_length:
                    total_length = (total_length // self.model_args.max_seq_length) * self.model_args.max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [t[i : i + self.model_args.max_seq_length] for i in range(0, total_length, self.model_args.max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

            with self.train_args.main_process_first(desc="grouping texts together"):
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=self.model_args.preprocessing_num_workers,
                    load_from_cache_file=not self.model_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {self.model_args.max_seq_length}",
                )

        # Return tokenized datasets
        return tokenized_datasets
