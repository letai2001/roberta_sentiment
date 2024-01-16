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
output_model_dir = "./mlm_bert"  # @param {type:"string"}

model_type = 'roberta'  # @param {type:"string"}

dataset_name = 'sepidmnorozy/Vietnamese_sentiment'  # @param {type:"string"}

# Let’s arbitrarily pick its size to be 5,000.
vocab_size = 4000   # @param {type:"number"}

max_seq_length = 128   # @param {type:"number"}

mlm_probability = 0.15   # @param {type:"number"}

whole_word_mask = False   # @param {type:"boolean"}

line_by_line = True   # @param {type:"boolean"}

pad_to_max_length = True   # @param {type:"boolean"}

# Create model folder
# Batch size GPU/TPU core/CPU training.
batch_size = 64  # @param {type:"number"}

# The initial learning rate for Adam. Defaults to 5e-5.
learning_rate = 1e-4  # @param {type:"number"}

# Total number of training epochs to perform (if not an integer, will perform the
# decimal part percents of the last epoch before stopping training). max_steps = 200_000,
num_train_epochs = 20  # @param {type:"number"}

# How often to show logs. I will se this to plot history loss and calculate perplexity.
logging_steps = 100  # @param {type:"number"}

# Number of updates steps before two checkpoint saves. Defaults to 500
save_steps = 100  # @param {type:"number"}

# The weight decay to apply (if not zero).
weight_decay = 0.0  # @param {type:"number"}

# Epsilon for the Adam optimizer. Defaults to 1e-8
adam_epsilon = 1e-8  # @param {type:"number"}

# Maximum gradient norm (for gradient clipping). Defaults to 0.
max_grad_norm = 1.0  # @param {type:"number"}

# The total number of saved models.
save_total_limit = 3  # @param {type:"number"}

# Set prediction loss to `True` in order to return loss for perplexity calculation.
prediction_loss_only = True # @param {type: "boolean"}

# Whether to run training or not.
do_train = True # @param {type: "boolean"}

# Whether to run evaluation or not.
do_eval = True # @param {type: "boolean"}

# Overwrite the content of the output directory.
overwrite_output_dir = True # @param {type: "boolean"}


local_path_tokenize = 'C:\\Users\\Admin\\Downloads\\roberta\\tokenizer.json'

# Define arguments for training
# `TrainingArguments` contains a lot more arguments.
# For more details check the awesome documentation:
# https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
training_args = TrainingArguments(
                          output_dir=output_model_dir,
                          overwrite_output_dir=overwrite_output_dir,
                          do_train=do_train,
                          do_eval=do_eval,
                          per_device_train_batch_size=batch_size,
                          logging_steps=logging_steps,
                          prediction_loss_only=prediction_loss_only,
                          learning_rate = learning_rate,
                          weight_decay=weight_decay,
                          adam_epsilon = adam_epsilon,
                          max_grad_norm = max_grad_norm,
                          num_train_epochs = num_train_epochs,
                          save_steps = save_steps,
                          save_total_limit = save_total_limit)
