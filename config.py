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
