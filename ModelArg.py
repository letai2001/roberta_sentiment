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
            
            model_type: Loại mô hình sử dụng: bert, roberta
        
            config_name: Cấu hình của mô hình sử dụng: bert, roberta
            
            tokenizer_name: Tokenizer được sử dụng để xử lý dữ liệu cho việc huấn luyện mô hình.
            Thông thường, nó có cùng tên với model_name_or_path: bert-base-cased, roberta-base, v.v.
            
            model_name_or_path :Đường dẫn đến mô hình transformer hiện có 
            hoặc tên của mô hình transformer sẽ được sử dụng: bert-base-cased, roberta-base vv.
            
            dataset_name: Tên của bộ dữ liệu sử dụng (qua thư viện datasets).
            
            dataset_config_name: Tên cấu hình của bộ dữ liệu sẽ sử dụng (qua thư viện datasets). 
            
            cache_dir: Đường dẫn đến các tệp cache. Nó giúp tiết kiệm thời gian khi chạy lại code. 
            
            preprocessing_num_workers: Số lượng "quá trình" sử dụng cho quá trình tiền xử lý
            
            line_by_line: Có nên xử lý các dòng văn bản khác nhau trong tập dữ liệu như là các chuỗi khác nhau không?
            whole_word_mask: Được sử dụng như là flag để xác định liệu chúng ta có quyết định sử dụng việc che kín toàn bộ từ hay không. Việc che kín toàn bộ từ có nghĩa là toàn bộ từ sẽ được che kín
            trong quá trình đào tạo thay vì các token có thể là các phần của từ.
            
            mlm_probability: Khi đào tạo các mô hình ngôn ngữ bị mask . Cần phải có mlm=True. 
            Nó biểu thị xác suất che khuất các token khi đào tạo mô hình.
            
            max_seq_length: Độ dài tối đa của chuỗi đầu vào sau khi được mã hóa thành các token. Các chuỗi dài hơn sẽ bị cắt ngắn.
            
            pad_to_max_length: Có nên đệm tất cả các mẫu cho đến max_seq_length không? Nếu False, 
            pad  sẽ được đệm theo cách linh hoạt khi chia thành batch sao cho đạt đến độ dài tối đa trong batch.
            overwrite_cache:Nếu có bất kỳ tệp được lưu trữ trong bộ nhớ cache, hãy ghi đè lên chúng.
            validation_split_percentage:  Tỷ lệ phần trăm của tập huấn luyện được sử dụng làm tập validation nếu không có phân chia validation.
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
    def get_tokenizer(self, local_path, config):
        r"""
        Get model tokenizer.

        Using the ModelDataArguments return the model tokenizer and change
        `max_seq_length` from `args` if needed.

        Arguments:

            args (:obj:`ModelDataArguments`):
            Model and data configuration arguments needed to perform pre-training.

            local_path (:obj:`str`):
            Path to the trained tokenizer.

            config (:obj:`Config`):
            Model Configuration.

        Returns:

            :obj:`PreTrainedTokenizer`: Model transformers tokenizer.

        """

        # Check tokenizer configuration.
        if self.tokenizer_name:
            # Use tokenizer name if defined.
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name,
                                                    cache_dir=self.cache_dir)

        elif self.model_name_or_path:
            # Use tokenizer name of path if defined.
            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path,
                                                    cache_dir=self.cache_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained(local_path,
                                                    config=config,
                                                    cache_dir=self.cache_dir)
        
        # Setup data maximum number of tokens.
        if self.max_seq_length <= 0:
            # Set max_seq_length to maximum length of tokenizer.
            # Input max_seq_length will be the max possible for the model.
            self.max_seq_length = tokenizer.model_max_length
        else:
            # Never go beyond tokenizer maximum length.
            self.max_seq_length = min(self.max_seq_length, tokenizer.model_max_length)

        return tokenizer




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

