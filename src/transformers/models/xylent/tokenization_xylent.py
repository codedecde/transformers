# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from ..xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer


VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "xylent-base": "TODO(bapatra)<URL-base>",
    },
    "vocab_file": {
        "xylent-large": "TODO(bapatra)<URL-LARGE>",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "xylent-base": 514,
    "xylent-large": 514
}


class XYLentTokenizer(XLMRobertaTokenizer):
    vocab_file_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    """
    The XYLent tokenizer is based on the XLM-RoBERTa tokenizer. The primary difference is an increased
    vocabulary size from 250002 to 500002, and a different method for the original vocabulary allocation.
    """
