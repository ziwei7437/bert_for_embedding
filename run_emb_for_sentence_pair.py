import argparse
import os
import torch

import logging

from glue.tasks import get_task, MnliMismatchedProcessor
from shared import model_setup as shared_model_setup
from pytorch_pretrained_bert.modeling import BertModel
import shared.initialization as initialization
import shared.log_info as log_info
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from glue.embedding_runners import RunnerParameters, EmbeddingTaskRunner

from run_embedding import get_args

def main():
    pass


if __name__ == '__main__':
    main()