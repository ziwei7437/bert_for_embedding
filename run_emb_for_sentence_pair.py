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
from glue.embedding_runners_single import RunnerParameters, EmbeddingTaskRunner

from run_embedding import get_args

def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    args = get_args()
    log_info.print_args(args)

    device, n_gpu = initialization.init_cuda_from_args(args, logger=logger)
    initialization.init_seed(args, n_gpu=n_gpu, logger=logger)
    initialization.init_train_batch_size(args)
    initialization.init_output_dir(args)
    initialization.save_args(args)
    task = get_task(args.task_name, args.data_dir)

    # prepare examples, load model as encoder
    tokenizer = shared_model_setup.create_tokenizer(
        bert_model_name=args.bert_model,
        bert_load_mode=args.bert_load_mode,
        do_lower_case=args.do_lower_case,
        bert_vocab_path=args.bert_vocab_path,
    )
    all_state = shared_model_setup.load_overall_state(args.bert_load_path, relaxed=True)

    # Load Model...
    if args.bert_load_mode == "state_model_only":
        state_dict = all_state['model']
        bert_as_encoder = BertModel.from_state_dict(
            config_file=args.bert_config_json_path,
            state_dict=state_dict
        )
    else:
        assert args.bert_load_mode == "from_pretrained"
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank)
        bert_as_encoder = BertModel.from_pretrained(
            pretrained_model_name_or_path=args.bert_model,
            cache_dir=cache_dir
        )

    bert_as_encoder.to(device)

    runner_param = RunnerParameters(
        max_seq_length=args.max_seq_length,
        local_rank=args.local_rank, n_gpu=n_gpu, fp16=args.fp16,
        learning_rate=args.learning_rate, gradient_accumulation_steps=args.gradient_accumulation_steps,
        t_total=None, warmup_proportion=args.warmup_proportion,
        num_train_epochs=args.num_train_epochs,
        train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size,
    )

    runner = EmbeddingTaskRunner(
        bert_model=bert_as_encoder,
        optimizer=None,
        tokenizer=tokenizer,
        label_list=task.get_labels(),
        device=device,
        rparams=runner_param
    )

    # Run training set encoding...
    print("Run training set encoding ... ")
    train_examples = task.get_train_examples()
    train_dataset = runner.run_encoding(train_examples, verbose=True, mode='train')
    print("saving embeddings ... ")
    torch.save(train_dataset, os.path.join(args.output_dir, "train.dataset"))

    # Run development set encoding ...
    eval_examples = task.get_dev_examples()
    eval_dataset = runner.run_encoding(eval_examples, verbose=True, mode='eval')
    print("saving embeddings ... ")
    torch.save(eval_dataset, os.path.join(args.output_dir, 'dev.dataset'))

    # Run test set encoding ...
    test_examples = task.get_test_examples()
    test_dataset = runner.run_encoding(test_examples, verbose=True, mode='test')
    print("saving embeddings ... ")
    torch.save(test_dataset, os.path.join(args.output_dir, "test.dataset"))

    # HACK for MNLI mis-matched set ...
    if args.task_name == 'mnli':
        print("=== Start embedding task for MNLI mis-matched ===")
        mm_eval_examples = MnliMismatchedProcessor().get_dev_examples(task.data_dir)
        mm_eval_dataset = runner.run_encoding(mm_eval_examples, verbose=True, mode='eval')
        print("=== Saving eval dataset ===")
        torch.save(mm_eval_dataset,
                   os.path.join(args.output_dir, "mm_dev.dataset"))
        print("=== Saved ===")

        mm_test_examples = MnliMismatchedProcessor().get_test_examples(task.data_dir)
        mm_test_dataset = runner.run_encoding(mm_test_examples, verbose=True, mode='test')
        print("=== Saving tensor dataset ===")
        torch.save(mm_test_dataset,
                   os.path.join(args.output_dir, "mm_test.dataset"))
        print("=== Saved ===")


if __name__ == '__main__':
    main()