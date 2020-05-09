import collections as col
import logging
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as pl

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from glue import InputFeatures, InputFeaturesSeparated, Batch, BatchSeparated, InputExample, TokenizedExample
from glue import compute_metrics
from pytorch_pretrained_bert.utils import truncate_seq_pair
from shared.runners import warmup_linear

logger = logging.getLogger(__name__)


class LabelModes:
    CLASSIFICATION = "CLASSIFICATION"
    REGRESSION = "REGRESSION"


def tokenize_example(example, tokenizer):
    tokens_a = tokenizer.tokenize(example.text_a)
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
    else:
        tokens_b = example.text_b
    return TokenizedExample(
        guid=example.guid,
        tokens_a=tokens_a,
        tokens_b=tokens_b,
        label=example.label,
    )


def convert_example_to_feature_separated(example, tokenizer, max_seq_length, label_map):
    if isinstance(example, InputExample):
        example = tokenize_example(example, tokenizer)

    tokens_a, tokens_b = example.tokens_a, example.tokens_b
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:(max_seq_length - 2)]
    if len(tokens_b) > max_seq_length - 2:
        tokens_b = tokens_b[:(max_seq_length - 2)]

    tokens_a = ["[CLS]"] + tokens_a + ["[SEP]"]
    tokens_b = ["[CLS]"] + tokens_b + ["[SEP]"]
    segment_ids_a = [0] * len(tokens_a)
    segment_ids_b = [0] * len(tokens_b)

    input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
    input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)

    input_mask_a = [1] * len(input_ids_a)
    input_mask_b = [1] * len(input_ids_b)

    padding_a = [0] * (max_seq_length - len(input_ids_a))
    input_ids_a += padding_a
    input_mask_a += padding_a
    segment_ids_a += padding_a

    padding_b = [0] * (max_seq_length - len(input_ids_b))
    input_ids_b += padding_b
    input_mask_b += padding_b
    segment_ids_b += padding_b

    assert len(input_ids_a) == max_seq_length
    assert len(input_mask_a) == max_seq_length
    assert len(segment_ids_a) == max_seq_length
    assert len(input_ids_b) == max_seq_length
    assert len(input_mask_b) == max_seq_length
    assert len(segment_ids_b) == max_seq_length

    if is_null_label_map(label_map):
        label_id = example.label
    else:
        label_id = label_map[example.label]

    return InputFeaturesSeparated(
        guid=example.guid,
        input_ids_a=input_ids_a,
        input_ids_b=input_ids_b,
        input_mask_a=input_mask_a,
        input_mask_b=input_mask_b,
        segment_ids_a=segment_ids_a,
        segment_ids_b=segment_ids_b,
        label_id=label_id,
        tokens_a=tokens_a,
        tokens_b=tokens_b,
    )


def convert_examples_to_features_separated(examples, label_map, max_seq_length, tokenizer, verbose=True):
    features = []
    for (ex_index, example) in enumerate(examples):
        feature_instance = convert_example_to_feature_separated(
            example=example,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            label_map=label_map,
        )
        if verbose and ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens_a: %s" % " ".join([str(x) for x in feature_instance.tokens_a]))
            logger.info("input_ids_a: %s" % " ".join([str(x) for x in feature_instance.input_ids_a]))
            logger.info("input_mask_a: %s" % " ".join([str(x) for x in feature_instance.input_mask_a]))
            logger.info(
                "segment_ids_a: %s" % " ".join([str(x) for x in feature_instance.segment_ids_a]))

            logger.info("tokens_b: %s" % " ".join([str(x) for x in feature_instance.tokens_b]))
            logger.info("input_ids_b: %s" % " ".join([str(x) for x in feature_instance.input_ids_b]))
            logger.info("input_mask_b: %s" % " ".join([str(x) for x in feature_instance.input_mask_b]))
            logger.info(
                "segment_ids_b: %s" % " ".join([str(x) for x in feature_instance.segment_ids_b]))

            logger.info("label: %s (id = %d)" % (example.label, feature_instance.label_id))
        features.append(feature_instance)
    return features


def convert_to_dataset_separated(features, label_mode):
    full_batch = features_to_data_separated(features, label_mode=label_mode)
    if full_batch.label_ids is None:
        dataset = TensorDataset(full_batch.input_ids_a, full_batch.input_mask_a,
                                full_batch.segment_ids_a, full_batch.input_ids_b,
                                full_batch.input_mask_b, full_batch.segment_ids_b)
    else:
        dataset = TensorDataset(full_batch.input_ids_a, full_batch.input_mask_a,
                                full_batch.segment_ids_a,
                                full_batch.input_ids_b, full_batch.input_mask_b,
                                full_batch.segment_ids_b,
                                full_batch.label_ids)
    return dataset, full_batch.tokens_a, full_batch.tokens_b


def features_to_data_separated(features, label_mode):
    if label_mode == LabelModes.CLASSIFICATION:
        label_type = torch.long
    elif label_mode == LabelModes.REGRESSION:
        label_type = torch.float
    else:
        raise KeyError(label_mode)
    return BatchSeparated(
        input_ids_a=torch.tensor([f.input_ids_a for f in features], dtype=torch.long),
        input_mask_a=torch.tensor([f.input_mask_a for f in features], dtype=torch.long),
        segment_ids_a=torch.tensor([f.segment_ids_a for f in features], dtype=torch.long),
        label_ids=torch.tensor([f.label_id for f in features], dtype=label_type),
        tokens_a=[f.tokens_a for f in features],
        tokens_b=[f.tokens_b for f in features],
        input_ids_b=torch.tensor([f.input_ids_b for f in features], dtype=torch.long),
        input_mask_b=torch.tensor([f.input_mask_b for f in features], dtype=torch.long),
        segment_ids_b=torch.tensor([f.segment_ids_b for f in features], dtype=torch.long),
    )


class HybridLoaderSeparated:
    def __init__(self, dataloader, tokens_a, tokens_b):
        self.dataloader = dataloader
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b

    def __iter__(self):
        batch_size = self.dataloader.batch_size
        for i, batch in enumerate(self.dataloader):
            if len(batch) == 7:
                input_ids_a, input_mask_a, segment_ids_a, input_ids_b, input_mask_b, segment_ids_b, label_ids = batch
            elif len(batch) == 6:
                input_ids_a, input_mask_a, segment_ids_a, input_ids_b, input_mask_b, segment_ids_b = batch
                label_ids = None
            else:
                raise RuntimeError()
            batch_tokens_a = self.tokens_a[i * batch_size: (i + 1) * batch_size]
            batch_tokens_b = self.tokens_b[i * batch_size: (i + 1) * batch_size]
            yield BatchSeparated(
                input_ids_a=input_ids_a,
                input_mask_a=input_mask_a,
                segment_ids_a=segment_ids_a,
                label_ids=label_ids,
                tokens_a=batch_tokens_a,
                tokens_b=batch_tokens_b,
                input_ids_b=input_ids_b,
                input_mask_b=input_mask_b,
                segment_ids_b=segment_ids_b,
            )

    def __len__(self):
        return len(self.dataloader)


class RunnerParameters:
    def __init__(self, max_seq_length, local_rank, n_gpu, fp16,
                 learning_rate, gradient_accumulation_steps, t_total, warmup_proportion,
                 num_train_epochs, train_batch_size, eval_batch_size):
        self.max_seq_length = max_seq_length
        self.local_rank = local_rank
        self.n_gpu = n_gpu
        self.fp16 = fp16
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.t_total = t_total
        self.warmup_proportion = warmup_proportion
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size


def is_null_label_map(label_map):
    return len(label_map) == 1 and label_map[None] == 0


def get_label_mode(label_map):
    if is_null_label_map(label_map):
        return LabelModes.REGRESSION
    else:
        return LabelModes.CLASSIFICATION


class EmbeddingTaskRunner:
    def __init__(self, bert_model, optimizer, tokenizer,
                 label_list, device, rparams):
        self.bert_model = bert_model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.label_map = {v: i for i, v in enumerate(label_list)}
        self.device = device
        self.rparams = rparams

    def run_encoding(self, train_examples, eval_examples, verbose=True):
        if verbose:
            logger.info("***** Running Encoding Task *****")
            if train_examples is not None:
                logger.info("  Train Num examples = %d", len(train_examples))
            if eval_examples is not None:
                logger.info("  Eval Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", self.rparams.train_batch_size)

        self.bert_model.eval()
        train_tensor_list_a, train_tensor_list_b, train_labels_tensor_list = [], [], []
        eval_tensor_list_a, eval_tensor_list_b, eval_labels_tensor_list = [], [], []
        train_dataset_embeddings, eval_dataset_embeddings = None, None

        if train_examples is not None:
            print("=== Run encoding for training set ===")
            train_dataloader = self.get_train_dataloader_separated(train_examples)
            for step, batch in enumerate(tqdm(train_dataloader)):
                self.run_encoding_step(
                    step, batch, train_tensor_list_a, train_tensor_list_b, train_labels_tensor_list)
            train_embeddings_a = torch.cat(train_tensor_list_a).cpu()
            train_embeddings_b = torch.cat(train_tensor_list_b).cpu()
            train_labels = torch.cat(train_labels_tensor_list).cpu()
            print("shape of train set sentence a: {}".format(train_embeddings_a.shape))
            print("shape of train set sentence b: {}".format(train_embeddings_b.shape))
            print("shape of train set labels: {}".format(train_labels.shape))
            train_dataset_embeddings = TensorDataset(train_embeddings_a, train_embeddings_b, train_labels)

        if eval_examples is not None:
            eval_dataloader = self.get_eval_dataloader_separated(eval_examples)
            print("=== Run encoding for dev set ===")
            for step, batch in enumerate(tqdm(eval_dataloader)):
                self.run_encoding_step(
                    step, batch, eval_tensor_list_a, eval_tensor_list_b, eval_labels_tensor_list
                )
            eval_embeddings_a = torch.cat(eval_tensor_list_a).cpu()
            eval_embeddings_b = torch.cat(eval_tensor_list_b).cpu()
            eval_labels = torch.cat(eval_labels_tensor_list).cpu()
            print("shape of dev set sentence a: {}".format(eval_embeddings_a.shape))
            print("shape of dev set sentence b: {}".format(eval_embeddings_b.shape))
            print("shape of dev set labels: {}".format(eval_labels.shape))
            eval_dataset_embeddings = TensorDataset(eval_embeddings_a, eval_embeddings_b, eval_labels)

        return train_dataset_embeddings, eval_dataset_embeddings

    def run_encoding_step(self, step, batch, tensor_list_a, tensor_list_b, label_tensor_list):
        batch = batch.to(self.device)
        self.bert_model.eval()
        with torch.no_grad():
            _, pooled_output_a = self.bert_model(batch.input_ids_a, batch.segment_ids_a, batch.input_mask_a,
                                                 output_all_encoded_layers=False)
            _, pooled_output_b = self.bert_model(batch.input_ids_b, batch.segment_ids_b, batch.input_mask_b,
                                                 output_all_encoded_layers=False)
            tensor_list_a.append(pooled_output_a)
            tensor_list_b.append(pooled_output_b)
            if label_tensor_list is not None:
                label_tensor_list.append(batch.label_ids)

    def run_test_encoding(self, test_examples, verbose=True):
        if verbose:
            logger.info("  Test Num examples = %d", len(test_examples))
            logger.info("  Batch size = %d", self.rparams.train_batch_size)

        self.bert_model.eval()
        test_tensor_list_a, test_tensor_list_b = [], []

        print("=== Run encoding for test set ===")
        test_data_loader = self.get_eval_dataloader_separated(test_examples)
        for step, batch in enumerate(tqdm(test_data_loader)):
            self.run_encoding_step(
                step, batch, test_tensor_list_a, test_tensor_list_b, None
            )
        test_embeddings_a = torch.cat(test_tensor_list_a).cpu()
        test_embeddings_b = torch.cat(test_tensor_list_b).cpu()
        print("shape of test set sentence a: {}".format(test_embeddings_a.shape))
        print("shape of test set sentence b: {}".format(test_embeddings_b.shape))
        test_dataset_embeddings = TensorDataset(test_embeddings_a, test_embeddings_b)
        return test_dataset_embeddings

    def get_train_dataloader_separated(self, train_examples, verbose=True):
        train_features = convert_examples_to_features_separated(
            train_examples, self.label_map, self.rparams.max_seq_length, self.tokenizer,
            verbose=verbose
        )
        train_data, train_tokens_a, train_tokens_b = convert_to_dataset_separated(
            train_features, label_mode=get_label_mode(self.label_map)
        )
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=self.rparams.train_batch_size,
        )
        return HybridLoaderSeparated(train_dataloader, train_tokens_a, train_tokens_b)

    def get_eval_dataloader_separated(self, eval_examples, verbose=True):
        eval_features = convert_examples_to_features_separated(
            eval_examples, self.label_map, self.rparams.max_seq_length, self.tokenizer,
            verbose=verbose,
        )
        eval_data, eval_tokens_a, eval_tokens_b = convert_to_dataset_separated(
            eval_features, label_mode=get_label_mode(self.label_map),
        )
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=self.rparams.eval_batch_size,
        )
        return HybridLoaderSeparated(eval_dataloader, eval_tokens_a, eval_tokens_b)


if __name__ == "__main__":
    from glue.tasks import get_task
    from shared import model_setup as shared_model_setup
    from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
    from pytorch_pretrained_bert.modeling import BertModel

    task = get_task("wnli", "../../jiant_data/WNLI")
    train_examples = task.get_train_examples()
    label_map = {k: v for v, k in enumerate(task.get_labels())}
    tokenizer = shared_model_setup.create_tokenizer(
        bert_model_name="bert-base-uncased",
        bert_load_mode="from_pretrained",
        do_lower_case=True,
    )
    bert_vocab_path = "../cache/bert_metadata/uncased_L-12_H-768_A-12/vocab.txt"
    train_features = convert_examples_to_features_separated(
        train_examples, label_map=label_map, max_seq_length=100, tokenizer=tokenizer, verbose=True)
    train_data, train_tokens_a, train_tokens_b = convert_to_dataset_separated(
        train_features, label_mode=get_label_mode(label_map)
    )
    cache_dir = PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1)

    # bert_as_encoder = BertModel.from_pretrained(
    #     pretrained_model_name_or_path="bert-base-uncased",
    #     cache_dir=cache_dir
    # )
    # rparam = RunnerParameters(
    #     128, -1, 0, False, 2e-5, 1, 2000, 0.1, 1, 32, 32
    # )
    # runner = EmbeddingTaskRunner(
    #     bert_as_encoder, None, tokenizer, task.get_labels(), "cpu", rparam
    # )
    # runner.run_encoding(train_examples, None)

