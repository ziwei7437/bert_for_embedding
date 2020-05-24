import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import tqdm

import logging

from glue import Batch, InputExample, InputFeatures
from glue.embedding_runners import (
    LabelModes,
    tokenize_example,
    is_null_label_map,
    get_label_mode,
)
from pytorch_pretrained_bert.utils import truncate_seq_pair

logger = logging.getLogger(__name__)


def convert_example_to_feature(example, tokenizer, max_seq_length, label_map):
    if isinstance(example, InputExample):
        example = tokenize_example(example, tokenizer)

    tokens_a, tokens_b = example.tokens_a, example.tokens_b
    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if is_null_label_map(label_map):
        label_id = example.label
    else:
        label_id = label_map[example.label]
    return InputFeatures(
        guid=example.guid,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        tokens=tokens,
    )


def convert_examples_to_features(examples, label_map, max_seq_length, tokenizer, verbose=True):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        feature_instance = convert_example_to_feature(
            example=example,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            label_map=label_map,
        )
        if verbose and ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in feature_instance.tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in feature_instance.input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in feature_instance.input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in feature_instance.segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, feature_instance.label_id))

        features.append(feature_instance)
    return features


def convert_to_dataset(features, label_mode):
    full_batch = features_to_data(features, label_mode=label_mode)
    if full_batch.label_ids is None:
        dataset = TensorDataset(full_batch.input_ids, full_batch.input_mask,
                                full_batch.segment_ids)
    else:
        dataset = TensorDataset(full_batch.input_ids, full_batch.input_mask,
                                full_batch.segment_ids, full_batch.label_ids)
    return dataset, full_batch.tokens


def features_to_data(features, label_mode):
    if label_mode == LabelModes.CLASSIFICATION:
        label_type = torch.long
    elif label_mode == LabelModes.REGRESSION:
        label_type = torch.float
    else:
        raise KeyError(label_mode)
    return Batch(
        input_ids=torch.tensor([f.input_ids for f in features], dtype=torch.long),
        input_mask=torch.tensor([f.input_mask for f in features], dtype=torch.long),
        segment_ids=torch.tensor([f.segment_ids for f in features], dtype=torch.long),
        label_ids=torch.tensor([f.label_id for f in features], dtype=label_type),
        tokens=[f.tokens for f in features],
    )


class HybridLoader:
    def __init__(self, dataloader, tokens):
        self.dataloader = dataloader
        self.tokens = tokens

    def __iter__(self):
        batch_size = self.dataloader.batch_size
        for i, batch in enumerate(self.dataloader):
            if len(batch) == 4:
                input_ids, input_mask, segment_ids, label_ids = batch
            elif len(batch) == 3:
                input_ids, input_mask, segment_ids = batch
                label_ids = None
            else:
                raise RuntimeError()
            batch_tokens = self.tokens[i * batch_size: (i + 1) * batch_size]
            yield Batch(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
                tokens=batch_tokens,
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

    def run_encoding(self, examples, verbose=True, mode='train'):
        if mode != 'train' and mode != 'test' and mode != 'eval':
            raise ValueError("mode must be either of 'train', 'eval', or 'test'.")
        if mode == 'train':
            batch_size = self.rparams.train_batch_size
        elif mode == 'eval':
            batch_size = self.rparams.eval_batch_size
        else:
            batch_size = self.rparams.eval_batch_size
        if verbose:
            logger.info("***** Running Encoding Task *****")
            logger.info("  {} num examples = {}".format(mode, len(examples)))
            logger.info("  batch size = %d", batch_size)
        self.bert_model.eval()

        tensor_list, labels_list = [], []
        if mode == 'test':
            labels_list = None

        print("=== Run encoding for training set ===")
        dataloader = self.get_dataloader(examples, batch_size)
        for step, batch in enumerate(tqdm(dataloader)):
            self.run_encoding_step(
                step, batch, tensor_list, labels_list)
        embeddings = torch.cat(tensor_list).cpu()
        labels = torch.cat(labels_list).cpu()
        print("shape of {} set sentence a: {}".format(mode, embeddings.shape))
        print("shape of {} set labels: {}".format(mode, labels.shape))
        dataset = TensorDataset(embeddings, labels)
        return dataset

    def run_encoding_step(self, step, batch, tensor_list, label_list):
        batch = batch.to(self.device)
        self.bert_model.eval()
        with torch.no_grad():
            _, pooled_output = self.bert_model(batch.input_ids,
                                               batch.segment_ids, batch.input_mask,
                                               output_all_encoded_layers=False)

        tensor_list.append(pooled_output)
        if label_list is not None:
            label_list.append(batch.label_ids)

    def get_dataloader(self, examples, batch_size, verbose=True):
        features = convert_examples_to_features(
            examples, self.label_map, self.rparams.max_seq_length, self.tokenizer, verbose=verbose
        )
        data, tokens = convert_to_dataset(
            features, label_mode=get_label_mode(self.label_map)
        )
        sampler = SequentialSampler(data)
        data_loader = DataLoader(
            data, sampler=sampler, batch_size=batch_size,
        )
        return HybridLoader(data_loader, tokens)
