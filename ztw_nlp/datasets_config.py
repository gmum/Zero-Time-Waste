import os

import torch.utils.data
from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups
from transformers import AutoTokenizer


def get_rte(tokenizer_name=None, padding="max_length", max_seq_length=128, truncation=True):
    return get_glue_dataset('rte', tokenizer_name=tokenizer_name, padding=padding, max_seq_length=max_seq_length,
                            truncation=truncation)


def get_qqp(tokenizer_name=None, padding="max_length", max_seq_length=128, truncation=True):
    return get_glue_dataset('qqp', tokenizer_name=tokenizer_name, padding=padding, max_seq_length=max_seq_length,
                            truncation=truncation)


def get_qnli(tokenizer_name=None, padding="max_length", max_seq_length=128, truncation=True):
    return get_glue_dataset('qnli', tokenizer_name=tokenizer_name, padding=padding, max_seq_length=max_seq_length,
                            truncation=truncation)


def get_mrpc(tokenizer_name=None, padding="max_length", max_seq_length=128, truncation=True):
    return get_glue_dataset('mrpc', tokenizer_name=tokenizer_name, padding=padding, max_seq_length=max_seq_length,
                            truncation=truncation)


def get_sst2(tokenizer_name=None, padding="max_length", max_seq_length=128, truncation=True):
    return get_glue_dataset('sst2', tokenizer_name=tokenizer_name, padding=padding, max_seq_length=max_seq_length,
                            truncation=truncation)


class NlpHfDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        y = sample.pop('label')
        return sample, y


class NlpDictDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset['label'])

    def __getitem__(self, idx):
        sample = {k: v[idx] for k, v in self.dataset.items()}
        y = sample.pop('label')
        return sample, y


def get_glue_dataset(task_name, tokenizer_name=None, padding="max_length", max_seq_length=128, truncation=True):
    train_dataset = load_dataset("glue", task_name, split='train', cache_dir=os.environ['TRANSFORMERS_CACHE_DIR'])
    validation_dataset = load_dataset("glue", task_name, split='validation',
                                      cache_dir=os.environ['TRANSFORMERS_CACHE_DIR'])

    task_to_keys = {
        # "cola": ("sentence", None),
        # "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        # "stsb": ("sentence1", "sentence2"),
        # "wnli": ("sentence1", "sentence2"),
    }
    if task_name not in task_to_keys.keys():
        raise NotImplementedError()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                              use_fast=True,
                                              cache_dir=os.environ['TRANSFORMERS_CACHE_DIR'])
    sentence1_key, sentence2_key = task_to_keys[task_name]

    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args, padding=padding, max_length=max_seq_length, truncation=truncation
        )

        return result

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        desc=f"Running tokenizer on {task_name}:train",
    )
    validation_dataset = validation_dataset.map(
        preprocess_function,
        batched=True,
        desc=f"Running tokenizer on {task_name}:validation",
    )

    if tokenizer_name.startswith('bert'):
        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )
        validation_dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )
    elif 'roberta' in tokenizer_name or 'distilbert' in tokenizer_name:
        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )
        validation_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )
    else:
        raise NotImplementedError()

    # GLUE tasks are tested on validation set
    return NlpHfDatasetWrapper(train_dataset), \
        NlpHfDatasetWrapper(validation_dataset), \
        NlpHfDatasetWrapper(validation_dataset)


def get_ag_news(tokenizer_name=None, padding="max_length", max_seq_length=128, truncation=True):
    train_dataset = load_dataset("ag_news", split='train', cache_dir=os.environ['TRANSFORMERS_CACHE_DIR'])
    test_dataset = load_dataset("ag_news", split='test', cache_dir=os.environ['TRANSFORMERS_CACHE_DIR'])

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                              use_fast=True,
                                              cache_dir=os.environ['TRANSFORMERS_CACHE_DIR'])

    def preprocess_function(examples):

        result = tokenizer(
            examples['text'], padding=padding, max_length=max_seq_length, truncation=truncation
        )

        return result

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        desc=f"Running tokenizer on ag_news:train",
    )
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        desc=f"Running tokenizer on ag_news:test",
    )

    if tokenizer_name.startswith('bert'):
        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )
        test_dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )
    elif 'roberta' in tokenizer_name or 'distilbert' in tokenizer_name:
        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )
        test_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )
    else:
        raise NotImplementedError()

    # GLUE tasks are tested on validation set
    return NlpHfDatasetWrapper(train_dataset), \
        NlpHfDatasetWrapper(test_dataset), \
        NlpHfDatasetWrapper(test_dataset)


def get_dbpedia14(tokenizer_name=None, padding="max_length", max_seq_length=128, truncation=True):
    train_dataset = load_dataset("dbpedia_14", split='train', cache_dir=os.environ['TRANSFORMERS_CACHE_DIR'])
    test_dataset = load_dataset("dbpedia_14", split='test', cache_dir=os.environ['TRANSFORMERS_CACHE_DIR'])

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                              use_fast=True,
                                              cache_dir=os.environ['TRANSFORMERS_CACHE_DIR'])

    def preprocess_function(examples):

        result = tokenizer(
            examples['content'], padding=padding, max_length=max_seq_length, truncation=truncation
        )

        return result

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        desc=f"Running tokenizer on dbpedia14:train",
    )
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        desc=f"Running tokenizer on dbpedia14:test",
    )

    if tokenizer_name.startswith('bert'):
        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )
        test_dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )
    elif 'roberta' in tokenizer_name or 'distilbert' in tokenizer_name:
        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )
        test_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )
    else:
        raise NotImplementedError()

    # GLUE tasks are tested on validation set
    return NlpHfDatasetWrapper(train_dataset), \
        NlpHfDatasetWrapper(test_dataset), \
        NlpHfDatasetWrapper(test_dataset)


def get_20newsgroups(tokenizer_name=None, padding="max_length", max_seq_length=512, truncation=True):
    newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True, data_home=os.environ['TRANSFORMERS_CACHE_DIR'])
    newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True, data_home=os.environ['TRANSFORMERS_CACHE_DIR'])

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                              use_fast=True,
                                              cache_dir=os.environ['TRANSFORMERS_CACHE_DIR'])

    is_bert = tokenizer_name.startswith('bert')

    def preprocess_function(examples):
        result = tokenizer(
            examples.data,
            add_special_tokens=True,
            return_token_type_ids=is_bert,
            padding=padding,
            max_length=max_seq_length,
            truncation=truncation
        )

        output = {
            'input_ids': torch.tensor(result['input_ids']),
            'attention_mask': torch.tensor(result['attention_mask']),
            'label': torch.tensor(examples.target)
        }
        if is_bert:
            output['token_type_ids'] = torch.tensor(result['token_type_ids'])
        return output

    train_dataset = preprocess_function(newsgroups_train)
    validation_dataset = preprocess_function(newsgroups_test)

    return NlpDictDatasetWrapper(train_dataset), \
        NlpDictDatasetWrapper(validation_dataset), \
        NlpDictDatasetWrapper(validation_dataset)


DATASETS_NAME_MAP = {
    'rte': get_rte,
    'qqp': get_qqp,
    'qnli': get_qnli,
    'mrpc': get_mrpc,
    'sst2': get_sst2,
    '20newsgroups': get_20newsgroups,
    'ag_news': get_ag_news,
    'dbpedia14': get_dbpedia14
}

DATASET_TO_SEQUENCE_LENGTH = {
    'rte': 128,
    'qqp': 128,
    'qnli': 128,
    'mrpc': 128,
    'sst2': 128,
    '20newsgroups': 512,
    'ag_news': 128,
    'dbpedia14': 128
}
DATASET_TO_NUM_CLASSES = {
    'rte': 2,
    'qqp': 2,
    'qnli': 2,
    'mrpc': 2,
    'sst2': 2,
    '20newsgroups': 20,
    'ag_news': 4,
    'dbpedia14': 14,
}
MODEL_TO_TOKENIZER_NAME = {
    'bert_base': 'bert-base-uncased',
    'bert_large': 'bert-large-uncased',
    'roberta_base': 'roberta-base',
    'distilbert_base': 'distilbert-base-uncased',
}
