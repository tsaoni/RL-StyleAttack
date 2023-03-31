import os
import linecache
import torch
import numpy as np
from typing import Callable, Dict, Iterable, List, Union
from pathlib import Path
from torch.utils.data import Dataset, Sampler
from functools import cached_property

from utils import (
    check_variable_status, 
    encode_line, 
    label_to_tensor, 
    trim_batch, 
)

class BaseDataset(Dataset):

    DEFAULT_MAX_SOURCE_LENGTH = 150
    DEFAULT_MAX_TARGET_LENGTH = 150

    def __init__(
        self,
        tokenizer,
        data_dir=None,
        max_source_length=DEFAULT_MAX_SOURCE_LENGTH,
        max_target_length=None,
        type_path="train",
        n_obs=None,
        src_lang=None,
        tgt_lang=None,
        task_mode='sequence-classification',
        task_name=None,
        pad_to_max_len=False, 
    ):
        super().__init__()
        # read file
        check_variable_status(data_dir, name='data_dir', status='None')
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.len_file = Path(data_dir).joinpath(type_path + ".len")
        self.tokenizer = tokenizer
        if os.path.exists(self.len_file):
            self.src_lens = pickle_load(self.len_file)
            self.used_char_len = False
        else:
            self.src_lens = self.get_char_lens(self.src_file)
            self.used_char_len = True
        self.max_source_length = self._calc_max_sequence_len(max_source_length, str(self.src_file), pad_to_max_len=pad_to_max_len, datatype='source')
        self.max_target_length = self._calc_max_sequence_len(max_target_length, str(self.tgt_file), pad_to_max_len=pad_to_max_len, datatype='target')
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"

        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.task_mode = task_mode
        self.task_name = task_name

    def __len__(self):
        return len(self.src_lens)

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    def _calc_max_sequence_len(self, max_length, data_file, pad_to_max_len=False, datatype=None):
        if pad_to_max_len:
            max_length = 0
            lines = linecache.getlines(data_file)
            for line in lines:
                tokens_len = len(self.tokenizer.tokenize(line.rstrip("\n")))
                if tokens_len > max_length:
                    max_length = tokens_len
            print(f'the max length of {datatype} is changed to {max_length}. ')

        return max_length
                
    @cached_property
    def tgt_lens(self):
        """Length in characters of target documents"""
        return self.get_char_lens(self.tgt_file)

    def make_sortish_sampler(self, batch_size, distributed=False, shuffle=True, **kwargs):
        if distributed:
            return DistributedSortishSampler(self, batch_size, shuffle=shuffle, **kwargs)
        else:
            return SortishSampler(self.src_lens, batch_size, shuffle=shuffle)

    def make_dynamic_sampler(self, max_tokens_per_batch=1024, **kwargs):
        assert FAIRSEQ_AVAILABLE, "Dynamic batch size requires `pip install fairseq`"
        assert not self.used_char_len, "You must call  python make_len_file.py before calling make_dynamic_sampler"
        sorted_indices = list(self.make_sortish_sampler(1024, shuffle=False))

        def num_tokens_in_example(i):
            return min(self.src_lens[i], self.max_target_length)

        # call fairseq cython function
        batch_sampler: List[List[int]] = batch_by_size(
            sorted_indices,
            num_tokens_fn=num_tokens_in_example,
            max_tokens=max_tokens_per_batch,
            required_batch_size_multiple=64,
        )
        shuffled_batches = [batch_sampler[i] for i in np.random.permutation(range(len(batch_sampler)))]
        # move the largest batch to the front to OOM quickly (uses an approximation for padding)
        approximate_toks_per_batch = [max(self.src_lens[i] for i in batch) * len(batch) for batch in shuffled_batches]
        largest_batch_idx = np.argmax(approximate_toks_per_batch)
        shuffled_batches[0], shuffled_batches[largest_batch_idx] = (
            shuffled_batches[largest_batch_idx],
            shuffled_batches[0],
        )
        return shuffled_batches

    def __getitem__(self, item):
        raise NotImplementedError("You must implement this")

    def collate_fn(self, batch):
        raise NotImplementedError("You must implement this")

class GeneralDataset(BaseDataset):
    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        if self.task_mode == 'summarization':
            batch_encoding: Dict[str, torch.Tensor] = self.tokenizer.prepare_seq2seq_batch(
                [x["src_texts"] for x in batch],
                # src_lang=self.src_lang,
                tgt_texts=[x["tgt_texts"] for x in batch],
                # tgt_lang=self.tgt_lang,
                max_length=self.max_source_length,
                max_target_length=self.max_target_length,
                return_tensors="pt",
            ).data
            
        elif self.task_mode == 'sequence-classification':
            source_inputs = [encode_line(self.tokenizer, x["src_texts"], \
                            self.max_source_length, dataset_name=self.task_name) for x in batch]
            target_inputs = [label_to_tensor(x['tgt_texts']) for x in batch]

            source_ids = [x["input_ids"].squeeze() for x in source_inputs]
            src_mask = [x["attention_mask"].squeeze() for x in source_inputs]
            # src_token_type_ids = [x["token_type_ids"].squeeze() for x in source_inputs]
            target_ids = target_inputs
                
            input_ids = torch.stack(source_ids)
            masks = torch.stack(src_mask)
            # token_type_ids = torch.stack(src_token_type_ids)
            target_ids = torch.stack(target_ids).squeeze().to(torch.long)
            pad_token_id = self.pad_token_id
            # source_ids, source_mask, source_token_type_ids = trim_batch(input_ids, token_type_ids, pad_token_id, attention_mask=masks)
            # ''' dont use trim_batch '''
            source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
            # source_ids, source_mask = input_ids, masks
            batch_encoding = {
                "input_ids": source_ids,
                "attention_mask": source_mask,
                # "token_type_ids": source_token_type_ids,
                "labels": target_ids,
            }

        batch_encoding.update(
            **dict(
                src_texts=[x["src_texts"] for x in batch], 
                tgt_texts=[x["tgt_texts"] for x in batch], 
            )
        )

        # batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])
        return batch_encoding

class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."

    def __init__(self, data, batch_size, shuffle=True):
        self.data, self.bs, self.shuffle = data, batch_size, shuffle

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(sortish_sampler_indices(self.data, self.bs, shuffle=self.shuffle))


def sortish_sampler_indices(data: List, bs: int, shuffle=True) -> np.array:
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."
    if not shuffle:
        return np.argsort(np.array(data) * -1)

    def key_fn(i):
        return data[i]

    idxs = np.random.permutation(len(data))
    sz = bs * 50
    ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
    sort_idx = np.concatenate([sorted(s, key=key_fn, reverse=True) for s in ck_idx])
    sz = bs
    ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
    max_ck = np.argmax([key_fn(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
    ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
    sort_idx = np.concatenate(np.random.permutation(ck_idx[1:])) if len(ck_idx) > 1 else np.array([], dtype=np.int)
    sort_idx = np.concatenate((ck_idx[0], sort_idx))
    return sort_idx


class DistributedSortishSampler(Sampler):
    """Copied from torch DistributedSampler"""

    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, add_extra_examples=True, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        if add_extra_examples:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(dataset)
            self.num_samples = len(self.available_indices)
        self.batch_size = batch_size
        self.add_extra_examples = add_extra_examples
        self.shuffle = shuffle

    def __iter__(self) -> Iterable:
        g = torch.Generator()
        g.manual_seed(self.epoch)

        sortish_data = [self.dataset.src_lens[i] for i in self.available_indices]
        sortish_indices = sortish_sampler_indices(sortish_data, self.batch_size, shuffle=self.shuffle)
        indices = [self.available_indices[i] for i in sortish_indices]
        assert len(indices) == self.num_samples
        return iter(indices)

    @cached_property
    def available_indices(self) -> np.array:
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        available_indices = indices[self.rank : self.total_size : self.num_replicas]
        return available_indices

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch