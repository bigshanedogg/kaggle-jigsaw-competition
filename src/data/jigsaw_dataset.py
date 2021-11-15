import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader
from src.data.dataset import DatasetFromDir, StreamingDatasetFromDir, DatasetFromFile
from src.utils.common import clean_text, convert_to_tensor, shuffle_dictionary_lists, is_cpu_device

class JigsawDataset(DatasetFromFile):
    __constant__ = ["data_path", "timesteps", "batch_size", "nprocs", "encoding", "extension"]

    def __init__(self, data_path, tokenizer, timesteps, batch_size, device="cpu", nprocs=1, encoding="utf-8", extension="json"):
        DatasetFromFile.__init__(self=self, file_path=data_path, batch_size=batch_size, device=device, nprocs=nprocs, encoding=encoding, extension=extension)
        self.tokenizer = tokenizer
        self.timesteps = timesteps
        self.preprocess()

    def preprocess(self):
        output = []

        for worker, less_toxic_sentence, more_toxic_sentence in tqdm(self.data, initial=0, total=len(self.data), desc="Preprocessing data"):
            output_row = dict()

            # clean_text
            less_toxic_sentence = clean_text(less_toxic_sentence)
            more_toxic_sentence = clean_text(more_toxic_sentence)

            # encode sentence_less_toxic
            _less_toxic_input_ids = self.tokenizer.encode(less_toxic_sentence, add_special_tokens=False)
            less_toxic_input_ids = [self.tokenizer.eos_token_id] + _less_toxic_input_ids + [self.tokenizer.bos_token_id]
            less_toxic_attention_mask = [1] * len(less_toxic_input_ids)

            # encode sentence_more_toxic
            _more_toxic_input_ids = self.tokenizer.encode(more_toxic_sentence, add_special_tokens=False)
            more_toxic_input_ids = [self.tokenizer.eos_token_id] + _more_toxic_input_ids + [self.tokenizer.bos_token_id]
            more_toxic_attention_mask = [1] * len(more_toxic_input_ids)

            # assert max_length
            if len(less_toxic_input_ids) < 1 or len(less_toxic_input_ids) > self.timesteps: continue
            if len(more_toxic_input_ids) < 1 or len(more_toxic_input_ids) > self.timesteps: continue

            # padding
            less_toxic_input_ids += [0] * (self.timesteps - len(less_toxic_input_ids))
            less_toxic_attention_mask += [0] * (self.timesteps - len(less_toxic_attention_mask))
            more_toxic_input_ids += [0] * (self.timesteps - len(more_toxic_input_ids))
            more_toxic_attention_mask += [0] * (self.timesteps - len(more_toxic_attention_mask))

            output_row["less_toxic_input_ids"] = less_toxic_input_ids
            output_row["less_toxic_attention_mask"] = less_toxic_attention_mask
            output_row["more_toxic_input_ids"] = more_toxic_input_ids
            output_row["more_toxic_attention_mask"] = more_toxic_attention_mask
            target = 1
            output_row["labels"] = target
            output.append(output_row)

        self.raw_data = self.data.copy()
        self.data = output
        self.data_size = len(output)


class JigsawDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, device):
        num_workers = 1
        pin_memory = False
        if not is_cpu_device(device):
            num_workers = 2
            pin_memory = True
        DataLoader.__init__(self=self, dataset=dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        self.device = device
        self.collate_fn = self._collate_fn

    def _collate_fn(self, batch, shuffle=True):
        inputs = dict()
        for row in batch:
            for k, v in row.items():
                if k not in inputs: inputs[k] = []
                inputs[k].append(v)

        # shuffle
        if shuffle: inputs = shuffle_dictionary_lists(dictionaries=[inputs])[0]
        return inputs