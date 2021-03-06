from itertools import chain, islice
from torch.utils.data import Dataset, IterableDataset
from src.data.interface import DatasetInterface
from src.utils.common import read_file, convert_to_tensor

class DatasetFromObject(Dataset, DatasetInterface):
    def __init__(self, data, batch_size, shuffle=False, device=None, nprocs=1):
        self.data = data
        self.shuffle = shuffle
        self.data_size = len(self.data)
        DatasetInterface.__init__(self=self, batch_size=batch_size, device=device, nprocs=nprocs)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        output = self.data[idx]
        return output

class DatasetFromDir(Dataset, DatasetInterface):
    def __init__(self, data_dir, batch_size, device="cpu", nprocs=1, encoding="utf-8", extension="json"):
        if data_dir is not None and not data_dir.endswith("/"): data_dir += "/"
        self.data_dir = data_dir
        self.file_path_list = self.get_file_path_list(data_dir=data_dir, extension=extension)
        self.encoding = encoding
        self.extension = extension
        self.data = self.get_all_data()
        self.data_size = len(self.data)
        DatasetInterface.__init__(self=self, batch_size=batch_size, device=device, nprocs=nprocs)

    def get_all_data(self):
        data = []
        for file_path in self.file_path_list:
            rows = read_file(file_path=file_path, extension=self.extension, encoding=self.encoding)
            data += rows
        return data

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        row = self.data[idx]
        row = {k: convert_to_tensor(v, self.device) for k, v in row.items()}
        return row

class StreamingDatasetFromDir(IterableDataset, DatasetInterface):
    def __init__(self, data_dir, batch_size, device="cpu", nprocs=1, encoding="utf-8", extension="json"):
        if data_dir is not None and not data_dir.endswith("/"): data_dir += "/"
        self.data_dir = data_dir
        self.file_path_list = self.get_file_path_list(data_dir=data_dir, extension=extension)
        self.encoding = encoding
        self.extension = extension
        self.data_size = self.count_lines()
        DatasetInterface.__init__(self=self, batch_size=batch_size, device=device, nprocs=nprocs)

    def count_lines(self):
        cnt = 0
        data = []
        for file_path in self.file_path_list:
            rows = read_file(file_path=file_path, extension=self.extension, encoding=self.encoding)
            data += rows
            cnt += len(rows)
        return cnt

    def get_all_data(self):
        data = []
        for file_path in self.file_path_list:
            rows = read_file(file_path=file_path, extension=self.extension, encoding=self.encoding)
            data += rows
        return data

    def get_stream(self, file_path_list, start, end):
        return islice(chain.from_iterable(map(self.parse_file, file_path_list)), start, end)

    def __len__(self):
        return self.data_size

    def __iter__(self):
        if not self.iter_range_update: self.set_iter_range()
        return self.get_stream(self.file_path_list, start=self.iter_start, end=self.iter_end)


class DatasetFromFile(IterableDataset, DatasetInterface):
    def __init__(self, file_path, batch_size, delimiter="\t", encoding="utf-8", extension="json", device=None, nprocs=1):
        self.file_path = file_path
        self.delimiter = delimiter
        self.encoding = encoding
        self.extension = extension
        self.data = self.get_all_data()
        self.data_size = len(self.data)
        DatasetInterface.__init__(self=self, batch_size=batch_size, device=device, nprocs=nprocs)

    def count_lines(self):
        rows = read_file(file_path=self.file_path, extension=self.extension, encoding=self.encoding)
        cnt = len(rows)
        return cnt

    def get_all_data(self):
        data = read_file(file_path=self.file_path, extension=self.extension, encoding=self.encoding)
        return data

    def get_stream(self, file_path, start, end):
        return islice(self.parse_file(file_path), start, end)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        row = self.data[idx]
        row = {k: convert_to_tensor(v, self.device) for k, v in row.items()}
        return row