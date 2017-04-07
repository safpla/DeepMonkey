import h5py
import numpy as np


class DataSet(object):
    def __init__(self, filename, num_classes):
        f = h5py.File(filename, "r")
        self._dataset = f["monkeyData"]
        self._labelset = f["monkeyLabel"]
        self._num_examples = self._labelset.len();
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_classes = num_classes
        self._index = np.arange(self._num_examples)
        self._shape_of_data = np.shape(self._dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
    def labelset(self):
        return self._labelset

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            np.random.shuffle(self._index)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        batch_index = self._index[start:end]
        batch_index = list(np.sort(batch_index))
        label = self._labelset[batch_index]
        label_sparse = np.zeros([batch_size, self._num_classes])
        data = np.zeros([batch_size, 3, self._shape_of_data[2], self._shape_of_data[3]])
        data_raw = self._dataset[batch_index]
        data[:, 0:2, :, :] = data_raw
        data[:, 2, :, :] = data_raw[:, 0, :, :] - data_raw[:, 1, :, :]
        data = np.transpose(data, (0, 3, 2, 1))
        for i in xrange(batch_size):
            label_sparse[i][label[i]] = 1
        return data, label_sparse

    def shuffle(self):
        np.random.shuffle(self._index)
        return



