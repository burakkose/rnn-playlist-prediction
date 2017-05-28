import numpy as np
import os
from keras.utils.data_utils import get_file
from sklearn.utils import shuffle


class DatasetMode:
    small = "yes_small"
    big = "yes_big"
    complete = "yes_complete"


class DataConstants:
    dataset = "dataset"
    origin = "https://www.cs.cornell.edu/~shuochen/lme/dataset.tar.gz"
    song_hash = "song_hash.txt"
    train = "train.txt"
    test = "test.txt"


def load(mode=DatasetMode.small):
    base_path = get_file(DataConstants.dataset, origin=DataConstants.origin, untar=True)
    base_path = os.path.join(base_path, mode)

    train_path = os.path.join(base_path, DataConstants.train)
    test_path = os.path.join(base_path, DataConstants.test)
    song_path = os.path.join(base_path, DataConstants.song_hash)

    songs = dict(read_song_hash(song_path))
    train, test = read_dataset(train_path, test_path)

    return train, test, songs


def data_augmentation(dataset, future=False):
    x, y = [], []
    for row in dataset:
        row = reversed(row) if future else row
        for idx in range(0, len(row) - 1):
            x.append([e + 1 for e in row[0:idx + 1]])
            y.append(row[idx + 1])
    return shuffle(x, y)


def read_song_hash(path):
    for line in open(path).readlines():
        line = line.split("\t")
        yield (line[0], "| ".join(line[1:]))


def read_dataset(train_path, test_path):
    return prepare_data(readlines(train_path), with_augmentation=True), \
           prepare_data(readlines(test_path))


def prepare_data(data, with_augmentation=False):
    data = [np.array(p, dtype="int64") for p in data if len(p) > 1]
    if with_augmentation:
        return data_augmentation(data)
    return zip(*list(map(lambda row: (row[:-1], row[-1]), data)))


def readlines(path):
    return [line.split() for line in open(path).readlines()[2:]]
