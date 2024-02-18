import tonic
import argparse
import numpy as np


def get_dataset(name, cache_dir):
    if name == 'shd':
        shd_train = tonic.datasets.SHD(save_to=cache_dir, train=True)
        shd_test = tonic.datasets.SHD(save_to=cache_dir, train=False)
        return shd_train, None, shd_test
    elif name == 'ssc':
        ssc_train = tonic.datasets.SSC(save_to=cache_dir, split='train')
        ssc_val = tonic.datasets.SSC(save_to=cache_dir, split='valid')
        ssc_test = tonic.datasets.SSC(save_to=cache_dir, split='test')
        return ssc_train, ssc_val, ssc_test
    elif name == 'dvs_gesture':
        dvs_train = tonic.datasets.DVSGesture(save_to=cache_dir, train=True)
        dvs_test = tonic.datasets.DVSGesture(save_to=cache_dir, train=False)
        return dvs_train, None, dvs_test
    elif name == 'dvs_lip':
        dvs_train = tonic.datasets.DVSLip(save_to=cache_dir, train=True)
        dvs_test = tonic.datasets.DVSLip(save_to=cache_dir, train=False)
        return dvs_train, None, dvs_test


def get_dset_stats(dataset):
    num_samples = len(dataset)
    num_classes = len(dataset.classes)

    seq_len = np.zeros(num_samples, dtype=int)
    for i, sample in enumerate(dataset):
        data, label = sample
        seq_len[i] = len(data['t'])

    return num_samples, num_classes, np.max(seq_len), np.median(seq_len)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--directory", type=str, default="./cache_dir",
                            help="name of directory where data is cached")
    args = argparser.parse_args()

    for dataset in ['shd', 'ssc', 'dvs_gesture']:
        data = get_dataset(dataset, args.directory)
        for dset in data:
            if dset is not None:
                print(f"Dataset: {dataset}")
                num_samples, num_classes, max_seq_len, median_seq_len = get_dset_stats(dset)
                print(f"Number of samples: {num_samples}")
                print(f"Number of classes: {num_classes}")
                print(f"Max sequence length: {max_seq_len}")
                print(f"Median sequence length: {median_seq_len}")


if __name__ == '__main__':
    main()
