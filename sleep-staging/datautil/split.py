def train_valid_test_split(windows_dataset, train=0.6, valid=0.2, test=0.2):
    assert(train+valid+test == 1)
    nb_ds = len(windows_dataset.datasets)
    train_size = int(nb_ds*train)
    valid_size = int(nb_ds*valid)
    train_idx = list(range(train_size))
    valid_idx = list(range(train_size, train_size+valid_size))
    test_idx = list(range(train_size+valid_size, nb_ds))
    splitted = windows_dataset.split(by=[train_idx, valid_idx, test_idx])
    train_set = splitted['0']
    valid_set = splitted['1']
    test_set = splitted['2']
    return train_set, valid_set, test_set