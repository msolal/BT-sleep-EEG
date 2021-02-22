def train_valid_test_split(windows_dataset, train=0.6, valid=0.2, test=0.2):
    assert(train+valid+test == 1)
    nb_ds = len(windows_dataset.datasets)
    train_size = int(nb_ds*train)
    valid_size = int(nb_ds*valid)
    train_idx = list(range(train_size))
    valid_idx = list(range(train_size, train_size+valid_size))
    test_idx = list(range(train_size+valid_size, nb_ds))
    if len(train_idx) == 0:
        if len(valid_idx) == 0:
            splitted = windows_dataset.split(by=[test_idx])
            train_set = None
            valid_set = None
            test_set = splitted['0']
        else:
            splitted = windows_dataset.split(by=[valid_idx, test_idx])
            train_set = None
            valid_set = splitted['0']
            test_set = splitted['1']
    elif len(valid_idx) == 0:
        splitted = windows_dataset.split(by=[train_idx, test_idx])
        train_set = splitted['0']
        valid_set = None
        test_set = splitted['1']
    elif len(test_idx) == 0:
        splitted = windows_dataset.split(by=[train_idx, valid_idx])
        train_set = splitted['0']
        valid_set = splitted['1']
        test_set = None
    else:
        splitted = windows_dataset.split(by=[train_idx, valid_idx, test_idx])
        train_set = splitted['0']
        valid_set = splitted['1']
        test_set = splitted['2']
    return train_set, valid_set, test_set
