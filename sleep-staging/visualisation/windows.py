import numpy as np

def view_nb_windows(plots_path, train_set, valid_set, test_set):
    train_set_stats = view_nb_windows_aux(train_set)
    valid_set_stats = view_nb_windows_aux(valid_set)
    test_set_stats = view_nb_windows_aux(test_set)
    windows = (f'Number of windows in each set:\n\nTraining set: {train_set_stats}\nValidation set: {valid_set_stats}\nTest set: {test_set_stats}')
    windows_file = open(plots_path+'windows.txt', 'w')
    windows_file.write(windows)
    windows_file.close()
    return windows


def view_nb_windows_aux(myset):
    if myset is not None:
        event_id = myset.datasets[0].windows.event_id
        train_set_windows = sum(np.bincount(dataset.windows.events[:, -1], minlength=6) for dataset in myset.datasets)
        train_set_stats = f'{sum(train_set_windows)} events: \n'
        for event_name in event_id.keys():
            train_set_stats += f'{event_name}: {train_set_windows[event_id[event_name]]}\n'
        return train_set_stats
    else: 
        return 'None\n'