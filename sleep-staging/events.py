# %%
from datasets.mass_bids import MASS_SS3
from datasets.sleep_physionet import SleepPhysionet
from braindecode.datasets import BaseConcatDataset
from mne_bids.stats import count_events
from datautil.preprocess import zscore
from datautil.preprocess import MNEPreproc, NumpyPreproc, preprocess
from datautil.windowers import create_windows_from_events
from datautil.split import train_valid_test_split


# %%
train_valid_ds = MASS_SS3(subject_ids=8, preload=True, crop_wake_mins=30, resample=100)
test_ds = SleepPhysionet(subject_ids=2, preload=True, crop_wake_mins=30)

dataset = BaseConcatDataset([train_valid_ds, test_ds])

# %%
mapping = {  # We merge stages 3 and 4 following AASM standards.
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4
}

high_cut_hz = 30
preprocessors = [NumpyPreproc(fn=lambda x: x * 1e6),
                 MNEPreproc(fn='filter', l_freq=None, h_freq=high_cut_hz)]
preprocess(dataset, preprocessors)

window_size_s = 30
sfreq = int(dataset.datasets[0].raw.info['sfreq'])
window_size_samples = window_size_s * sfreq
windows_dataset = create_windows_from_events(
    dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0,
    window_size_samples=window_size_samples,
    window_stride_samples=window_size_samples, preload=True, mapping=mapping)

preprocess(windows_dataset, [MNEPreproc(fn=zscore)])

# %%


# %%
mapping = {  # We merge stages 3 and 4 following AASM standards.
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4
}

window_size_s = 30
sfreq = int(dataset.datasets[0].raw.info['sfreq'])
window_size_samples = window_size_s * sfreq

windows_dataset = create_windows_from_events(
    dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0,
    window_size_samples=window_size_samples,
    window_stride_samples=window_size_samples, preload=True, mapping=mapping)

# %%
train_set, valid_set, test_set = train_valid_test_split(windows_dataset)


# %%
n_events_per_subject = [len(ds.windows.events) for ds in windows_dataset.datasets]
index_subjects = np.argsort(n_events_per_subject)
fold1 = list(index_subjects[::5])
fold2 = list(index_subjects[1::5])
fold3 = list(index_subjects[2::5])
fold4 = list(index_subjects[3::5])
fold5 = list(index_subjects[4::5])
splitted = windows_dataset.split(by=[fold1, fold2, fold3, fold4, fold5])



# %%
