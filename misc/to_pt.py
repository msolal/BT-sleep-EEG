# %%
from datasets.mass_bids import MASS_SS3
from datautil.preprocess import zscore
from datautil.preprocess import MNEPreproc, NumpyPreproc, preprocess
from datautil.windowers import create_windows_from_events
from braindecode.datautil import save_concat_dataset

# %%
dataset = MASS_SS3(subject_ids=['030001'], crop_wake_mins=30)

# %%
high_cut_hz = 30
preprocessors = [
    NumpyPreproc(fn=lambda x: x * 1e6),
    MNEPreproc(fn='filter', l_freq=None, h_freq=high_cut_hz),
]
preprocess(dataset, preprocessors)

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

preprocess(windows_dataset, [MNEPreproc(fn=zscore)])
# %%
save_concat_dataset('preproc/MASS_SS3/', windows_dataset)
# %%
