# %%
# 0. Setting up the environment

import torch
import numpy as np

from datasets.mass import MASS_SS3
from datautil.preprocess import zscore
from datasets.sleep_physionet import SleepPhysionet
from datautil.windowers import create_windows_from_events
from datautil.preprocess import MNEPreproc, NumpyPreproc, preprocess

# GPU or CPU?
if torch.cuda.is_available():
    print('CUDA-enabled GPU found. Training should be faster.')
    device = 'cuda'
else:
    print('No GPU, training will be carried out on CPU, might be slower')
    device = 'cpu'

# %%
# 1. Loading the data

dataset = SleepPhysionet(subject_ids=[0, 1], recording_ids=[1], crop_wake_mins=30)
# dataset = MASS_SS3(subject_ids=[1, 2], crop_wake_mins=30)


# %%
# 2. Preprocessing

high_cut_hz = 30

preprocessors = [
    # bandpass filter
    MNEPreproc(fn='filter', l_freq=None, h_freq=high_cut_hz),
]

# Transform the data
preprocess(dataset, preprocessors)

# %%
# Extracting windows

mapping = {  # We merge stages 3 and 4 following AASM standards.
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4
}

window_size_s = 30
# sfreq = 100
sfreq = int(dataset.datasets[0].raw.info['sfreq'])
window_size_samples = window_size_s * sfreq

windows_dataset = create_windows_from_events(
    dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0,
    window_size_samples=window_size_samples,
    window_stride_samples=window_size_samples, preload=True, mapping=mapping)

# %%
# Window preprocessing

preprocess(windows_dataset, [MNEPreproc(fn=zscore)])

# %%
# 3. Making train, valid and test splits

# We seed the random number generators to make our splits reproducible
torch.manual_seed(87)
np.random.seed(87)

# Use recordings of the first 10 subjects as test set
test_idx = list(range(10))
test_ds, train_ds = datasets.pick_recordings(dataset, test_idx)

# Split remaining recordings into training and validation sets
n_subjects_valid = max(1, int(len(train_ds.datasets) * 0.2))
train_ds, valid_ds = datasets.train_test_split(train_ds, n_subjects_valid,
                                               split_by='subj_nb')

print('Number of examples in each set:')
print(f'Training: {len(train_ds)}')
print(f'Validation: {len(valid_ds)}')
print(f'Test: {len(test_ds)}')

# Plot imbalance between classes
classes_mapping = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'R'}
y_train = pd.Series([y for _, y in train_ds]).map(classes_mapping)
ax = y_train.value_counts().plot(kind='barh')
ax.set_xlabel('Number of training examples')
ax.set_ylabel('Sleep stage')
ax.figure.savefig(plots_path + '3-class-imbalance')

# Change class weight to balance
train_y = np.concatenate([ds.epochs_labels for ds in train_ds.datasets])
class_weights = compute_class_weight('balanced', classes=np.unique(train_y),
                                     y=train_y)
print(class_weights)

# %%
# 4. Creating the neural network

# Create model
model = models.SleepStagerChambon2018(n_channels, sfreq, n_classes=5)

# Export model to device
print(f'Using device \'{device}\'.')
model = model.to(device)

# %%
# 5. Train and monitor network

# Create dataloaders
train_batch_size = 128  # Important hyperparameter
valid_batch_size = 256  # Can be as large as memory allows, no impact on perf
num_workers = 0  # Nb of processes for data loading process; 0 = main

loader_train = DataLoader(
    train_ds, batch_size=train_batch_size, shuffle=True,
    num_workers=num_workers)
loader_valid = DataLoader(
    valid_ds, batch_size=valid_batch_size, shuffle=False,
    num_workers=num_workers)
loader_test = DataLoader(
    test_ds, batch_size=valid_batch_size, shuffle=False,
    num_workers=num_workers)

optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=0)
criterion = CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))

n_epochs = 10
patience = 5
best_model, history = training.train(
    model, loader_train, loader_valid, optimizer, criterion, n_epochs,
    patience, device, metric=cohen_kappa_score)

# Visualising the learning curves
history_df = pd.DataFrame(history)
ax1 = history_df.plot(x='epoch', y=['train_loss', 'valid_loss'], marker='o')
ax1.set_ylabel('Loss')
ax1.figure.savefig(plots_path + '4-learning-curve-1')
ax2 = history_df.plot(x='epoch', y=['train_perf', 'valid_perf'], marker='o')
ax2.set_ylabel('Cohen\'s kappa')
ax2.figure.savefig(plots_path + 'plots/4-learning-curve-2')

# Compute test performance

best_model.eval()

y_pred_all, y_true_all = list(), list()
for batch_x, batch_y in loader_test:
    batch_x = batch_x.to(device=device, dtype=torch.float32)
    batch_y = batch_y.to(device=device, dtype=torch.int64)
    output = model.forward(batch_x)
    y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
    y_true_all.append(batch_y.cpu().numpy())

y_pred = np.concatenate(y_pred_all)
y_true = np.concatenate(y_true_all)
rec_ids = np.concatenate(  # indicates which recording each example comes from
    [[i] * len(ds) for i, ds in enumerate(test_ds.datasets)])

test_bal_acc = balanced_accuracy_score(y_true, y_pred)
test_kappa = cohen_kappa_score(y_true, y_pred)

print(f'Test balanced accuracy: {test_bal_acc:0.3f}')
print(f'Test Cohen\'s kappa: {test_kappa:0.3f}')


# %%
# 6. Visualising results

conf_mat = confusion_matrix(y_true, y_pred)
visualisation.plot_confusion_matrix(conf_mat, classes_mapping). \
            savefig(plots_path + '5-confusion-matrix')

mask = rec_ids == 0  # pick a recording number

t = np.arange(len(y_true[mask])) * 30 / 3600

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(t, y_true[mask], label='True')
ax.plot(t, y_pred[mask], alpha=0.7, label='Predicted')
ax.set_yticks([0, 1, 2, 3, 4])
ax.set_yticklabels(['W', 'N1', 'N2', 'N3', 'R'])
ax.set_xlabel('Time (h)')
ax.set_title('Hypnogram')
ax.legend()
ax.figure.savefig(plots_path + '6-hypnogram')
