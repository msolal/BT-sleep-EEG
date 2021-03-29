# %%

from mne_bids import BIDSPath, read_raw_bids, write_raw_bids
import os

# bids_root = '/storage/store2/data/mass-bids/SS3/'
bids_root = '/storage/store2/derivatives/Physionet/preprocessed/'
# bids_root = '/media/pallanca/datapartition/maelys/data/BIDS/'
# derivatives = 'preprocessed'
# if derivatives is not None:
#     bids_root += 'derivatives/'+derivatives+'/'

# subject = '030001'
subject = 'SC400'
# subject = 'ABlo590819'

bids_path = BIDSPath(subject=subject, root=bids_root)

sessions = os.listdir(f'{bids_path.root}/sub-{bids_path.subject}')
for session in sessions:
    bids_path.update(session=session[4:])
    raw = read_raw_bids(bids_path)


# %%
