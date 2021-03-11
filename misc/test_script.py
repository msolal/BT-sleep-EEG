from mne_bids import BIDSPath, read_raw_bids, write_raw_bids 

# bids_root = '/storage/store2/data/mass-bids/SS3/'
# bids_root = '/storage/store2/data/SleepPhysionet-bids/'
bids_root = '/media/pallanca/datapartition/maelys/data/BIDS/'
derivatives = 'preprocessed'
if derivatives is not None:
    bids_root += 'derivatives/'+derivatives+'/'

# subject = '030001'
# subject = '4001EC'
subject = 'ABlo590819'

bids_path = BIDSPath(subject=subject, root=bids_root)
raw = read_raw_bids(bids_path)