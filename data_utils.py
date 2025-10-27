import torch
from torchvision import transforms
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset
import numpy as np
import yaml

class WaveformDataset(Dataset):
    def __init__(self, data, labels=False):
        """
        Args:
            data (dict): Dictionary with keys 'waveforms', 'arrival_times', and 'num_photons'.
                - 'waveforms': list or np.ndarray of shape (N, L)
                - 'arrival_times': list or np.ndarray where each entry can be:
                  * a single time (scalar) for one flash
                  * a list/array of times for multiple flashes
                  * None/empty for no flashes
                - 'num_photons': list or np.ndarray where each entry can be:
                  * a single time (scalar) for one hit
                  * a list/array of num_photons for multiple flashes
                  * None/empty for no flashes

        Supports multiple flashes per waveform by creating a binary indicator array
        where multiple time bins can be set to 1.0 for multiple flashes.
        """
        self.labels = labels
        
        # NEW - padding
        max_hits = max(len(arr) for arr in data['arrival_times'])
        arrival_times = self._pad_sequences(data['arrival_times'], max_hits, pad_value=-1)
        nphotons = self._pad_sequences(data['num_photons'], max_hits, pad_value=0)
        
        waveforms = np.asarray(data['waveforms'])
        offset = 0
    
        # Ensure waveforms is 2D: (N, L)
        if waveforms.ndim == 1:
            waveforms = waveforms[:, None]
        elif waveforms.ndim > 2:
            waveforms = waveforms.reshape(waveforms.shape[0], -1)
    
        N, L = waveforms.shape
        assert len(arrival_times) == N, "Mismatch between waveforms and arrival_times length"
    
        # Convert arrival_times to binary indicator array of shape (N, L)
        arrival_bin = np.zeros((N, L), dtype=np.float32)
        photon_bin = np.zeros((N, L), dtype=np.int32)
        label_bins = np.zeros((N, L//10), dtype=np.int32)
        hit_times_list = []
        photon_list = []

        for i, times in enumerate(arrival_times):
            # Handle different input formats
            if times is None or (isinstance(times, (list, np.ndarray)) and len(times) == 0):
                hit_times_list.append([])
                photon_list.append([])
                continue
                
            # Convert to list if it's a single time
            if np.isscalar(times):
                times = [times]
                photons = [nphotons[i]]
            else:
                times = np.asarray(times).flatten()
                photons = np.asarray(nphotons[i]).flatten()
            
            # Store hit times for this waveform
            hit_times_list.append(times)
            photon_list.append(photons)
            
            # Set binary indicators for all flashes in this waveform
            for j, t in enumerate(times):
                t_idx = int(np.clip(t + offset, 0, L - 1))  # Clamp to valid index range, INCLUDING OFFSET FROM WAVEFORM GEN
                arrival_bin[i, t_idx] = 1.0
                photon_bin[i, t_idx] = photons[j]
                
        # Convert to torch tensors
        self.waveforms = torch.from_numpy(waveforms).float()
        self.arrival_times = torch.from_numpy(arrival_bin).float()  # already 2D: (N, L)
        self.photon_per_times = torch.from_numpy(photon_bin).int()
        self.hit_times_list = hit_times_list
        self.photon_list = photon_list
        self.token_labels = torch.tensor(data['token_labels']).int()

    def __len__(self):
        return self.waveforms.shape[0]

    def __getitem__(self, idx):
        if self.labels == True:
            return self.waveforms[idx], self.arrival_times[idx], self.hit_times_list[idx], self.photon_per_times[idx], self.photon_list[idx], self.token_labels[idx]
        else:
            return self.waveforms[idx], self.arrival_times[idx], self.hit_times_list[idx], self.photon_per_times[idx], self.photon_list[idx]
        
    def _pad_sequences(self, seq_list, max_len, pad_value=-1):
        """Pad 1D arrays in seq_list to max_len with pad_value."""
        padded = np.full((len(seq_list), max_len), pad_value, dtype=np.int64)
        for i, seq in enumerate(seq_list):
            seq = np.asarray(seq)
            length = min(len(seq), max_len)
            padded[i, :length] = seq[:length]
        return padded

def load_models(config_path, model_classes, device="cuda"):
    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)

    models = {}
    for name, cfg in configs.items():
        if not isinstance(cfg, dict):
            continue
            
        if not cfg.get("include", True):
            continue
            
        print(name)
        cls = model_classes[cfg["class"]]
        model = cls(**cfg.get("args", {}))
        model.to(device)

        checkpoint = torch.load(cfg["checkpoint"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        models[name] = [model, cfg.get("reg_loss")]
    return models

##### Loading in Data + Make a Single Dataloader #####
def custom_collate_fn(batch):
    """
    Custom collate function for WaveformDataset.
    Each item in batch is a tuple: (waveform, arrival_time).
    Returns:
        waveforms: Tensor of shape (batch_size, waveform_length)
        arrival_times: Tensor of shape (batch_size,) or (batch_size, 1)
        hit_times: Tensor of shape (?) with a list of hit times per sample
    """
    waveforms, arrival_times, hit_times, photon_bins, photon_list = zip(*batch)
    waveforms = torch.stack(waveforms, dim=0)

     # Normalizing waveforms
    waveforms = (waveforms - waveforms.mean(dim=1, keepdim=True)) / (waveforms.std(dim=1, keepdim=True) + 1e-8)
    waveforms = waveforms.unsqueeze(1)  # add channel dimension [B,1,L]

    # for binary classification
    arrival_times = torch.stack(arrival_times, dim=0)
    arrival_times = arrival_times.unsqueeze(1) # adding channel dimension
    photon_bins = torch.stack(photon_bins, dim=0)
    photon_bins = photon_bins.unsqueeze(1)

    # for regression, just use hit times
    hit_times = [item[2] for item in batch]
    hit_times = torch.tensor(hit_times)
    photon_list = [item[4] for item in batch]
    photon_list = torch.tensor(photon_list)
    
    return waveforms, arrival_times, hit_times, photon_bins, photon_list

def labelled_collate_fn(batch):
    """
    !! Custom collate function for WaveformDataset clustering evaluation.
    Each item in batch is a tuple:
    Returns:
        waveforms: Tensor of shape (batch_size, waveform_length)
        arrival_times: Tensor of shape (batch_size,) or (batch_size, 1)
        hit_times: Tensor of shape (?) with a list of hit times per sample
        token_labels: Tensor of shape (batch_size, waveform_length // 10)
    """
    waveforms, arrival_times, hit_times, photon_bins, photon_list, label_list = zip(*batch)
    waveforms = torch.stack(waveforms, dim=0)
    token_labels = torch.stack(label_list, dim=0)

     # Normalizing waveforms
    waveforms = (waveforms - waveforms.mean(dim=1, keepdim=True)) / (waveforms.std(dim=1, keepdim=True) + 1e-8)
    waveforms = waveforms.unsqueeze(1)  # add channel dimension [B,1,L]

    # for binary classification
    arrival_times = torch.stack(arrival_times, dim=0)
    arrival_times = arrival_times.unsqueeze(1) # adding channel dimension
    photon_bins = torch.stack(photon_bins, dim=0)
    photon_bins = photon_bins.unsqueeze(1)

    # for regression, just use hit times
    hit_times = [item[2] for item in batch]
    hit_times = torch.tensor(hit_times)
    photon_list = [item[4] for item in batch]
    photon_list = torch.tensor(photon_list)

    return waveforms, arrival_times, hit_times, photon_bins, photon_list, token_labels


def make_wf_dataloaders(path, batch_size=25, val_ratio=0.1, test_ratio=0.0, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    load_wfs = np.load(path, allow_pickle=True)
    dataset = WaveformDataset(load_wfs.item())

    g = torch.Generator()
    g.manual_seed(seed)

    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)
    train_size = total_size - val_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=g)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        collate_fn=custom_collate_fn,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    return train_loader, val_loader, test_loader

def make_labelled_dataloader(path, batch_size=25, seed=42):
    load_wfs = np.load(path, allow_pickle=True)
    dataset = WaveformDataset(load_wfs.item(), labels=True)

    total_size = len(dataset)
    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        collate_fn=labelled_collate_fn,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    
    return loader

class RandomMasking:
    def __init__(self, masking_p):
        #: masking_p: fraction of time bins to mask
        self.masking_p = masking_p

    def __call__(self, waveform_batch):
        B, C, L = waveform_batch.shape
        mask = torch.rand(B, L, device=waveform_batch.device) < self.masking_p
        mask = mask.unsqueeze(1).expand(-1, C, -1)
        return waveform_batch, waveform_batch * (~mask)