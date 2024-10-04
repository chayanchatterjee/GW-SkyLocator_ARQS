import numpy as np
import math
import pandas as pd
import seaborn as sns
import re
import random
import os
import time
import logging
import h5py
import healpy as hp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from astropy.utils import iers
from astropy.table import Table
from astropy import units as u
from pycbc.detector import Detector
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import normflows as nf

import ligo.skymap.kde as KDE
from ligo.skymap import io
import astropy_healpix as ah
from ligo.skymap.kde import moc
from astropy.table import Table
from astropy import units as u
from ligo.skymap.core import nest2uniq, uniq2nest, uniq2order, uniq2pixarea, uniq2ang
from ligo.skymap.core import rasterize as _rasterize
from pycbc.detector import Detector


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting script...")

# Disable Astropy IERS auto download
iers.conf.auto_download = False

def load_snr_and_injection_data(snr_file, params_file, idx_start=0, idx_end=None):
    with h5py.File(snr_file, 'r') as snr_f, h5py.File(params_file, 'r') as params_f:
        # Load SNR series
        h1_real = np.real(snr_f['h1_snr_series'][idx_start:idx_end][()])
        l1_real = np.real(snr_f['l1_snr_series'][idx_start:idx_end][()])
        v1_real = np.real(snr_f['v1_snr_series'][idx_start:idx_end][()])
        
        h1_imag = np.imag(snr_f['h1_snr_series'][idx_start:idx_end][()])
        l1_imag = np.imag(snr_f['l1_snr_series'][idx_start:idx_end][()])
        v1_imag = np.imag(snr_f['v1_snr_series'][idx_start:idx_end][()])
        
        # Load Injection_SNR and intrinsic parameters
        injection_snr = params_f['Injection_SNR'][idx_start:idx_end][()]
        ra = 2.0 * np.pi * params_f['ra'][idx_start:idx_end][()]
        dec = np.arcsin(1.0 - 2.0 * params_f['dec'][idx_start:idx_end][()])
    
    return h1_real, l1_real, v1_real, h1_imag, l1_imag, v1_imag, injection_snr, ra, dec

# Load data for training
logging.info("Loading SNR and injection data for training...")
h1_real_1, l1_real_1, v1_real_1, h1_imag_1, l1_imag_1, v1_imag_1, injection_snr_1, ra_1, dec_1 = load_snr_and_injection_data(
    '/workspace/LIGO/GW-SkyLocator/data/O3_noise_snr_series_GW170817_BBH_3_det_0_secs_1.hdf',
    '/workspace/LIGO/GW-SkyLocator/data/O3_noise_GW170817_BBH_3_det_parameters_1.hdf')

h1_real_2, l1_real_2, v1_real_2, h1_imag_2, l1_imag_2, v1_imag_2, injection_snr_2, ra_2, dec_2 = load_snr_and_injection_data(
    '/workspace/LIGO/GW-SkyLocator/data/O3_noise_snr_series_GW170817_BBH_3_det_0_secs_2.hdf',
    '/workspace/LIGO/GW-SkyLocator/data/O3_noise_GW170817_BBH_3_det_parameters_2.hdf', idx_start=10000)

h1_real_4, l1_real_4, v1_real_4, h1_imag_4, l1_imag_4, v1_imag_4, injection_snr_4, ra_4, dec_4 = load_snr_and_injection_data(
    '/workspace/LIGO/GW-SkyLocator/data/O3_noise_snr_series_GW170817_BBH_3_det_0_secs_4.hdf',
    '/workspace/LIGO/GW-SkyLocator/data/O3_noise_GW170817_BBH_3_det_parameters_4.hdf')

# Concatenate data from different banks
h1_real = np.concatenate([h1_real_1, h1_real_2, h1_real_4], axis=0)
l1_real = np.concatenate([l1_real_1, l1_real_2, l1_real_4], axis=0)
v1_real = np.concatenate([v1_real_1, v1_real_2, v1_real_4], axis=0)

h1_imag = np.concatenate([h1_imag_1, h1_imag_2, h1_imag_4], axis=0)
l1_imag = np.concatenate([l1_imag_1, l1_imag_2, l1_imag_4], axis=0)
v1_imag = np.concatenate([v1_imag_1, v1_imag_2, v1_imag_4], axis=0)

injection_snr = np.concatenate([injection_snr_1, injection_snr_2, injection_snr_4], axis=0)

# Apply the filter: Only keep samples with Injection SNR > 12
logging.info("Filtering samples with Injection SNR > 12...")
snr_filter = injection_snr > 12

h1_real = h1_real[snr_filter]
l1_real = l1_real[snr_filter]
v1_real = v1_real[snr_filter]
h1_imag = h1_imag[snr_filter]
l1_imag = l1_imag[snr_filter]
v1_imag = v1_imag[snr_filter]

# Apply the same filter to the intrinsic parameters (ra, dec)
ra = np.concatenate([ra_1, ra_2, ra_4], axis=0)
dec = np.concatenate([dec_1, dec_2, dec_4], axis=0)

ra = ra[snr_filter]
dec = dec[snr_filter]

ra = ra - np.pi
ra_x = np.cos(ra)
ra_y = np.sin(ra)

ra = ra[:, None]
ra_x = ra_x[:, None]
ra_y = ra_y[:, None]

dec = dec[:, None]

y_train = np.concatenate((ra_x, ra_y, dec), axis=1).astype('float64')

# Expand dimensions for concatenation
h1_real = h1_real[:, :, None]
l1_real = l1_real[:, :, None]
v1_real = v1_real[:, :, None]

h1_imag = h1_imag[:, :, None]
l1_imag = l1_imag[:, :, None]
v1_imag = v1_imag[:, :, None]

logging.info("Concatenating SNR data...")
X_train_real = np.concatenate((h1_real, l1_real, v1_real), axis=2)
X_train_imag = np.concatenate((h1_imag, l1_imag, v1_imag), axis=2)


logging.info("Loading test SNR series data...")

def load_test_data(test_snr_file, test_params_file, num_test=10000):
    with h5py.File(test_snr_file, 'r') as snr_f, h5py.File(test_params_file, 'r') as params_f:
        # Load SNR series
        h1_test_real = np.real(snr_f['h1_snr_series'][0:num_test][()])
        l1_test_real = np.real(snr_f['l1_snr_series'][0:num_test][()])
        v1_test_real = np.real(snr_f['v1_snr_series'][0:num_test][()])
        
        h1_test_imag = np.imag(snr_f['h1_snr_series'][0:num_test][()])
        l1_test_imag = np.imag(snr_f['l1_snr_series'][0:num_test][()])
        v1_test_imag = np.imag(snr_f['v1_snr_series'][0:num_test][()])
        
        # Expand dimensions
        h1_test_real = h1_test_real[:, :, None]
        l1_test_real = l1_test_real[:, :, None]
        v1_test_real = v1_test_real[:, :, None]
        
        h1_test_imag = h1_test_imag[:, :, None]
        l1_test_imag = l1_test_imag[:, :, None]
        v1_test_imag = v1_test_imag[:, :, None]
        
        # Concatenate real and imaginary parts
        X_test_real = np.concatenate((h1_test_real, l1_test_real, v1_test_real), axis=2)
        X_test_imag = np.concatenate((h1_test_imag, l1_test_imag, v1_test_imag), axis=2)
        
        # Load intrinsic parameters
        ra_test = 2.0 * np.pi * params_f['ra'][0:num_test][()]
        dec_test = np.arcsin(1.0 - 2.0 * params_f['dec'][0:num_test][()])
        gps_time_test = params_f['gps_time'][0:num_test][()]
        
    return X_test_real, X_test_imag, h1_test_real, l1_test_real, v1_test_real, h1_test_imag, l1_test_imag, v1_test_imag, ra_test, dec_test, gps_time_test

def load_test_real_events_data(test_snr_file, test_params_file):
    with h5py.File(test_snr_file, 'r') as snr_f, h5py.File(test_params_file, 'r') as params_f:
        # Load SNR series
        h1_real = np.real(snr_f['h1_snr_series'][()][None,:])
        l1_real = np.real(snr_f['l1_snr_series'][()][None,:])
        v1_real = np.real(snr_f['v1_snr_series'][()][None,:])
        
        h1_imag = np.imag(snr_f['h1_snr_series'][()][None,:])
        l1_imag = np.imag(snr_f['l1_snr_series'][()][None,:])
        v1_imag = np.imag(snr_f['v1_snr_series'][()][None,:])
        
        # Load Injection_SNR and intrinsic parameters
        ra = 2.0 * np.pi * params_f['ra'][()]
        dec = np.arcsin(1.0 - 2.0 * params_f['dec'][()])
        gps_time_test = params_f['gps_time'][()]
    
    return h1_real, l1_real, v1_real, h1_imag, l1_imag, v1_imag, ra, dec, gps_time_test


# Load test data
#X_test_real, X_test_imag, h1_test_real, l1_test_real, v1_test_real, h1_test_imag, l1_test_imag, v1_test_imag, ra_test, dec_test, gps_time_test = load_test_data(
#    "/workspace/LIGO/GW-SkyLocator/data/O3_noise_snr_series_GW170817_BBH_3_det_0_secs_2.hdf",
#    "/workspace/LIGO/GW-SkyLocator/data/O3_noise_GW170817_BBH_3_det_parameters_2.hdf",
#    num_test=10000
#)

# Load real events test data
h1_real_test_1, l1_real_test_1, v1_real_test_1, h1_imag_test_1, l1_imag_test_1, v1_imag_test_1, ra_test_1, dec_test_1, gps_time_test_1 = load_test_real_events_data(
    '/workspace/LIGO/GW-SkyLocator/data/real_event_snr_time_series_GW200224_222234.hdf',
    '/workspace/LIGO/GW-SkyLocator/data/real_event_parameters_GW200224_222234.hdf')

h1_real_test_2, l1_real_test_2, v1_real_test_2, h1_imag_test_2, l1_imag_test_2, v1_imag_test_2, ra_test_2, dec_test_2, gps_time_test_2 = load_test_real_events_data(
    '/workspace/LIGO/GW-SkyLocator/data/real_event_snr_time_series_GW190521.hdf',
    '/workspace/LIGO/GW-SkyLocator/data/real_event_parameters_GW190521.hdf')

# Concatenate data from different banks
h1_real_test = np.concatenate([h1_real_test_1, h1_real_test_2], axis=0)
l1_real_test = np.concatenate([l1_real_test_1, l1_real_test_2], axis=0)
v1_real_test = np.concatenate([v1_real_test_1, v1_real_test_2], axis=0)

h1_imag_test = np.concatenate([h1_imag_test_1, h1_imag_test_2], axis=0)
l1_imag_test = np.concatenate([l1_imag_test_1, l1_imag_test_2], axis=0)
v1_imag_test = np.concatenate([v1_imag_test_1, v1_imag_test_2], axis=0)

# Expand dimensions for concatenation
h1_real_test = h1_real_test[:, :, None]
l1_real_test = l1_real_test[:, :, None]
v1_real_test = v1_real_test[:, :, None]

h1_imag_test = h1_imag_test[:, :, None]
l1_imag_test = l1_imag_test[:, :, None]
v1_imag_test = v1_imag_test[:, :, None]

logging.info("Concatenating SNR data...")
X_test_real = np.concatenate((h1_real_test, l1_real_test, v1_real_test), axis=2)
X_test_imag = np.concatenate((h1_imag_test, l1_imag_test, v1_imag_test), axis=2)

logging.info("Loading test intrinsic parameters...")
# Compute RA and Dec for test data

ra_test = np.concatenate([ra_test_1, ra_test_2], axis=0)
dec_test = np.concatenate([dec_test_1, dec_test_2], axis=0)

ra_test = ra_test - np.pi
ra_x_test = np.cos(ra_test)
ra_y_test = np.sin(ra_test)

ra_test = ra_test[:, None]
ra_x_test = ra_x_test[:, None]
ra_y_test = ra_y_test[:, None]

dec_test = dec_test[:, None]

gps_time_test = np.concatenate([gps_time_test_1, gps_time_test_2])

y_test = np.concatenate((ra_x_test, ra_y_test, dec_test), axis=1).astype('float64')

# Function to standardize the real part of the SNR time series sample by sample.
logging.info("Standardizing data...")

def standardize_data(X_train_real, X_test_real):
    # Compute standard deviation along the sample dimension
    X_train_real_std = X_train_real.std(axis=1, keepdims=True)
    X_test_real_std = X_test_real.std(axis=1, keepdims=True)
    
    # Avoid division by zero
    X_train_real_std[X_train_real_std == 0] = 1.0
    X_test_real_std[X_test_real_std == 0] = 1.0
    
    X_train_real_standardized = X_train_real / X_train_real_std
    X_test_real_standardized = X_test_real / X_test_real_std
    
    return X_train_real_standardized, X_test_real_standardized

X_train_real, X_test_real = standardize_data(X_train_real, X_test_real)

# Stack real and imaginary parts
X_train = np.hstack((X_train_real, X_train_imag))
X_test = np.hstack((X_test_real, X_test_imag))

# Compute amplitude ratios and phase lags at the merger
logging.info("Computing amplitude ratios and phase lags at merger...")

def compute_metrics_at_merger(h1_real, l1_real, v1_real, h1_imag, l1_imag, v1_imag):
    # Find the indices of the peak (merger) points in each detector
    merger_idx_h1 = np.argmax(np.abs(h1_real), axis=1)
    merger_idx_l1 = np.argmax(np.abs(l1_real), axis=1)
    merger_idx_v1 = np.argmax(np.abs(v1_real), axis=1)

    # Precompute indices for gathering values at the merger
    idx_range = np.arange(h1_real.shape[0])

    # Efficiently gather values at the merger points
    h1_real_merger = h1_real[idx_range, merger_idx_h1]
    l1_real_merger = l1_real[idx_range, merger_idx_l1]
    v1_real_merger = v1_real[idx_range, merger_idx_v1]
    
    h1_imag_merger = h1_imag[idx_range, merger_idx_h1]
    l1_imag_merger = l1_imag[idx_range, merger_idx_l1]
    v1_imag_merger = v1_imag[idx_range, merger_idx_v1]

    # Compute time delays using precomputed indices
    time_diff_h1_l1 = merger_idx_h1 - merger_idx_l1
    time_diff_h1_v1 = merger_idx_h1 - merger_idx_v1
    time_diff_l1_v1 = merger_idx_l1 - merger_idx_v1

    time_diffs = np.stack([time_diff_h1_l1, time_diff_h1_v1, time_diff_l1_v1], axis=1)

    # Compute amplitude ratios at the merger
    amp_ratios = np.stack([h1_real_merger / l1_real_merger, 
                           h1_real_merger / v1_real_merger, 
                           l1_real_merger / v1_real_merger], axis=1)

    # Compute phase lags at the merger using the imaginary parts (Fourier transforms)
    phase_h1 = np.angle(h1_imag_merger)
    phase_l1 = np.angle(l1_imag_merger)
    phase_v1 = np.angle(v1_imag_merger)
    
    phase_lags = np.stack([phase_h1 - phase_l1, phase_h1 - phase_v1, phase_l1 - phase_v1], axis=1)

    # Combine all metrics into a single array
    combined_metrics = np.concatenate([time_diffs, amp_ratios, phase_lags], axis=1)
    
    # Clear unnecessary variables from memory
    del h1_real_merger, l1_real_merger, v1_real_merger
    del h1_imag_merger, l1_imag_merger, v1_imag_merger
    
    return combined_metrics

# Compute the new input features (amplitude ratios and phase lags at merger)
metrics_train = compute_metrics_at_merger(h1_real.squeeze(), l1_real.squeeze(), v1_real.squeeze(), 
                                         h1_imag.squeeze(), l1_imag.squeeze(), v1_imag.squeeze())
metrics_test = compute_metrics_at_merger(h1_real_test.squeeze(), l1_real_test.squeeze(), v1_real_test.squeeze(), 
                                         h1_imag_test.squeeze(), l1_imag_test.squeeze(), v1_imag_test.squeeze())

# Function to standardize the metrics
def scale_labels(data_train, data_test):
    mean = data_train.mean(axis=0)
    std = data_train.std(axis=0)
    std[std == 0] = 1.0  # Avoid division by zero
    data_train_standardized = (data_train - mean) / std
    data_test_standardized = (data_test - mean) / std
    return data_train_standardized, data_test_standardized, mean, std

metrics_train, metrics_test, metrics_mean, metrics_std = scale_labels(metrics_train, metrics_test)

logging.info("Metrics standardized.")

# Convert all data to torch tensors
logging.info("Converting data to torch tensors...")
X_train = torch.tensor(X_train, dtype=torch.float64).to(device)
X_train_real = torch.tensor(X_train_real, dtype=torch.float64).to(device)
X_train_imag = torch.tensor(X_train_imag, dtype=torch.float64).to(device)
y_train = torch.tensor(y_train, dtype=torch.float64).to(device)
metrics_train = torch.tensor(metrics_train, dtype=torch.float64).to(device)

X_test = torch.tensor(X_test, dtype=torch.float64).to(device)
X_test_real = torch.tensor(X_test_real, dtype=torch.float64).to(device)
X_test_imag = torch.tensor(X_test_imag, dtype=torch.float64).to(device)
y_test = torch.tensor(y_test, dtype=torch.float64).to(device)
metrics_test = torch.tensor(metrics_test, dtype=torch.float64).to(device)

class GWDataset(Dataset):
    def __init__(self, X, metrics, y):
        self.X = X
        self.metrics = metrics
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.metrics[idx], self.y[idx]

# Create Dataset and DataLoader
logging.info("Creating Dataset and DataLoader...")
dataset = GWDataset(X_train, metrics_train, y_train)
batch_size = 256
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation=nn.ReLU()):
        super(ResidualUnit, self).__init__()
        self.activation = activation
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            self.activation,
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        return self.activation(self.main(x) + self.skip(x))

class ResNet34Encoder(nn.Module):
    def __init__(self, input_channels, metrics_dim):
        super(ResNet34Encoder, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Define Residual Layers
        self.layer1 = self._make_layer(32, 32, blocks=3, stride=1)
        self.layer2 = self._make_layer(32, 64, blocks=4, stride=2)
        self.layer3 = self._make_layer(64, 128, blocks=6, stride=2)
        self.layer4 = self._make_layer(128, 256, blocks=3, stride=2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        
        # Metrics MLP
        self.metrics_mlp = nn.Sequential(
            nn.Linear(metrics_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64)
        )
        
        # Merge and Output
        self.merge = nn.Sequential(
            nn.Sigmoid(),
            nn.LayerNorm(64 + 256)
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualUnit(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualUnit(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x, metrics):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        
        metrics = self.metrics_mlp(metrics)
        
        merged = torch.cat([x, metrics], dim=1)
        out = self.merge(merged)
        return out

class NormalizingFlowModel(nn.Module):
    def __init__(self, encoder_output_dim, num_features=3, num_layers=5):
        super(NormalizingFlowModel, self).__init__()
        
        self.encoder_output_dim = encoder_output_dim
        self.num_features = num_features
        
        latent_size = 3
        hidden_units = 256
        hidden_layers = 3

        # Set base distribution
        q0 = nf.distributions.DiagGaussian(latent_size, trainable=False)

        flows = []
        for i in range(num_layers):
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, 
                                                             num_context_channels=self.encoder_output_dim)]

            flows += [nf.flows.LULinearPermute(latent_size)]

    
        # Construct flow model
        self.flow = nf.ConditionalNormalizingFlow(q0, flows).to(device)


    def forward(self, x, context):
        return self.flow.log_prob(x, context=context)

    def sample(self, num_samples, context):
        
        # Repeat the context for each sample in the batch dimension (num_samples)
        expanded_features = context.repeat(num_samples, 1)  # Shape: [num_samples, context_dim]

        # Sample from the normalizing flow conditioned on the repeated features
        samples = self.flow.sample(num_samples, context=expanded_features)[0].detach().cpu().numpy()

        # Convert back to torch tensor if needed
        return torch.tensor(samples, dtype=torch.float64, device=device)


logging.info("Building ResNet34 encoder and Normalizing Flow models...")

# Instantiate the encoder
input_channels = X_train.shape[1]  # 820 (assuming 820 features)
metrics_dim = metrics_train.shape[1]  # 9 (time diffs + amp ratios + phase lags)
encoder = ResNet34Encoder(input_channels, metrics_dim).to(device).double()

# Instantiate the normalizing flow
encoder_output_dim = 64 + 256  # From ResNet and metrics MLP
flow_model = NormalizingFlowModel(
    encoder_output_dim=encoder_output_dim,
    num_features=3,
    num_layers=5
).to(device).double()

# Verify encoder parameters
for name, param in encoder.named_parameters():
    assert param.dtype == torch.float64, f"Encoder parameter {name} is not float64."

# Verify flow model parameters
for name, param in flow_model.named_parameters():
    assert param.dtype == torch.float64, f"Flow model parameter {name} is not float64."


# Define optimizer
optimizer = optim.Adam(list(encoder.parameters()) + list(flow_model.parameters()), lr=1e-4, weight_decay=1e-5)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=30, verbose=True)

# Early stopping parameters
early_stopping_patience = 35
best_val_loss = np.inf
epochs_no_improve = 0
num_epochs = 30

# Split dataset into training and validation
logging.info("Splitting dataset into training and validation sets...")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Lists to store loss history
train_losses = []
val_losses = []

logging.info("Starting training...")

for epoch in range(num_epochs):
    encoder.train()
    flow_model.train()
    running_loss = 0.0
    for batch_X, batch_metrics, batch_y in train_loader:
        optimizer.zero_grad()
        
        # Forward pass through encoder
        encoded_features = encoder(batch_X, batch_metrics)
        
        # Compute log_prob
        log_prob = flow_model(batch_y, context=encoded_features)
        loss = -log_prob.mean()
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_X.size(0)
    
    epoch_loss = running_loss / train_size
    train_losses.append(epoch_loss)
    
    # Validation
    encoder.eval()
    flow_model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for val_X, val_metrics, val_y in val_loader:
            encoded_features = encoder(val_X, val_metrics)
            log_prob = flow_model(val_y, context=encoded_features)
            loss = -log_prob.mean()
            val_running_loss += loss.item() * val_X.size(0)
    
    val_epoch_loss = val_running_loss / val_size
    val_losses.append(val_epoch_loss)
    
    logging.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}")
    
    # Step the scheduler
    scheduler.step(val_epoch_loss)
    
    # Early stopping
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        epochs_no_improve = 0
        # Save the best model
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'flow_state_dict': flow_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': train_losses,
            'val_loss': val_losses,
        }, 'best_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_patience:
            logging.info("Early stopping triggered.")
            break

# Load the best model
logging.info("Loading the best model from checkpoint...")
checkpoint = torch.load('best_model.pth')
encoder.load_state_dict(checkpoint['encoder_state_dict'])
flow_model.load_state_dict(checkpoint['flow_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
train_losses = checkpoint['train_loss']
val_losses = checkpoint['val_loss']

def plot_loss_curves(train_losses, val_losses, save_path='/workspace/LIGO/GW-SkyLocator/results/Loss_curve_new.png'):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, "r--", label="Loss on training data")
    plt.plot(val_losses, "r", label="Loss on validation data")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(save_path, dpi=400)
#    plt.show()

logging.info("Plotting loss curves...")
plot_loss_curves(train_losses, val_losses)

# Save encoder and flow models
logging.info("Saving encoder and flow model weights...")
torch.save(encoder.state_dict(), '/workspace/LIGO/GW-SkyLocator/Saved_models/ResNet34_BBH_weights_new_PyTorch.pth')
torch.save(flow_model.state_dict(), '/workspace/LIGO/GW-SkyLocator/Saved_models/NF_BBH_weights_new_PyTorch.pth')

def bayestar_adaptive_grid_new(probdensity, flow, preds, delta, top_nside=16, rounds=8):
    """
    Create a sky map by evaluating a function on an adaptive grid.

    Perform the BAYESTAR adaptive mesh refinement scheme as described in
    Section VI of Singer & Price 2016, PRD, 93, 024013
    :doi:10.1103/PhysRevD.93.024013. This computes the sky map
    using a provided analytic function and refines the grid, dividing the
    highest 25% into subpixels and then recalculating their values. The extra
    given args and kwargs will be passed to the given probdensity function.

    Parameters
    ----------
    probdensity : callable
        Probability density function. The first argument consists of
        column-stacked array of right ascension and declination in radians.
        The return value must be a 1D array of the probability density in
        inverse steradians with the same length as the argument.
    top_nside : int
        HEALPix NSIDE resolution of initial evaluation of the sky map
    rounds : int
        Number of refinement rounds, including the initial sky map evaluation

    Returns
    -------
    skymap : astropy.table.Table
        An astropy Table with UNIQ and PROBDENSITY columns, representing
        a multi-ordered sky map
    probs : list
        List of probability densities at each refinement round
    """
    probs = []
    top_npix = ah.nside_to_npix(top_nside)
    nrefine = top_npix // 4
    cells = list(zip([0] * nrefine, [top_nside // 2] * nrefine, range(nrefine)))
    
    for iround in range(rounds - 1):
        logging.info(f'adaptive refinement round {iround+1} of {rounds-1} ...')
        # Sort cells based on probability density
        cells_sorted = sorted(cells, key=lambda p_n_i: p_n_i[0] / (p_n_i[1]**2))
        # Refine the top nrefine cells
        new_nside, new_ipix = [], []
        for _, nside, ipix in cells_sorted[-nrefine:]:
            for i in range(4):
                new_nside.append(nside * 2)
                new_ipix.append(ipix * 4 + i)
        
        theta, phi = hp.pix2ang(new_nside, new_ipix, nest=True)
        ra = phi
        ra = np.mod(ra - delta, 2.0 * np.pi)
        dec = 0.5 * np.pi - theta
        
        ra = ra - np.pi
        ra_x = np.cos(ra)
        ra_y = np.sin(ra)
        
        pixels = np.stack([ra_x, ra_y, dec], axis=1)
        pixels = torch.tensor(pixels, dtype=torch.float64).to(device)
        
        # Compute probability density
        p = probdensity(flow, pixels, preds)
        probs.append(p)
        
        # Update the refined cells with new probabilities
        cells_sorted[-nrefine:] = list(zip(p, new_nside, new_ipix))
    
    # Normalize probabilities
    post, nside, ipix = zip(*cells_sorted)
    post = np.array(post)
    nside = np.array(nside)
    ipix = np.array(ipix)
    post /= np.sum(post * hp.nside2pixarea(nside).astype(float))
    
    # Convert from NESTED to UNIQ pixel indices
    order = np.log2(nside).astype(int)
    uniq = nest2uniq(order.astype(np.int8), ipix)
    
    return Table([uniq, post], names=['UNIQ', 'PROBDENSITY'], copy=False), probs

def nf_prob_density(flow, pixels, preds):
    """
    Compute probability density using the Normalizing Flow model.

    Parameters
    ----------
    flow : NormalizingFlowModel
        The trained normalizing flow model
    pixels : torch.Tensor
        Tensor of shape [num_pixels, 3]
    preds : torch.Tensor
        Tensor of shape [batch_size, encoder_output_dim]

    Returns
    -------
    prob_density : numpy.ndarray
        Probability density values for each pixel
    """
    with torch.no_grad():
        # Repeat the context for each sample in the batch dimension (num_samples)
        expanded_features = preds.repeat(len(pixels), 1)  # Shape: [num_samples, context_dim]
        log_prob = flow(pixels, context=expanded_features)
        prob_density = torch.exp(log_prob).cpu().numpy()
    return prob_density

# Load the best model
logging.info("Loading the best model for inference...")
checkpoint = torch.load('best_model.pth')
encoder.load_state_dict(checkpoint['encoder_state_dict'])
flow_model.load_state_dict(checkpoint['flow_state_dict'])
encoder.eval()
flow_model.eval()

logging.info("Starting inference and sky map generation...")

for i in range(2):
    logging.info(f"Processing test sample {i+1}/2...")
    x_test = X_test[i].unsqueeze(0)  # Shape: [1, 820]
    metrics_params = metrics_test[i].unsqueeze(0)  # Shape: [1, 9]
    
    # HEALPix parameters
    nside = 16
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix), nest=True)
    
    ra_pix = phi
    dec_pix = -theta + 0.5 * np.pi
    
    gps_time_GW170817 = 1187008882.4  # Fixed merger time
    
    # Calculating the difference in GPS time between the fixed merger time used in training 
    # and the variable merger times of the test samples
    detector = Detector('H1')
    delta = detector.gmst_estimate(gps_time_test[i].item()) - detector.gmst_estimate(gps_time_GW170817)
    
    # Translation of the RA angles to account for the change in merger times
    ra_pix = np.mod(ra_pix - delta, 2.0 * np.pi)
    
    ra_pix = ra_pix - np.pi
    ra_pix_x = np.cos(ra_pix)
    ra_pix_y = np.sin(ra_pix)
    
    pixels = np.stack([ra_pix_x, ra_pix_y, dec_pix], axis=1)
#    pixels = (pixels - y_mean)/y_std
    pixels = torch.tensor(pixels, dtype=torch.float64).to(device)
    
    with torch.no_grad():
        preds = encoder(x_test, metrics_params)
        prob_density = nf_prob_density(flow_model, pixels, preds)
    
    # Apply adaptive refinement
    hpmap, probs_nf_sample = bayestar_adaptive_grid_new(nf_prob_density, flow_model, preds, delta)
    
    # Save the sky map
    sky_map_path = f'/workspace/LIGO/GW-SkyLocator/results/Test_real_events_PyTorch_30_epochs_{i}.fits'
    io.fits.write_sky_map(sky_map_path, hpmap, nest=True)
    logging.info(f"Skymap saved to {sky_map_path}")
