import json
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import pandas as pd

# Create mapping between classes and their indices
from utils.interface_utils import cprint


def make_index_dict(label_csv):
    classes = pd.read_csv(label_csv)['category'].unique()
    index_lookup = {}
    for i in range(len(classes)):
        index_lookup[classes[i]] = i
    return index_lookup


def make_name_dict(label_csv):
    classes = pd.read_csv(label_csv)['category'].unique()
    name_lookup = {}
    for i in range(len(classes)):
        name_lookup[i] = classes[i]
    return name_lookup


def preemphasis(signal, coeff=0.97):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, audio_cfg, label_csv=None, verbose=True):
        """
        Dataset that manages audio recordings
        :param audio_cfg: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        # load fold's metadata
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.data = data_json['data']
        self.audio_cfg = audio_cfg

        cprint('---------------the {:s} dataloader---------------'.format(self.audio_cfg.get('mode')), verbose=verbose)
        self.melbins = self.audio_cfg.get('num_mel_bins')
        self.freqm = self.audio_cfg.get('freqm')
        self.timem = self.audio_cfg.get('timem')
        cprint('now using following mask: {:d} freq, {:d} time'.format(self.audio_cfg.get('freqm'),
                                                                       self.audio_cfg.get('timem')), verbose=verbose)
        self.mixup = self.audio_cfg.get('mixup')
        cprint('now using mix-up with rate {:f}'.format(self.mixup), verbose=verbose)
        self.dataset = self.audio_cfg.get('dataset')
        cprint('now process ' + self.dataset, verbose=verbose)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_cfg.get('mean')
        self.norm_std = self.audio_cfg.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using
        # src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_cfg.get('skip_norm') if self.audio_cfg.get('skip_norm') else False
        if self.skip_norm:
            cprint('now skip normalization (use it ONLY when you are computing the normalization stats).', verbose)
        else:
            cprint('use dataset mean {:.3f} and std {:.3f} to normalize the input.'
                   .format(self.norm_mean, self.norm_std), verbose=verbose)

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        cprint('number of classes is {:d}'.format(self.label_num), verbose=verbose)

    def _wav2fbank(self, filename, filename2=None):
        # mixup
        if filename2 is None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0,
                                                  frame_shift=10)

        target_length = self.audio_cfg.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 is None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < self.mixup:
            datum = self.data[index]
            # find another sample to mix, also do balance sampling
            # sample the other sample from the uniform distribution
            mix_sample_idx = random.randint(0, len(self.data) - 1)
            mix_datum = self.data[mix_sample_idx]
            # get the mixed fbank
            fbank, mix_lambda = self._wav2fbank(datum['wav'], mix_datum['wav'])
            # initialize the label
            label_indices = np.zeros(self.label_num)
            # add sample 1 labels
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda
            # add sample 2 labels
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += 1.0 - mix_lambda
            label_indices = torch.FloatTensor(label_indices)
        # if not do mixup
        else:
            datum = self.data[index]
            label_indices = np.zeros(self.label_num)
            fbank, mix_lambda = self._wav2fbank(datum['wav'])
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0

            label_indices = torch.FloatTensor(label_indices)

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank, label_indices

    def __len__(self):
        return len(self.data)
