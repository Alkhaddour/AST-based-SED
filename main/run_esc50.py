# Load libraries
import os
import torch
from torch.utils.data import WeightedRandomSampler
from main.dataloader import  AudiosetDataset
from models.ast import ASTModel
from utils.custom_classes import Dot_dict
from utils.interface_utils import cprint
from train_test import train

# Define experiment args (TODO: parse from cmd args)
args = Dot_dict()
args.model = 'ast'
args.dataset = 'esc50'
args.imagenet_pretrain = True
args.audioset_pretrain = True
if args.audioset_pretrain:
    args.lr = 1e-5
else:
    args.lr = 1e-4
args.n_class = 50
args.freqm = 24
args.timem = 96
args.mixup = 0
args.epoch = 25
args.batch_size = 1
args.fstride = 10
args.tstride = 10
args.num_workers = 12
args.n_epochs = 25
args.n_print_steps = 100

# Create experiment stamp and base dir
timestamp = 'ts002'
args.base_exp_dir = f'../exp/test-{args.dataset}-f{args.fstride}-t{args.tstride}-imp{args.imagenet_pretrain}-' \
                    f'asp{args.audioset_pretrain}-b{args.batch_size}-lr{args.lr}-ts{timestamp}'

# Load meta data
label_csv = '../data/ESC-50/esc50.csv'  # '../data/ESC-50/esc_class_labels_indices.csv'  #
os.makedirs(args.base_exp_dir, exist_ok=True)

#
for fold in range(1, 6):
    cprint(f'Processing fold {fold}')
    # set fold's output dir
    args.exp_dir = os.path.join(args.base_exp_dir, str(fold))
    cprint(f"\nCreating experiment directory: {args.exp_dir}")
    os.makedirs(os.path.join(args.exp_dir, 'models'), exist_ok=True)
    # set train and test data
    tr_data = f'../data/ESC-50/meta_folds/esc_train_data_{fold}.json'
    te_data = f'../data/ESC-50/meta_folds/esc_eval_data_{fold}.json'

    cprint('Training an audio spectrogram transformer model')
    # dataset spectrogram mean and std, used to normalize the input
    norm_stats = {'audioset': [-4.2677393, 4.5689974],
                  'esc50': [-6.6268077, 5.358466]}
    target_length = {'audioset': 1024, 'esc50': 512}

    audio_conf = {'num_mel_bins': 128, 'target_length': target_length[args.dataset], 'freqm': args.freqm,
                  'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset, 'mode': 'train',
                  'mean': norm_stats[args.dataset][0], 'std': norm_stats[args.dataset][1]}

    val_audio_conf = {'num_mel_bins': 128, 'target_length': target_length[args.dataset], 'freqm': 0, 'timem': 0,
                      'mixup': 0, 'dataset': args.dataset, 'mode': 'evaluation', 'mean': norm_stats[args.dataset][0],
                      'std': norm_stats[args.dataset][1]}

    train_loader = torch.utils.data.DataLoader(
        AudiosetDataset(tr_data, label_csv=label_csv, audio_cfg=audio_conf),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        AudiosetDataset(te_data, label_csv=label_csv, audio_cfg=val_audio_conf),
        batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    audio_model = ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                           input_tdim=target_length[args.dataset], imagenet_pretrain=args.imagenet_pretrain,
                           audioset_pretrain=args.audioset_pretrain, model_size='base384')

    cprint('Now starting training for {:d} epochs'.format(args.n_epochs))
    train(audio_model, train_loader, val_loader, args)

