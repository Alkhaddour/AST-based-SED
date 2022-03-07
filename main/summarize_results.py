import os
import numpy as np

from utils.custom_classes import Dot_dict
from utils.interface_utils import cprint

args = Dot_dict()
# Set path to experiment file
args.exp_path = None  # '../exp/test-esc50-f10-t10-impTrue-aspTrue-b1-lr1e-05-tsts000'
mAP_list = []
acc_list = []
for fold in range(1, 6):
    result = np.loadtxt(os.path.join(args.exp_path, str(fold) + '/result.csv'), delimiter=',')
    if fold == 1:
        cum_result = np.zeros([result.shape[0], result.shape[1]])
    cum_result = cum_result + result
result = cum_result / 5
np.savetxt(os.path.join(args.exp_path, 'result.csv'), result, delimiter=',')
# note this is choose the best epoch based on AVERAGED accuracy across 5 folds, not the best epoch for each fold
best_epoch = np.argmax(result[:, 0])
np.savetxt(os.path.join(args.exp_path,'best_result.csv'), result[best_epoch, :], delimiter=',')

acc_fold = []
cprint('--------------Result Summary--------------', verbose=True)
for fold in range(1, 6):
    result = np.loadtxt(os.path.join(args.exp_path, str(fold) + '/result.csv'), delimiter=',')
    # note this is the best epoch based on AVERAGED accuracy across 5 folds, not the best epoch for each fold (which
    # leads to over-optimistic results), this gives more fair result.
    acc_fold.append(result[best_epoch, 0])
    print('Fold {:d} accuracy: {:.4f}'.format(fold, result[best_epoch, 0]))
acc_fold.append(np.mean(acc_fold))
cprint('The averaged accuracy of 5 folds is {:.3f}'.format(acc_fold[-1]), verbose=True)
np.savetxt(os.path.join(args.exp_path, 'acc_fold.csv'), acc_fold, delimiter=',')
