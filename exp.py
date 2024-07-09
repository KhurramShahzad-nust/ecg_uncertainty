"""
uncertainty

Shenda Hong, July, 2020
"""

import os
import pickle
import random
import sys
import warnings
from tqdm import tqdm
from shutil import copyfile
from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from scipy import io as sio
from scipy.stats import entropy, pearsonr, spearmanr, ttest_ind

plt.rcParams['pdf.fonttype'] = 42

label_map = {0:'N', 1:'AF', 2:'I-AVB', 3:'LBBB', 4:'RBBB', 5:'PAC', 6:'PVC', 7:'STD', 8:'STE'}

def plot_ecg(data, title='', fs=500):
    """
    data has shape: (n_lead, n_length)
    """
    names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    n_lead, n_length = data.shape[0], data.shape[1]
    x_margin = 200
    y_margin = 1
    gap = 2.9/2
    x = n_length + 2*x_margin
    y = (n_lead-1)*gap + 2*y_margin
    
    base_x = np.array([0,0.04,0.04,0.12,0.12,0.4]) * fs - 2*x_margin//3
    base_y = np.array([0,0,1.0,1.0,0,0])

    height = 10
    width = 10*(x/fs/0.2)/(y/0.5)
    
    fig = plt.figure(figsize=(width,height), dpi=400)
    ax = fig.add_subplot(1, 1, 1)
    for i in range(n_lead):
        ax.plot(data[i]+gap*(11-i), 'k', linewidth=1)
        ax.annotate(names[i], (-x_margin//3, gap*(11-i)+0.5))
        ax.plot(base_x, base_y+gap*(11-i), 'k', linewidth=1)
    
    major_x_ticks = np.arange(-x_margin, n_length+x_margin, fs*0.2)
    minor_x_ticks = np.arange(-x_margin, n_length+x_margin, fs*0.04)
    major_y_ticks = np.arange(-y_margin, n_lead*gap+y_margin, 0.5)
    minor_y_ticks = np.arange(-y_margin, n_lead*gap+y_margin, 0.1)
    ax.set_xticks(major_x_ticks)
    ax.set_yticks(major_y_ticks)
    ax.tick_params(colors='w')
    ax.set_title(title)

    ax.grid(True, which='both', color='#FF8C00', linewidth=0.5) # '#CC5500'
    ax.set_xlim([-x_margin, n_length+x_margin//2])
    ax.set_ylim([-y_margin, n_lead*gap+y_margin//2])

    x_mesh = np.arange(-x_margin, n_length+x_margin//2, fs*0.04)
    y_mesh = np.arange(-y_margin, n_lead*gap+y_margin//2, 0.1)
    xv, yv = np.meshgrid(x_mesh, y_mesh)
    plt.scatter(xv, yv, s=0.1, color='#FF8C00')

    return fig

def get_report(gt, pred):
    tmp_report = classification_report(gt, pred, zero_division=1, output_dict=True)
    label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    f1_list = []
    for i in label_list:
        if str(i) in tmp_report:
            f1_list.append(tmp_report[str(i)]['f1-score'])
        else:
            f1_list.append(0.0)
    return f1_list

def get_confusion_matrix_recall(gt, pred, normalized=True):
    """
    recall mat
    """
    cm = confusion_matrix(gt, pred)
    cm_norm = cm / np.sum(cm, axis=1)[:,None]
    if normalized:
        return cm_norm
    else:
        return cm

def get_confusion_matrix_precision(gt, pred, normalized=True):
    """
    precision mat
    """
    cm = confusion_matrix(gt, pred)
    cm_norm = cm / np.sum(cm, axis=0)[None,:]
    if normalized:
        return cm_norm
    else:
        return cm

def get_confusion_matrix_image(gt, pred, mode='recall', normalized=True, title='Normalized Confusion Matrix'):
    if mode == 'recall':
        cm = get_confusion_matrix_recall(gt, pred, normalized)
    elif mode == 'precision':
        cm = get_confusion_matrix_precision(gt, pred, normalized)
    fig = plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalized:
                text = plt.text(j, i, '{:.2f}'.format(cm[i, j]), ha="center", va="center", color="k")
            else:
                text = plt.text(j, i, '{:d}'.format(cm[i, j]), ha="center", va="center", color="k")

    plt.savefig('img/overall_cm.pdf')

def load_gt_and_pred():

    with open('data/model_pred_prob_nodata.pkl', 'rb') as fin:
        res = pickle.load(fin)
    
    ### gt: (1166,), index of ground-truth label
    gt = res['test_target']

    ### mc_pred_prob: (50, 1166, 9)
    mc_pred_prob = res['pred_prob']

    ### model_pred_prob: (1166, 9)
    model_pred_prob = np.mean(mc_pred_prob, axis=0)

    return gt, model_pred_prob, mc_pred_prob

def uncertainty_metric(p):
    return entropy(p)

def compute_uncertainty(mc_pred_prob):
    """
    mc_pred_prob: (50, 1166, 9)
    total_uncertainty, model_uncertainty, expected_data_uncertainty: (1166,)
    """
    total_uncertainty = np.apply_along_axis(uncertainty_metric, -1, np.mean(mc_pred_prob, axis=0))
    expected_data_uncertainty = np.mean(np.apply_along_axis(uncertainty_metric, -1, mc_pred_prob), axis=0)
    model_uncertainty = total_uncertainty - expected_data_uncertainty
    return total_uncertainty, model_uncertainty, expected_data_uncertainty

def exp1(model_uncertainty, expected_data_uncertainty, final_gt, final_pred, f1_list):
    """
    compare model_uncertainty and expected_data_uncertainty by class
    """
    plt.figure()
    sns.distplot(model_uncertainty, bins=30, hist=True, kde=False, label='Model uncertainty')
    sns.distplot(expected_data_uncertainty, bins=30, hist=True, kde=False, label='Data uncertainty')
    plt.legend(fontsize=16)
    plt.xlabel('Uncertainty', fontsize=16)
    plt.ylabel('Number of Samples', fontsize=16)
    plt.tight_layout()
    plt.savefig('img/uncertainty_distribution.pdf')

    r, p = pearsonr(model_uncertainty, expected_data_uncertainty)
    plt.figure()
    plt.scatter(model_uncertainty, expected_data_uncertainty, alpha=0.3)
    plt.xlabel('Model uncertainty', fontsize=16)
    plt.ylabel('Data uncertainty', fontsize=16)
    plt.title('Correlation: {:.4f}, p-value: {:.4e}'.format(r, p))
    plt.tight_layout()
    plt.savefig('img/corr.pdf')
        
    df = []
    for i in np.unique(final_gt):
        selected = (final_gt == i)
        i_model_uncertainty = np.mean(model_uncertainty[selected])
        i_expected_data_uncertainty = np.mean(expected_data_uncertainty[selected])
        row = [i, label_map[i], np.sum(selected), i_model_uncertainty, i_expected_data_uncertainty, f1_list[i]]
        df.append(row)
    
    df = pd.DataFrame(df, columns=['class', 'label', 'number', 'model uncertainty', 'data uncertainty', 'f1-score'])
    print(df)
    df.to_csv('output/uncertainty.csv')

def exp2(total_uncertainty, model_uncertainty, expected_data_uncertainty, final_gt, final_pred, f1_list):
    """
    how the overall f1 score changes with respect to model/data uncertainty
    """
    number_of_bins = 10

    def get_mean_score(final_gt, final_pred, idx):
        return np.mean(np.array(final_gt)==np.array(final_pred)), np.mean(get_report(np.array(final_gt)[idx], np.array(final_pred)[idx]))

    total_uncertainty_idx = np.argsort(total_uncertainty)
    model_uncertainty_idx = np.argsort(model_uncertainty)
    data_uncertainty_idx = np.argsort(expected_data_uncertainty)

    total_uncertainty_bin_acc = []
    model_uncertainty_bin_acc = []
    data_uncertainty_bin_acc = []

    total_uncertainty_bin_f1 = []
    model_uncertainty_bin_f1 = []
    data_uncertainty_bin_f1 = []

    bin_size = len(total_uncertainty) // number_of_bins
    for i in range(number_of_bins):
        idx = total_uncertainty_idx[i*bin_size:(i+1)*bin_size]
        total_uncertainty_bin_acc.append(get_mean_score(final_gt, final_pred, idx)[0])
        total_uncertainty_bin_f1.append(get_mean_score(final_gt, final_pred, idx)[1])
        
        idx = model_uncertainty_idx[i*bin_size:(i+1)*bin_size]
        model_uncertainty_bin_acc.append(get_mean_score(final_gt, final_pred, idx)[0])
        model_uncertainty_bin_f1.append(get_mean_score(final_gt, final_pred, idx)[1])
        
        idx = data_uncertainty_idx[i*bin_size:(i+1)*bin_size]
        data_uncertainty_bin_acc.append(get_mean_score(final_gt, final_pred, idx)[0])
        data_uncertainty_bin_f1.append(get_mean_score(final_gt, final_pred, idx)[1])
    
    x = np.arange(number_of_bins) / number_of_bins

    plt.figure()
    plt.plot(x, total_uncertainty_bin_acc, label='Total uncertainty', color='r')
    plt.plot(x, model_uncertainty_bin_acc, label='Model uncertainty', color='g')
    plt.plot(x, data_uncertainty_bin_acc, label='Data uncertainty', color='b')
    plt.xlabel('Uncertainty percentile', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig('img/uncertainty_acc.pdf')

    plt.figure()
    plt.plot(x, total_uncertainty_bin_f1, label='Total uncertainty', color='r')
    plt.plot(x, model_uncertainty_bin_f1, label='Model uncertainty', color='g')
    plt.plot(x, data_uncertainty_bin_f1, label='Data uncertainty', color='b')
    plt.xlabel('Uncertainty percentile', fontsize=16)
    plt.ylabel('F1-score', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig('img/uncertainty_f1.pdf')

def main():
    final_gt, model_pred_prob, mc_pred_prob = load_gt_and_pred()

    ### predicted label: (1166,)
    final_pred = np.argmax(model_pred_prob, axis=-1)

    ### get f1-score for each class
    f1_list = get_report(final_gt, final_pred)
    
    ### get confusion matrix image
    get_confusion_matrix_image(final_gt, final_pred)

    ### compute uncertainty
    total_uncertainty, model_uncertainty, expected_data_uncertainty = compute_uncertainty(mc_pred_prob)

    ### how the overall f1 score changes with respect to model/data uncertainty
    exp2(total_uncertainty, model_uncertainty, expected_data_uncertainty, final_gt, final_pred, f1_list)
    
    ### compare model_uncertainty and expected_data_uncertainty by class
    exp1(model_uncertainty, expected_data_uncertainty, final_gt, final_pred, f1_list)
    
if __name__ == "__main__":
    main()


    ### load data
    final_gt, model_pred_prob, mc_pred_prob = load_gt_and_pred()
    final_pred = np.argmax(np.mean(mc_pred_prob, axis=0), axis=1)
    
    ### overall perf
    f1_list = get_report(final_gt, final_pred)
    title = 'Overall Performance, F1: {:.4f}'.format(np.mean(f1_list))
    get_confusion_matrix_image(final_gt, final_pred, normalized=True, title=title)

    ### compute uncertainty
    total_uncertainty, model_uncertainty, expected_data_uncertainty = compute_uncertainty(mc_pred_prob)

    ### exp1
    exp1(model_uncertainty, expected_data_uncertainty, final_gt, final_pred, f1_list)
    
    ### exp2
    final_uncertainty = expected_data_uncertainty
    all_thresh = np.arange(0.1,0.5,0.02)
    exp2(final_uncertainty, all_thresh)

    ### exp3
    thresh = 0.2
    final_uncertainty = expected_data_uncertainty
    exp3(final_uncertainty, thresh)

    ### exp4
    final_uncertainty = expected_data_uncertainty
    topk = 30
    exp4(final_uncertainty, topk, model_pred_prob, final_gt, final_pred)
    
    
    
