import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score, balanced_accuracy_score
import sklearn.metrics as sk

import torch.nn.functional as F

recall_level_default = 0.95

def validation_balnced_accuracy(model, loader, method, device):
    pred_list = []
    target_list = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if method == 'linear':
                outputs = model(inputs)
                outputs = model.linear(outputs)
            elif method == 'rein':
                features = model.forward_features(inputs)
                features = features[:, 0, :]
                outputs = model.linear(features)      
            else:
                features = model.forward_features(inputs)
                features = features[:, 0, :]
                outputs = model.linear_rein(features)           
            _, predicted = outputs.max(1)  
            target_list.append(targets.cpu())
            pred_list.append(predicted.cpu())
    target_list = torch.cat(target_list, dim=0)
    pred_list = torch.cat(pred_list, dim=0)  

    accuracy = balanced_accuracy_score(target_list, pred_list)
    return accuracy


def validation_kohen_kappa_ours(model, loader, device, adapter = 'rein'):
    targets_list = []
    preds_list = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            features = model.forward_features(inputs)
            if adapter == 'rein':
                features = features[:, 0, :]
            outputs = model.linear_rein(features) # should be changed to linear_rein for reinfn
            pred = outputs.max(1).indices

            targets_list.append(targets.cpu().view(-1))
            preds_list.append(pred.cpu().view(-1))
            # print(batch_idx, pred.cpu().view(-1))
    targets_list = torch.cat(targets_list)
    preds_list = torch.cat(preds_list)
    print(targets_list.shape, preds_list.shape)

    kappa = cohen_kappa_score(targets_list.numpy(), preds_list.numpy(), weights = 'quadratic')
    return kappa

def validation_kohen_kappa(model, loader, device):
    targets_list = []
    preds_list = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            features = model.forward_features(inputs)
            features = features[:, 0, :]
            outputs = model.linear(features) # should be changed to linear_rein for reinfn
            pred = outputs.max(1).indices

            targets_list.append(targets.cpu().view(-1))
            preds_list.append(pred.cpu().view(-1))
            # print(batch_idx, pred.cpu().view(-1))
    targets_list = torch.cat(targets_list)
    preds_list = torch.cat(preds_list)
    print(targets_list.shape, preds_list.shape)

    kappa = cohen_kappa_score(targets_list.numpy(), preds_list.numpy(), weights = 'quadratic')
    return kappa


def validation_kohen_kappa_linear(model, loader, device):
    targets_list = []
    preds_list = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            features = model(inputs)
            outputs = model.linear(features) # should be changed to linear_rein for reinfn
            pred = outputs.max(1).indices

            targets_list.append(targets.cpu().view(-1))
            preds_list.append(pred.cpu().view(-1))
            # print(batch_idx, pred.cpu().view(-1))
    targets_list = torch.cat(targets_list)
    preds_list = torch.cat(preds_list)
    print(targets_list.shape, preds_list.shape)

    kappa = cohen_kappa_score(targets_list.numpy(), preds_list.numpy(), weights = 'quadratic')
    return kappa

def validation_accuracy(model, loader, device):
    total = 0
    correct = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = model.linear(outputs)
            #print(outputs.shape)
            total += targets.size(0)
            _, predicted = outputs.max(1)  
            correct += predicted.eq(targets).sum().item()
    valid_accuracy = correct/total
    return valid_accuracy

def validation_accuracy_lora(model, loader, device):
    total = 0
    correct = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model.forward_features(inputs)
            outputs = model.linear(outputs)
            #print(outputs.shape)
            total += targets.size(0)
            _, predicted = outputs.max(1)  
            correct += predicted.eq(targets).sum().item()
    valid_accuracy = correct/total
    return valid_accuracy

def get_f1_score(model, loader, device):
    y_pred = []
    y_true = []
    
    m = torch.nn.Sigmoid()
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = model.linear(outputs)
            outputs = m(outputs) > 0.5
            #print(outputs.shape)
            y_pred.extend(outputs.cpu())
            y_true.extend(targets.cpu())
    y_pred = torch.stack(y_pred)
    y_true = torch.stack(y_true)
    # print(y_pred)
    f1 = f1_score(y_true, y_pred, average='micro')
    return f1


def validation_accuracy_resnet(model, loader, device):
    total = 0
    correct = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            #print(outputs.shape)
            total += targets.size(0)
            _, predicted = outputs.max(1)  
            correct += predicted.eq(targets).sum().item()
    valid_accuracy = correct/total
    return valid_accuracy


def validation_accuracy_rein(model, loader, device, adapter='rein', no_rein = False):
    total = 0
    correct = 0
    
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            features = model.forward_features_no_rein(inputs) if no_rein else model.forward_features(inputs)
            if adapter == 'rein':
                features = features[:, 0, :]
            outputs = model.linear(features)

            total += targets.size(0)
            _, predicted = outputs.max(1)  
            correct += predicted.eq(targets).sum().item()
    valid_accuracy = correct/total
    return valid_accuracy

def get_f1_score_rein(model, loader, device):
    y_pred = []
    y_true = []
    
    m = torch.nn.Sigmoid()
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            features = model.forward_features(inputs)
            features = features[:, 0, :]
            outputs = model.linear(features)
            outputs = m(outputs) > 0.5
            #print(outputs.shape)
            y_pred.extend(outputs.cpu())
            y_true.extend(targets.cpu())
    y_pred = torch.stack(y_pred)
    y_true = torch.stack(y_true)
    # print(y_pred)
    f1 = f1_score(y_true, y_pred, average='micro')
    return f1

def validation_accuracy_shared(model, loader, device):
    total = 0
    correct = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            features = model.forward_features(inputs)
            features = features[:, 0, :]
            outputs = model.linear(features)

            features = model.forward_features_no_rein(inputs)
            features = features[:, 0, :]
            outputs_ = model.linear(features)
            outputs = outputs+outputs_
            
            total += targets.size(0)
            _, predicted = outputs.max(1)  
            correct += predicted.eq(targets).sum().item()
    valid_accuracy = correct/total
    return valid_accuracy



def validation_accuracy_ours(model, loader, device, adapter='rein', use_rein1 = False):
    total = 0
    correct = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            features = model.forward_features(inputs)
            if adapter == 'rein':
                features = features[:, 0, :]
            outputs = model.linear_rein(features)
            outputs = outputs #+ outputs_

            total += targets.size(0)
            _, predicted = outputs.max(1)  
            correct += predicted.eq(targets).sum().item()
    valid_accuracy = correct/total
    return valid_accuracy

def validation_accuracy_ours(model, loader, device, adapter='rein', use_rein1 = False):
    total = 0
    correct = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            features = model.forward_features(inputs)
            if adapter == 'rein':
                features = features[:, 0, :]
            outputs = model.linear_rein(features)
            outputs = outputs #+ outputs_

            total += targets.size(0)
            _, predicted = outputs.max(1)  
            correct += predicted.eq(targets).sum().item()
    valid_accuracy = correct/total
    return valid_accuracy


def validation_accuracy_ours_head3(model, loader, device, adapter='rein', use_rein1 = False):
    total = 0
    correct = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if use_rein1:
                features = model.forward_features1(inputs)
                if adapter == 'rein':
                    features = features[:, 0, :]
                outputs = model.linear_rein1(features)
            else:
                features = model.forward_features2(inputs)
                if adapter == 'rein':
                    features = features[:, 0, :]
                outputs = model.linear_rein2(features)

            total += targets.size(0)
            _, predicted = outputs.max(1)  
            correct += predicted.eq(targets).sum().item()
    valid_accuracy = correct/total
    return valid_accuracy

def validation_accuracy_rein_full(model, loader, device):
    total = 0
    correct = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            features = model.forward_features(inputs)
            features = features[:, 1:, :].mean(1)
            outputs = model.linear(features)
            outputs = outputs #+ outputs_

            total += targets.size(0)
            _, predicted = outputs.max(1)  
            correct += predicted.eq(targets).sum().item()
    valid_accuracy = correct/total
    return valid_accuracy


def validation_accuracy_linear(model, loader, device, adapter='rein'):
    total = 0
    correct = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            features = model.forward_features_no_rein(inputs)
            if adapter == 'rein':
                features = features[:, 0, :]
            outputs = model.linear(features)

            total += targets.size(0)
            _, predicted = outputs.max(1)  
            correct += predicted.eq(targets).sum().item()
    valid_accuracy = correct/total
    return valid_accuracy

def validation_accuracy_lora_linear(model, loader, device):
    total = 0
    correct = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model.forward_features_no_lora(inputs)
            outputs = model.linear(outputs)
            #print(outputs.shape)
            total += targets.size(0)
            _, predicted = outputs.max(1)  
            correct += predicted.eq(targets).sum().item()
    valid_accuracy = correct/total
    return valid_accuracy
def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=recall_level_default):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def show_performance(pos, neg, method_name='Ours', recall_level=recall_level_default, file=None):
    '''
    :param pos: 1's class, class to detect, outliers, or wrongly predicted
    example scores
    :param neg: 0's class scores
    '''

    auroc, aupr, fpr = get_measures(pos[:], neg[:], recall_level)

    print('\t\t\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))
    # print('FDR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fdr))
    if not file is None:
        file.write('{}\n'.format(method_name))
        file.write('FPR{:d}:\t\t\t{:.2f}\n'.format(int(100 * recall_level), 100 * fpr))
        file.write('AUROC:\t\t\t{:.2f}\n'.format(100 * auroc))
        file.write('AUPR:\t\t\t{:.2f}\n'.format(100 * aupr))
        file.write('\n')


def print_measures(auroc, aupr, fpr, method_name='Ours', recall_level=recall_level_default):
    print('\t\t\t\t' + method_name)
    print('  FPR{:d} AUROC AUPR'.format(int(100*recall_level)))
    print('& {:.2f} & {:.2f} & {:.2f}'.format(100*fpr, 100*auroc, 100*aupr))
    #print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    #print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
    #print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))


def print_measures_with_std(aurocs, auprs, fprs, method_name='Ours', recall_level=recall_level_default):
    print('\t\t\t\t' + method_name)
    print('  FPR{:d} AUROC AUPR'.format(int(100*recall_level)))
    print('& {:.2f} & {:.2f} & {:.2f}'.format(100*np.mean(fprs), 100*np.mean(aurocs), 100*np.mean(auprs)))
    print('& {:.2f} & {:.2f} & {:.2f}'.format(100*np.std(fprs), 100*np.std(aurocs), 100*np.std(auprs)))
    #print('FPR{:d}:\t\t\t{:.2f}\t+/- {:.2f}'.format(int(100 * recall_level), 100 * np.mean(fprs), 100 * np.std(fprs)))
    #print('AUROC: \t\t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(aurocs), 100 * np.std(aurocs)))
    #print('AUPR:  \t\t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(auprs), 100 * np.std(auprs)))
