import random
import shutil

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from ..constant import *


def get_all_edges_nodes(train_edges, val_edges, test_edges, num_nodes):
    train_edge_nodes = get_edges_nodes(train_edges, num_nodes)
    val_edge_nodes = get_edges_nodes(val_edges, num_nodes)
    test_edge_nodes = get_edges_nodes(test_edges, num_nodes)

    return train_edge_nodes, val_edge_nodes, test_edge_nodes


def get_edges_nodes(edges, num_nodes):
    v = torch.tensor([num_nodes, 1], dtype=torch.float, device=edges.device)
    edge_src_nodes = torch.matmul(edges[[0, 1]].transpose(1, 0).to(torch.float), v).to(torch.long)
    edge_trg_nodes = torch.matmul(edges[[0, 2]].transpose(1, 0).to(torch.float), v).to(torch.long)
    return edge_src_nodes, edge_trg_nodes


def func_createM(T, bandwidth, m_choice, device):
    M = np.zeros((T, T))
    for i in range(bandwidth):
        A = M[i:, :T - i]
        diag_val = 1 if m_choice == 1 else 1 / (i + 1)
        np.fill_diagonal(A, diag_val)

    L = np.sum(M, axis=1)
    M = M / L[:, None]
    return torch.tensor(M, dtype=torch.float, device=device)


def compute_metrics(output, target, edges, is_weight=True):
    if is_weight:
        ap = 0.0
        roc_auc = 0.0
        T = len(edges[0].unique())
        for k in edges[0].unique():
            edges_mask = edges[0] == k
            predictions = output[edges_mask]
            sub_target = target[edges_mask]
            ap_slice, roc_slice = get_link_prediction_metrics(predictions, sub_target)
            ap += ap_slice
            roc_auc += roc_slice
        return {'AP': ap / T, 'ROC_AUC': roc_auc / T}
    else:
        ap, roc_auc = get_link_prediction_metrics(output, target)
        return {'AP': ap, 'ROC_AUC': roc_auc}


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return average_precision, roc_auc


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def split_data(labels, TS):
    val_steps = int(TS * VAL_RATE)
    test_steps = int(TS * TEST_RATE)
    train_steps = TS - val_steps - test_steps

    edges = labels._indices()
    indices = labels._values()

    # Training
    subs_train = edges[0] < train_steps
    edges_train = edges[:, subs_train]
    target_train = indices[subs_train]
    target_train = target_train[edges_train[0] != 0]
    edges_train = edges_train[:, edges_train[0] != 0]
    edges_train[0, :] = edges_train[0, :] - 1

    # Validation
    subs_val = (edges[0] >= val_steps) & (edges[0] < train_steps + val_steps)
    edges_val = edges[:, subs_val]
    edges_val[0] -= val_steps
    target_val = indices[subs_val]

    K_val = torch.sum(edges_val[0] - (train_steps - val_steps - 1) > 0)
    edges_val = edges_val[:, edges_val[0] != 0]
    edges_val[0, :] = edges_val[0, :] - 1

    # Testing
    subs_test = (edges[0] >= test_steps + val_steps)
    edges_test = edges[:, subs_test]
    edges_test[0] -= (test_steps + val_steps)
    target_test = indices[subs_test]

    K_test = torch.sum(edges_test[0] - (train_steps - test_steps - 1) > 0)
    edges_test = edges_test[:, edges_test[0] != 0]
    edges_test[0, :] = edges_test[0, :] - 1

    return edges_train, target_train, edges_val, target_val, K_val, edges_test, target_test, K_test


def get_results_sava_path(lr, lam, num_feature, m_choice, fft=True, enable_cl=True, tensor_con=True):
    save_path = f'./results/lr_{lr}_lam_{lam}_num_features_{num_feature}_M_{m_choice}'
    if fft:
        save_path += '_fft'
    if enable_cl:
        save_path += '_cl'
    if tensor_con:
        save_path += '_tensor_con'
    return save_path


def normalize(tensor):
    sums = tensor.sum(dim=2, keepdim=True) + 1e-10
    normalized_tensor = tensor / sums
    return normalized_tensor


def log_metric(state, epoch, metrics, loss):
    log = f"Ep {epoch}. {state} "
    for metric_name, test_metric in metrics.items():
        log += f"{metric_name}: {test_metric:.16f}  "
    log += f"Loss: {loss:.16f}"
    log += "\n"
    return log


def get_save_parameter(lr, lam, num_feature, run, tau, args):
    save_path = get_results_sava_path(lr, lam, num_feature, args.m_choice, args.fft, args.enable_cl, args.tensor_con)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    layers = len(args.hidden_features)
    save_res_fname = f'{save_path}/{args.model_name}_layers_{layers}_{args.dataset_name}_tau_{tau}_run_{run}'

    args.save_model_name = f'{args.model_name}_seed_{args.seed + run}'
    save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
    shutil.rmtree(save_model_folder, ignore_errors=True)
    os.makedirs(save_model_folder, exist_ok=True)

    return save_res_fname, save_model_folder
