import torch.optim

from loss import Loss
from models.model import SGDDyG
from utils import util, DataLoader
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args


def train(args, num_feature, lr, lam, tau):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    TS, labels, A_train, A_val, A_test, N = DataLoader.load_data(args, device)
    T = len(A_train) - 1
    M = util.func_createM(T, args.bandwidth, args.m_choice, device)

    edges_train, target_train, edges_val, target_val, K_val, edges_test, target_test, K_test = util.split_data(labels, TS)
    train_edge_nodes, val_edge_nodes, test_edge_nodes = util.get_all_edges_nodes(edges_train, edges_val, edges_test, N)

    enable_cl = args.enable_cl

    for run in range(args.num_runs):
        util.set_seed(args.seed)

        save_res_fname, save_model_folder = util.get_save_parameter(lr, lam, num_feature, run, tau, args)
        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder, model_name=args.model_name)

        model = SGDDyG(T, N,
                       hidden_features=args.hidden_features,
                       num_feature=num_feature,
                       out_features=1,
                       bandwidth=args.bandwidth,
                       tgc_dropout=args.tgc_dropout,
                       fft_dropout=args.fft_dropout,
                       fft=args.fft,
                       tensor_con=args.tensor_con)
        model = model.to(device=device)

        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        criterion = Loss(model.X, lam, enable_cl, tau)

        logs = []
        for ep in range(1, args.epochs + 1):
            optimizer.zero_grad()

            model.train()
            if enable_cl:
                output_train_1, h1 = model(A_train[:-1], train_edge_nodes, M, False)
                output_train_2, h2 = model(A_train[:-1], train_edge_nodes, M, True)
                output_train = output_train_1
                loss_train = criterion(output_train, target_train, h1, h2)
            else:
                output_train, _ = model(A_train[:-1], train_edge_nodes, M, False)
                loss_train = criterion(output_train, target_train)
            train_metrics = util.compute_metrics(output_train, target_train, edges_train)

            loss_train.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()

                if enable_cl:
                    output_val_1, h1 = model(A_val[:-1], val_edge_nodes, M, False)
                    output_val_2, h2 = model(A_val[:-1], val_edge_nodes, M, True)
                    output_val = output_val_1
                    loss_val = criterion(output_val[-K_val:], target_val[-K_val:], h1, h2)
                else:
                    output_val, _ = model(A_val[:-1], val_edge_nodes, M, False)
                    loss_val = criterion(output_val[-K_val:], target_val[-K_val:])

                val_metrics = util.compute_metrics(output_val[-K_val:], target_val[-K_val:], edges_val[:, -K_val:])

            log = util.log_metric('Train', ep, train_metrics, loss_train)
            log += util.log_metric('Val', ep, val_metrics, loss_val)
            print(log)
            logs.append(log)

            val_metric_indicator = []
            for metric_name in val_metrics.keys():
                val_metric_indicator.append((metric_name, val_metrics[metric_name], True))
            early_stop = early_stopping.step(val_metric_indicator, model)
            if early_stop:
                break

        metric_names = ['AP', 'ROC_AUC']
        for metric_name in metric_names:
            early_stopping.load_checkpoint(model, metric_name)
            model.eval()
            output_test, _ = model(A_test[:-1], test_edge_nodes, M, False)

            test_metric = util.compute_metrics(output_test[-K_test:], target_test[-K_test:], edges_test[:, -K_test:])
            log = (f"Test {metric_name}: {test_metric.get(metric_name)} in Val "
                   f"{metric_name}: {early_stopping.best_metrics.get(metric_name)}")
            print(log)
            logs.append(log)

        with open(save_res_fname, 'w') as f:
            f.write("\n".join(logs))
        print("Results saved for single trial")


def main():
    args = get_link_prediction_args()
    train(args, args.num_feature, args.lr, args.lam, args.tau)


if __name__ == '__main__':
    main()
