import os

import torch
from torch import nn


class EarlyStopping(object):
    def __init__(self, patience: int, save_model_folder: str, model_name: str = None):
        self.patience = patience
        self.counter = 0
        self.best_metrics = {}
        self.early_stop = False
        self.save_model_folder = save_model_folder
        self.model_name = model_name

    def step(self, metrics: list, model: nn.Module):
        metrics_to_save = {}
        for metric_tuple in metrics:
            metric_name, metric_value, higher_better = metric_tuple
            if higher_better:
                if self.best_metrics.get(metric_name) is None or metric_value > self.best_metrics[metric_name]:
                    if metric_name not in metrics_to_save:
                        metrics_to_save[metric_name] = (metric_value, model)
            else:
                if self.best_metrics.get(metric_name) is None or metric_value < self.best_metrics[metric_name]:
                    if metric_name not in metrics_to_save:
                        metrics_to_save[metric_name] = (metric_value, model)

        if metrics_to_save:
            for metric_name, (metric_value, model) in metrics_to_save.items():
                self.best_metrics[metric_name] = metric_value
                self.save_checkpoint(model, metric_name)

            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model: nn.Module, metric_name: str):
        save_path = os.path.join(self.save_model_folder, f"{self.model_name}_{metric_name}.pkl")
        torch.save(model.state_dict(), save_path)

    def load_checkpoint(self, model: nn.Module, metric_name: str, map_location: str = None):
        save_path = os.path.join(self.save_model_folder, f"{self.model_name}_{metric_name}.pkl")
        model.load_state_dict(torch.load(save_path, map_location=map_location))
