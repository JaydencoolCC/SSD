import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional

from attack import PredictionScoreAttack
from datasets_utils.dataset_tools import collate_fn
from utils import adjust_learning_rate
class SalemAttack(PredictionScoreAttack):
    def __init__(
        self,
        apply_softmax: bool,
        k: int = 3,
        attack_model: nn.Module = None,
        batch_size: int = 32,
        epochs: Optional[int] = None,
        lr: float = 0.01,
        log_training: bool = False,
    ):
        super().__init__('Salem Attack')
        self.k = k
        self.batch_size = batch_size
        if attack_model:
            self.attack_model = attack_model
        else:
            self.attack_model = nn.Sequential(nn.Linear(self.k, 128), nn.ReLU(), nn.Linear(128,64), nn.ReLU(), nn.Linear(64, 1))
        self.attack_model.to(self.device)
        self.apply_softmax = apply_softmax
        self.epochs = epochs
        self.lr = lr
        self.log_training = log_training

    def learn_attack_parameters(self, shadow_model: nn.Module, member_dataset: Dataset, non_member_dataset: Dataset):
        # Gather predictions by shadow model
        shadow_model.to(self.device)
        shadow_model.eval()
        features = []
        membership_labels = []
        if self.log_training:
            print('Compute attack model dataset')
        with torch.no_grad():
            shadow_model.eval()
            for i, dataset in enumerate([non_member_dataset, member_dataset]):
                loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4, collate_fn=collate_fn, shuffle=False, pin_memory=True)
                for images, boxes, labels, difficulties in loader:
                    images = images.to(self.device)
                    predicted_locs, predicted_scores = shadow_model(images) #(N, 8732, 4), (N, 8732, n_classes)
                    output = torch.cat([predicted_locs,predicted_scores], dim=-1) #(N, 8732, n_classes+4)
                    output = output.view(output.size(0), -1) #(N, 8732*(n_classes+4))
                    if self.apply_softmax:
                        prediction_scores = torch.softmax(output, dim=1)
                        features.append(prediction_scores)
                    else:
                        features.append(output)
                    membership_labels.append(torch.full((len(labels),), i))
                    

        # Compute top-k predictions
        membership_labels = torch.cat(membership_labels, dim=0) #[n]
        features = torch.cat(features, dim=0)
        attack_dataset = torch.utils.data.dataset.TensorDataset(features, membership_labels)

        # Train attack model
        self.attack_model.train()
        loss_fkt = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.attack_model.parameters(), lr=0.001, weight_decay=1e-4)
        
        if self.log_training:
            print('Train attack model')
        
        early_stopper = EarlyStopper(window=15, min_diff=0.05)
        epoch = 0
        parm = [50, 100]
        #self.epochs = 20
        while epoch != self.epochs:
            # if epoch in parm:
            #     adjust_learning_rate(optimizer,scale=0.1)
                
            num_corrects = 0
            total_samples = 0
            running_loss = 0.0
            for x, y in DataLoader(attack_dataset, batch_size=self.batch_size, shuffle=True):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.attack_model(x).squeeze()
                loss = loss_fkt(output, y.float())
                loss.backward()
                optimizer.step()

                preds = (output.sigmoid() >= 0.5).long().squeeze()
                num_corrects += torch.sum(preds == y.squeeze())
                total_samples += len(preds)
                running_loss += loss.item() * x.size(0)
            acc = num_corrects / total_samples
            train_loss = running_loss / len(attack_dataset)

            if self.log_training:
                print(f'Epoch {epoch}: Acc={acc:.4f} loss={train_loss}')
                
            if early_stopper.stop_early(train_loss):
                break

            epoch += 1

    def predict_membership(self, target_model: nn.Module, dataset: Dataset):
        predictions = []
        self.attack_model.eval()
        target_model.eval()
        with torch.no_grad():
            for images, boxes, labels, difficulties in DataLoader(dataset, batch_size=self.batch_size, num_workers=4, collate_fn=collate_fn):
                images = images.to(self.device)
                predicted_locs, predicted_scores = target_model(images) #(N, 8732, 4), (N, 8732, n_classes)
                target_output = torch.cat([predicted_locs,predicted_scores], dim=-1) #(N, 8732, n_classes+4)
                target_output = target_output.view(target_output.size(0), -1) #(N, 8732*(n_classes+4))
                if self.apply_softmax:
                    pred_scores = torch.softmax(target_output, dim=1)
                    top_pred_scores = torch.topk(pred_scores, k=self.k, dim=1, largest=True, sorted=True).values
                    attack_output = self.attack_model(top_pred_scores)
                else:
                    features = target_output
                    attack_output = self.attack_model(features)
                predictions.append(attack_output.sigmoid() >= 0.5)
        predictions = torch.cat(predictions, dim=0).squeeze()
        return predictions.cpu().numpy()

    def get_attack_model_prediction_scores(self, target_model: nn.Module, dataset: Dataset) -> torch.Tensor:
        predictions = []
        self.attack_model.eval()
        target_model.eval()
        with torch.no_grad():
           for images, boxes, labels, difficulties in DataLoader(dataset, batch_size=self.batch_size, num_workers=4, collate_fn=collate_fn):
                images = images.to(self.device)
                predicted_locs, predicted_scores = target_model(images)
                target_output = torch.cat([predicted_locs,predicted_scores], dim=-1) #(N, 8732, n_classes+4)
                target_output = target_output.view(target_output.size(0), -1) #(N, 8732*(n_classes+4))
                if self.apply_softmax:
                    pred_scores = torch.softmax(target_output, dim=1)
                    top_pred_scores = torch.topk(pred_scores, k=self.k, dim=1, largest=True, sorted=True).values
                    attack_output = self.attack_model(top_pred_scores)
                else:
                    features = target_output
                    attack_output = self.attack_model(features)
                predictions.append(attack_output.sigmoid())
        return torch.cat(predictions, dim=0).squeeze().cpu()



class EarlyStopper:
    def __init__(self, window, min_diff=0.005):
        self.window = window
        self.best_value = np.inf
        self.current_count = 0
        self.min_diff = min_diff

    def stop_early(self, value):
        if self.best_value <= (value + self.min_diff) and self.current_count >= self.window:
            self.current_count = 0
            return True

        if value < self.best_value and (self.best_value - value) >= self.min_diff:
            self.current_count = 0
            self.best_value = value

        self.current_count += 1

        return False