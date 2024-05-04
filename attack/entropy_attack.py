import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.distributions import Categorical
import numpy as np
from datasets_utils.dataset_tools import collate_fn
from attack import PredictionScoreAttack


class EntropyAttack(PredictionScoreAttack):
    def __init__(self, apply_softmax: bool, batch_size: int = 128, log_training: bool = False):
        super().__init__('Entropy Attack')

        self.batch_size = batch_size
        self.theta = 0.0
        self.apply_softmax = apply_softmax
        self.log_training = log_training

    def learn_attack_parameters(
        self, shadow_model: nn.Module, member_dataset: torch.utils.data.Dataset, non_member_dataset: Dataset
    ):
        # Gather entropy of predictions by shadow model
        
        
        shadow_model.to(self.device)
        shadow_model.eval()
        entropy_values = []
        membership_labels = []
        if self.log_training:
            print('Compute attack model dataset')
        with torch.no_grad():
            shadow_model.eval()
            for i, dataset in enumerate([non_member_dataset, member_dataset]):
                loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4, collate_fn=collate_fn, pin_memory=True, shuffle=False)
                for  images, boxes, labels, difficulties in loader:
                    images = images.to(self.device)
                    boxes = [b.to(self.device) for b in boxes]
                    labels = [l.to(self.device) for l in labels]
                    predicted_locs, predicted_scores = shadow_model(images)
                    if self.apply_softmax:
                        prediction_scores = torch.softmax(predicted_scores, dim=1)
                    else:
                        prediction_scores = predicted_scores
                    entropy = Categorical(logits=prediction_scores).entropy()
                    
                    entropy_values.append(torch.mean(entropy))
                    membership_labels.append(torch.full((len(labels),), i))
        entropy_values = torch.stack(entropy_values, dim=0).cpu().numpy()
        membership_labels = torch.cat(membership_labels, dim=0).cpu().numpy()

        # Compute threshold
        theta_best = 0.0
        num_corrects_best = 0
        for theta in np.linspace(min(entropy_values), max(entropy_values), 10000):
            num_corrects = (entropy_values[membership_labels == 0] >=
                            theta).sum() + (entropy_values[membership_labels == 1] < theta).sum()
            if num_corrects > num_corrects_best:
                num_corrects_best = num_corrects
                theta_best = theta
        self.theta = theta_best
        if self.log_training:
            print(
                f'Theta set to {self.theta} achieving {num_corrects_best / (len(member_dataset) + len(non_member_dataset))}'
            )

    def predict_membership(self, target_model: nn.Module, dataset: Dataset):
        entropy_values = []
        target_model.eval()
        with torch.no_grad():
            loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4, collate_fn=collate_fn, pin_memory=True)
            for  images, boxes, labels, difficulties in loader:
                images = images.to(self.device)
                boxes = [b.to(self.device) for b in boxes]
                labels = [l.to(self.device) for l in labels]
                predicted_locs, predicted_scores = target_model(images)
                if self.apply_softmax:
                    pred_scores = torch.softmax(predicted_scores, dim=1)
                else:
                    pred_scores = predicted_scores
                entropy = Categorical(logits=predicted_scores).entropy()
                entropy_values.append(torch.mean(entropy))
        entropy_values = torch.stack(entropy_values, dim=0)
        return (entropy_values < self.theta).cpu().numpy()

    def get_attack_model_prediction_scores(self, target_model: nn.Module, dataset: Dataset) -> torch.Tensor:
        entropy_values = []
        target_model.eval()
        with torch.no_grad():
            loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4, collate_fn=collate_fn, pin_memory=True)
            for  images, boxes, labels, difficulties in loader:
                images = images.to(self.device)
                boxes = [b.to(self.device) for b in boxes]
                labels = [l.to(self.device) for l in labels]
                predicted_locs, predicted_scores = target_model(images)
                if self.apply_softmax:
                    pred_scores = torch.softmax(predicted_scores, dim=1)
                else:
                    pred_scores = predicted_scores
                entropy = Categorical(logits=predicted_scores).entropy()
                entropy_values.append(torch.mean(entropy))
        entropy_values = torch.stack(entropy_values, dim=0)

        # we have to flip the sign of the values since low entropy means that the sample was a member
        return -entropy_values.cpu()
