from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from abc import abstractmethod
import numpy as np
import os
import sys
from torchmetrics.functional import roc, auroc, precision_recall_curve
from torchmetrics.utilities.compute import auc
from .attack_utils import AttackResult
from utils_tools.model_utils import *

class PredictionScoreAttack:
    """
    The base class for all score-based membership inference attacks.
    """
    def __init__(self, display_name: str = '', *kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.display_name = display_name

    @abstractmethod
    def learn_attack_parameters(
        self, shadow_model: nn.Module, member_dataset: torch.utils.data.Dataset, non_member_dataset: Dataset
    ):
        raise NotImplementedError('This function has to be implemented in a subclass')

    @abstractmethod
    def predict_membership(self, target_model: nn.Module, dataset: Dataset) -> np.ndarray:
        """
        Predicts whether the samples in the given dataset are a member of the given model.
        :param target_model: The given target model to predict the membership.
        :param dataset: The dataset that is going to be used to predict the membership.
        :returns: A numpy array containing bool values for each sample indicating whether the samples was is a member or not.
        """
        raise NotImplementedError('This function has to be implemented in a subclass')

    def evaluate(self, target_model: nn.Module, member_dataset: Dataset, non_member_dataset: Dataset) -> AttackResult:
        """
        Evaluates the attack by predicting the membership for the member dataset as well as the non-member dataset.
        Returns a `AttackResult`-object.
        :param target_model: The given target model
        :param member_dataset: The member dataset that was used to train the target model.
        :param non_member_dataset: The non-member dataset that was **not** used to train the target model.
        :param kwargs: Additional optional parameters for the `predict_membership`-method.
        """
        member_predictions = self.predict_membership(target_model, member_dataset) #0 or 1
        non_member_predictions = self.predict_membership(target_model, non_member_dataset)
        tp = member_predictions.sum()
        tn = len(non_member_dataset) - non_member_predictions.sum()
        fp = non_member_predictions.sum()
        fn = len(member_dataset) - member_predictions.sum()
        tpr = tp / len(member_dataset)
        tnr = tn / len(non_member_dataset)
        fpr = 1 - tnr
        fnr = 1 - tpr
        pre = tp / (tp + fp) if tp != 0 else 0.0
        rec = tp / (tp + fn)
        acc = (tp + tn) / (tp + tn + fp + fn)

        #用作度量的分数，如熵，最大值分数，loss等
        member_pred_scores = self.get_attack_model_prediction_scores(target_model, dataset=member_dataset)
        non_member_pred_scores = self.get_attack_model_prediction_scores(target_model, dataset=non_member_dataset)
        concat_preds = torch.cat((non_member_pred_scores, member_pred_scores))
        concat_targets = torch.tensor([0 for _ in non_member_pred_scores] + [1 for _ in member_pred_scores])

        # get the auroc, aupr and FPR@95%TPR
        auroc_value: float = auroc(
            preds=concat_preds,
            target=concat_targets,
            task="binary"
        ).item()
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
            preds=concat_preds,
            target=concat_targets,
            task="binary"
        )
        aupr_value: float = auc(x=pr_recall, y=pr_precision, reorder=True).item()
        tm_fpr, tm_tpr, tm_thresholds = roc(preds=concat_preds, target=concat_targets,task="binary")
        tpr_greater95_indices = np.where(tm_tpr >= 0.95)[0]
        fpr_at_tpr95 = tm_fpr[tpr_greater95_indices[0]].item()

        # get the mmps values
        # softmax 层的输出，[n, classes_num]
        # tp_pred_scores = self.get_pred_score_classified_as_members(
        #     target_model, member_dataset, apply_softmax=self.apply_softmax
        # )
        # tp_mmps = tp_pred_scores.max(dim=1)[0].mean() if len(tp_pred_scores) > 0 else 0

        # fp_pred_scores = self.get_pred_score_classified_as_members(
        #     target_model, non_member_dataset, apply_softmax=self.apply_softmax
        # )
        # fp_mmps = fp_pred_scores.max(dim=1)[0].mean() if len(fp_pred_scores) > 0 else 0

        # fn_pred_scores = self.get_pred_score_classified_as_non_members(
        #     target_model, member_dataset, apply_softmax=self.apply_softmax
        # )
        # fn_mmps = fn_pred_scores.max(dim=1)[0].mean() if len(fn_pred_scores) > 0 else 0

        # tn_pred_scores = self.get_pred_score_classified_as_non_members(
        #     target_model, non_member_dataset, apply_softmax=self.apply_softmax
        # )
        # tn_mmps = tn_pred_scores.max(dim=1)[0].mean() if len(tn_pred_scores) > 0 else 0

        tp_mmps=0
        fp_mmps=0
        fn_mmps=0
        tn_mmps=0
        
        result = AttackResult(
            attack_acc=acc,
            precision=pre,
            recall=rec,
            auroc=auroc_value,
            aupr=aupr_value,
            fpr_at_tpr95=fpr_at_tpr95,
            tpr=tpr,
            tnr=tnr,
            fpr=fpr,
            fnr=fnr,
            tp_mmps=tp_mmps,
            fp_mmps=fp_mmps,
            fn_mmps=fn_mmps,
            tn_mmps=tn_mmps
        )
        return result

    def get_attack_model_prediction_scores(self, target_model: nn.Module, dataset: Dataset) -> torch.Tensor:
        raise NotImplementedError('This function has to be implemented in a subclass')

    def get_pred_score_classified_as_members(self, target_model: nn.Module, dataset: Dataset, apply_softmax: bool):
        membership_predictions = self.predict_membership(target_model, dataset)

        predicted_member_indices = torch.nonzero(torch.tensor(membership_predictions).squeeze()).squeeze()
        if predicted_member_indices.ndim == 0:
            predicted_member_indices = predicted_member_indices.unsqueeze(0)

        samples, labels = [], []
        for idx in predicted_member_indices:
            sample, label = dataset[idx]
            samples.append(sample)
            labels.append(label)
        samples = [torch.tensor(sample) for sample in samples]
        samples = torch.stack(samples) if len(samples) > 0 else torch.empty(0)
        labels = torch.tensor(labels) if len(labels) > 0 else torch.empty(0)

        return get_model_prediction_scores(target_model, apply_softmax, torch.utils.data.TensorDataset(samples, labels))

    def get_pred_score_classified_as_non_members(self, target_model: nn.Module, dataset: Dataset, apply_softmax: bool):
        membership_predictions = self.predict_membership(target_model, dataset)

        predicted_member_indices = torch.nonzero(torch.logical_not(torch.tensor(membership_predictions)).squeeze()
                                                 ).squeeze()
        if predicted_member_indices.ndim == 0:
            predicted_member_indices = predicted_member_indices.unsqueeze(0)

        samples, labels = [], []
        for idx in predicted_member_indices:
           images, boxes, labels, difficulties  = dataset[idx]
           samples.append(images)
           labels.append(labels)
            
        samples = [torch.tensor(sample) for sample in samples]
        samples = torch.stack(samples) if len(samples) > 0 else torch.empty(0)
        labels = torch.tensor(labels) if len(labels) > 0 else torch.empty(0)

        return get_model_prediction_scores(target_model, apply_softmax, torch.utils.data.TensorDataset(samples, labels))
    
        """@return: pre_scores, labels: torch.Tensor, torch.Tensor
        """
    def get_pre_score_dataset(self, target_model: nn.Module, dataset: Dataset, apply_softmax: bool):
        pre_scores, labels = get_model_prediction_scores_with_lables(target_model, apply_softmax, dataset)
        pre_scores = pre_scores.max(dim=1)[0].tolist()
        labels = labels.tolist()
        return pre_scores, labels
    
    
    
        
        
        