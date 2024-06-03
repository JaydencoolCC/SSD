import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.distributions import Categorical
import numpy as np
from torchmetrics.functional import roc, auroc, precision_recall_curve
from torchmetrics.utilities.compute import auc
from .attack_utils import AttackResult
from attack import PredictionScoreAttack
from .attack_utils import cross_entropy
from models.ssd import SSD300, MultiBoxLoss
from utils_tools.roc import get_roc
from models.utils import array_tool as at
from datasets_utils.dataset_tools import collate_fn
from tqdm import tqdm
class MetricAttack(PredictionScoreAttack):
    def __init__(self, apply_softmax: bool, batch_size: int = 1, log_training: bool = True, metric_method = "Entropy", loss_type='ce', ts=False):
        super().__init__('loss Attack')
        self.metric = self.get_metric_method(metric_method)
        self.batch_size = batch_size
        self.theta = 0.0
        self.apply_softmax = apply_softmax
        self.log_training = log_training
        self.loss_type = loss_type
        self.ts = ts
        
    def learn_attack_parameters(
        self, shadow_model: nn.Module, member_dataset: torch.utils.data.Dataset, non_member_dataset: torch.utils.data.Dataset
    ):
        # Gather entropy of predictions by shadow model
        if(self.ts):
            criterion = MultiBoxLoss(priors_cxcy=shadow_model.model.priors_cxcy, loss_type=self.loss_type).to(self.device)
        else:
            criterion = MultiBoxLoss(priors_cxcy=shadow_model.priors_cxcy, loss_type=self.loss_type).to(self.device)
        
        shadow_model.to(self.device)
        shadow_model.eval()
        values = []
        membership_labels = []
        if self.log_training:
            print('Compute attack model dataset')
        with torch.no_grad():
            shadow_model.eval()
            for i, dataset in enumerate([non_member_dataset, member_dataset]):
                loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4, collate_fn=collate_fn, pin_memory=True, shuffle=False)
                for  images, boxes, labels, difficulties in tqdm(loader):
                    images = images.to(self.device)
                    boxes = [b.to(self.device) for b in boxes]
                    labels = [l.to(self.device) for l in labels]
                    predicted_locs, predicted_scores = shadow_model(images)
                    output = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar            
                    if self.apply_softmax:
                        prediction_scores = torch.softmax(output, dim=1)
                    else:
                        prediction_scores = output
                    values.append(output)
                    membership_labels.append(torch.full((len(labels),), i))
        loss_values = torch.stack(values, dim=0).cpu().numpy()
        membership_labels = torch.cat(membership_labels, dim=0).cpu().numpy()

        # Compute threshold
        theta_best = 0.0
        num_corrects_best = 0
        for theta in np.linspace(min(loss_values), max(loss_values), 100000):
            num_corrects = (loss_values[membership_labels == 0] >=
                            theta).sum() + (loss_values[membership_labels == 1] < theta).sum()
            if num_corrects > num_corrects_best:
                num_corrects_best = num_corrects
                theta_best = theta
        self.theta = theta_best
        

        # # Compute threshold 2
        # self.shadow_fpr, self.shadow_tpr, self.thresholds, self.auroc = get_roc(membership_labels, -loss_values)
        # threshold_idx = (self.shadow_tpr - self.shadow_fpr).argmax()
        # self.theta = - self.thresholds[threshold_idx]
        
        if self.log_training:
            print(
                f'Theta set to {self.theta} achieving {num_corrects_best / (len(member_dataset) + len(non_member_dataset))}'
            )

        
        
    def predict_membership(self, target_model: nn.Module, dataset: Dataset):
        values = []
        scores = []
        if(self.ts):
            criterion = MultiBoxLoss(priors_cxcy=target_model.model.priors_cxcy, loss_type=self.loss_type).to(self.device)
        else:
            criterion = MultiBoxLoss(priors_cxcy=target_model.priors_cxcy, loss_type=self.loss_type).to(self.device)
        target_model.eval()
        with torch.no_grad():
            loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4, collate_fn=collate_fn, pin_memory=True)
            for  images, boxes, labels, difficulties in loader:
                images = images.to(self.device)
                boxes = [b.to(self.device) for b in boxes]
                labels = [l.to(self.device) for l in labels]
                predicted_locs, predicted_scores = target_model(images)
                output = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar 
                if self.apply_softmax:
                    pred_scores = torch.softmax(output, dim=1)
                else:
                    pred_scores = output
                values.append(output)
                #scores.append(pred_scores.cpu()) #unit class_num
                #scores = values 
                scores.append(torch.tensor([output.cpu().item(),output.cpu().item()])) # TODO, no scores, so score is replaced by loss
        values = torch.stack(values, dim=0) #torch
        scores = torch.cat(scores) if len(scores) > 0 else torch.empty(0)

        print("Threshold: ", self.theta)
        nums_member = sum(values < self.theta)
        nums_non_member = sum(values > self.theta)
        print("Number of members: ", nums_member)
        print("Number of non members: ", nums_non_member)
        
        return (values < self.theta).cpu().numpy(), scores, -values.cpu()
    
    def get_attack_model_prediction_scores(self, target_model: nn.Module, dataset: Dataset) -> torch.Tensor:
        values = []
        
        if(self.ts):
            criterion = MultiBoxLoss(priors_cxcy=target_model.model.priors_cxcy, loss_type=self.loss_type).to(self.device)
        else:
            criterion = MultiBoxLoss(priors_cxcy=target_model.priors_cxcy, loss_type=self.loss_type).to(self.device)
            
        target_model.eval()
        target_model.eval()
        with torch.no_grad():
            loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4, collate_fn=collate_fn, pin_memory=True)
            for  images, boxes, labels, difficulties in loader:
                images = images.to(self.device)
                boxes = [b.to(self.device) for b in boxes]
                labels = [l.to(self.device) for l in labels]
                predicted_locs, predicted_scores = target_model(images)
                output = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar 
                if self.apply_softmax:
                    pred_scores = torch.softmax(output, dim=1)
                else:
                    pred_scores = output
                values.append(output)
        values = torch.stack(values, dim=0)
        return values.cpu()
    
    def evaluate(self, target_model: nn.Module, member_dataset: Dataset, non_member_dataset: Dataset) -> AttackResult:
        """
        Evaluates the attack by predicting the membership for the member dataset as well as the non-member dataset.
        Returns a `AttackResult`-object.
        :param target_model: The given target model
        :param member_dataset: The member dataset that was used to train the target model.
        :param non_member_dataset: The non-member dataset that was **not** used to train the target model.
        :param kwargs: Additional optional parameters for the `predict_membership`-method.
        """
        member_predictions, member_pred_scores, member_values= self.predict_membership(target_model, member_dataset) 
        non_member_predictions, non_member_pred_scores, non_member_values= self.predict_membership(target_model, non_member_dataset)
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
        
        concat_preds = torch.cat((non_member_values, member_values))
        concat_targets = torch.tensor([0 for _ in non_member_values] + [1 for _ in member_values])

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
        # tp_pred_scores = self.get_pred_score_classified_as_members(
        #     member_predictions, member_pred_scores
        # )
        # tp_mmps = tp_pred_scores.max(dim=1)[0].mean() if len(tp_pred_scores) > 0 else 0

        # fp_pred_scores = self.get_pred_score_classified_as_members(
        #     non_member_predictions, non_member_pred_scores
        # )
        # fp_mmps = fp_pred_scores.max(dim=1)[0].mean() if len(fp_pred_scores) > 0 else 0

        # fn_pred_scores = self.get_pred_score_classified_as_non_members(
        #     member_predictions, member_pred_scores
        # )
        # fn_mmps = fn_pred_scores.max(dim=1)[0].mean() if len(fn_pred_scores) > 0 else 0

        # tn_pred_scores = self.get_pred_score_classified_as_non_members(
        #     non_member_predictions,  non_member_pred_scores
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

    def get_pred_score_classified_as_members(self, membership_predictions, member_pred_scores):
        predicted_member_indices = torch.nonzero(torch.tensor(membership_predictions).squeeze()).squeeze() #get index that label is member
        if predicted_member_indices.ndim == 0:
            predicted_member_indices = predicted_member_indices.unsqueeze(0)    
        prediction_scores = []
        for idx in predicted_member_indices:
            prediction_scores.append(member_pred_scores[idx])
        
        prediction_scores = torch.stack(prediction_scores) if len(prediction_scores) > 0 else torch.empty(0)
        return prediction_scores
    
    def get_pred_score_classified_as_non_members(self, non_membership_predictions, non_member_pred_scores):
        # logical_not = not
        predicted_member_indices = torch.nonzero(torch.logical_not(torch.tensor(non_membership_predictions)).squeeze() 
                                                 ).squeeze()
        if predicted_member_indices.ndim == 0:
            predicted_member_indices = predicted_member_indices.unsqueeze(0)
        prediction_scores = []
        for idx in predicted_member_indices:
            prediction_scores.append(non_member_pred_scores[idx])

        prediction_scores = torch.stack(prediction_scores) if len(prediction_scores) > 0 else torch.empty(0)
        return prediction_scores
    
        """@return: pre_scores, labels: torch.Tensor, torch.Tensor
        """
        
    def get_metric_method(self, name: str):
        if name == "Entropy":
            #return Categorical(probs=prediction_scores).entropy()
            pass
        elif name == "loss":
            #return cross_entropy(prediction_scores, labels)
            pass
        elif name == "max_score":
            return self.get_attack_model_prediction_scores
        else:
            raise ValueError("Not know function")
        

class FasterRCNNLossAttack(MetricAttack):
    def __init__(self, apply_softmax: bool, batch_size: int = 1, log_training: bool = True, metric_method="loss"):
        super().__init__(apply_softmax, batch_size, log_training, metric_method)
        self.metric = self.get_metric_method(metric_method)
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
        values = []
        membership_labels = []
        if self.log_training:
            print('Compute attack model dataset')
        with torch.no_grad():
            shadow_model.eval()
            for i, dataset in enumerate([non_member_dataset, member_dataset]):
                loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=1, pin_memory=True, shuffle=False)                
                for img, bbox, label, scale in tqdm(loader):
                    scale = at.scalar(scale)
                    img, bbox, label = img.cuda().float(), bbox.cuda(), label.cuda()
                    loss = shadow_model.get_loss(img, bbox, label, scale)
                    if self.apply_softmax:
                        prediction_scores = torch.softmax(loss, dim=1)
                    else:
                        prediction_scores = loss
                    values.append(loss)
                    membership_labels.append(torch.full((len(label),), i))
        loss_values = torch.stack(values, dim=0).cpu().numpy()
        membership_labels = torch.cat(membership_labels, dim=0).cpu().numpy()

        # Compute threshold
        theta_best = 0.0
        num_corrects_best = 0
        for theta in np.linspace(min(loss_values), max(loss_values), 100000):
            num_corrects = (loss_values[membership_labels == 0] >=
                            theta).sum() + (loss_values[membership_labels == 1] < theta).sum()
            if num_corrects > num_corrects_best:
                num_corrects_best = num_corrects
                theta_best = theta
        self.theta = theta_best
        

        # # Compute threshold 2
        # self.shadow_fpr, self.shadow_tpr, self.thresholds, self.auroc = get_roc(membership_labels, -loss_values)
        # threshold_idx = (self.shadow_tpr - self.shadow_fpr).argmax()
        # self.theta = - self.thresholds[threshold_idx]
        
        if self.log_training:
            print(
                f'Theta set to {self.theta} achieving {num_corrects_best / (len(member_dataset) + len(non_member_dataset))}'
            )
        
    
    def predict_membership(self, target_model: nn.Module, dataset: Dataset):
        values = []
        scores = []
        target_model.eval()
        with torch.no_grad():
            loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=1, pin_memory=True, shuffle=False)                
            for img, bbox, label, scale in loader:
                scale = at.scalar(scale)
                img, bbox, label = img.cuda().float(), bbox.cuda(), label.cuda()
                loss = target_model.get_loss(img, bbox, label, scale)
                if self.apply_softmax:
                    prediction_scores = torch.softmax(loss, dim=1)
                else:
                    prediction_scores = loss
                values.append(loss)
                scores.append(torch.tensor([loss.cpu().item(),loss.cpu().item()])) # TODO, no scores, so score is replaced by loss
                
        values = torch.stack(values, dim=0) #torch
        scores = torch.cat(scores) if len(scores) > 0 else torch.empty(0)

        print("Threshold: ", self.theta)
        nums_member = sum(values < self.theta)
        nums_non_member = sum(values > self.theta)
        print("Number of members: ", nums_member)
        print("Number of non members: ", nums_non_member)
        
        return (values < self.theta).cpu().numpy(), scores, -values.cpu()

    def get_attack_model_prediction_scores(self, target_model: nn.Module, dataset: Dataset) -> torch.Tensor:
        values = []
        target_model.eval()
        with torch.no_grad():
            loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=1, pin_memory=True, shuffle=False)
            for img, bbox, label, scale in loader:
                scale = at.scalar(scale)
                img, bbox, label = img.cuda().float(), bbox.cuda(), label.cuda()
                loss = target_model.get_loss(img, bbox, label, scale)
                if self.apply_softmax:
                    pred_scores = torch.softmax(loss, dim=1)
                else:
                    pred_scores = loss
                values.append(loss)
        values = torch.stack(values, dim=0)
        return values.cpu()