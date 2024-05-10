import torch
from datasets_utils.dataset_tools import collate_fn


def get_model_prediction_scores(model, apply_softmax, dataset, batch_size=128, num_workers=4):
    model.eval()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
        
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    prediction_scores_total = []
    with torch.no_grad():
        model.to(device)
        for images, boxes, labels, difficulties in dataloader:
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            prediction_locs, prediction_scores = model(images)
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(prediction_locs, prediction_scores,
                                                                                       min_score=0.2, max_overlap=0.5,
                                                                                       top_k=200)
            det_scores_batch = [score.max().cpu() for score in det_scores_batch]
            prediction_scores_total.extend(det_scores_batch)

    prediction_scores_total = torch.stack(prediction_scores_total) if len(prediction_scores_total) > 0 else torch.empty(0)
    return prediction_scores_total

def get_model_prediction_scores_with_lables(model, apply_softmax, dataset, batch_size=128, num_workers=4):
    model.eval()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    prediction_scores = []
    labels = []
    with torch.no_grad():
        model.to(device)
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if apply_softmax:
                output = model(x).softmax(dim=1)
            else:
                output = model(x)
            prediction_scores.append(output.cpu())
            labels.append(y.cpu())

    prediction_scores = torch.cat(prediction_scores) if len(prediction_scores) > 0 else torch.empty(0)
    labels = torch.cat(labels) if len(labels) > 0 else torch.empty(0)
    return prediction_scores, labels