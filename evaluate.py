from utils import *
from datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter
import pickle
import argparse

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = './data'
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 64
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = './checkpoint/checkpoint_ssd300.pth.tar'

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()

# argparse
parser = argparse.ArgumentParser(description='PyTorch SSD Evaluation')
parser.add_argument('--checkpoint', default='./checkpoint/checkpoint_ssd300.pth.tar', type=str, help='Checkpoint path')
parser.add_argument('--data_folder', default='./data', type=str, help='Data folder')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for evaluation')
parser.add_argument('--workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--keep_difficult', default=True, type=bool, help='Keep difficult ground truth objects in evaluation')
parser.add_argument('--device', default='cuda', type=str, help='Device used for inference')
parser.add_argument('--split', default='test', type=str, help='Data split to evaluate on')
parser.add_argument('--action', default='test', type=str, help='train or test')

args = parser.parse_args()

# Load test data
test_dataset = PascalVOCDataset(data_folder,
                                split=args.split,
                                keep_difficult=keep_difficult, action= args.action)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)


def evaluate(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.2, max_overlap=0.45,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        mAP = 0
        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
        results = calculate_gt(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
        write_to_file(results)
        #write results to f   
    # Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)

def write_to_file(results):
    file = args.split + '_dataset.pkl'
    with open(file, 'wb') as file:
        # 写入数据
        for result in results:
            for key in result.keys():
                value = result[key]                     
                result[key] = value.cpu().item() if torch.is_tensor(result[key]) else value
        pickle.dump(results, file)
    
if __name__ == '__main__':
    evaluate(test_loader, model)
