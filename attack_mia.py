from tqdm import tqdm
from pprint import PrettyPrinter
import pickle
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from datasets import PascalVOCDataset
from utils import *

from attack import ThresholdAttack, SalemAttack, EntropyAttack, MetricAttack, DetAttack
from datasets_utils.dataset_tools import get_train_val_split, get_subsampled_dataset, print_attack_results, get_member_non_member_split, collate_fn

from model import SSD300, MultiBoxLoss
from utils_tools.loss import AttackLoss
# argparse
parser = argparse.ArgumentParser(description='PyTorch SSD Evaluation')
parser.add_argument('--checkpoint_target', default='./checkpoint/target_newssd300.pth.tar', type=str, help='Checkpoint path')
parser.add_argument('--checkpoint_shadow', default='./checkpoint/shadow_newssd300.pth.tar', type=str, help='Checkpoint path')
parser.add_argument('--data_folder', default='./data', type=str, help='Data folder')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for evaluation')
parser.add_argument('--workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--keep_difficult', default=True, type=bool, help='Keep difficult ground truth objects in evaluation')
parser.add_argument('--device', default='cuda', type=str, help='Device used for inference')
parser.add_argument('--split', default='test', type=str, help='Data split to evaluate on')
parser.add_argument('--action', default='test', type=str, help='train or test')
parser.add_argument('--seed', default=42, type=int, help='seed')
args = parser.parse_args()

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()
# Parameters
data_folder = './data'
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 1
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = './checkpoint/epoch_400_ssd300.pth.tar'
checkpoint_target = args.checkpoint_target
checkpoint_shadow = args.checkpoint_shadow

# Load model checkpoint that is to be evaluated
checkpoint_target = torch.load(checkpoint_target)
checkpoint_shadow = torch.load(checkpoint_shadow)

model_target = checkpoint_target['model']
model_shadow = checkpoint_shadow['model']

model_target = model_target.to(device)
model_shadow = model_shadow.to(device)
# Switch to eval mode
model_target.eval()
model_shadow.eval()



# Load test data
# Custom dataloaders
train_dataset = PascalVOCDataset(data_folder,
                                 split='train',
                                 keep_difficult=keep_difficult, 
                                 action='test')

test_dataset = PascalVOCDataset(data_folder,
                                     split='test',
                                     keep_difficult=keep_difficult,
                                     action='test')


test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                               collate_fn=collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

print("train_dataset: {}".format(len(train_dataset)))
print("test_dataset: {}".format(len(test_dataset)))

train_size = len(train_dataset) // 2
test_size = len(test_dataset) // 2

train_dataset = get_subsampled_dataset(train_dataset, dataset_size=train_size*2, proportion=None)
train_target, train_shadow = get_train_val_split(train_dataset, train_size, seed=args.seed, stratify=False, targets=None)

test_dataset = get_subsampled_dataset(test_dataset, dataset_size=test_size*2, proportion = None)
test_target, test_shadow= get_train_val_split(test_dataset, test_size, seed=args.seed, stratify=False, targets=None)


trainDataLoader_target = torch.utils.data.DataLoader(train_target, batch_size=batch_size, shuffle=True,
                                           collate_fn=collate_fn, num_workers=workers,
                                           pin_memory=True)  # note that we're passing the collate function here

trainDataLoader_shadow = torch.utils.data.DataLoader(train_shadow, batch_size=batch_size, shuffle=True,
                                           collate_fn=collate_fn, num_workers=workers,
                                           pin_memory=True)  # note that we're passing the collate function here

testDataLoade_target = torch.utils.data.DataLoader(test_target, batch_size=batch_size, shuffle=False,
                                           collate_fn=collate_fn, num_workers=workers,
                                           pin_memory=True)  # note that we're passing the collate function here

testDataLoade_shadow = torch.utils.data.DataLoader(test_shadow, batch_size=batch_size, shuffle=False,
                                               collate_fn=collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

member_target, non_member_target = get_member_non_member_split(train_target, test_target, 2000)
member_shadow, non_member_shadow = get_member_non_member_split(train_shadow, test_shadow, 2000)

def evaluate(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()
    criterion = AttackLoss().to(device)
    
    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py
    loss_sample = []
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
            
            # det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=0.2,
            #                                                  max_overlap=0.5, top_k=200)
            loss = criterion(det_boxes, det_scores, det_labels, boxes, labels)  # scalar  
            #loss = criterion(det_boxes_batch, det_scores_batch, det_labels_batch, boxes, labels)  # scalar            
                      
            #print("loss: ", loss)
            loss_sample.append(loss.cpu().item())
            
        mAP = 0
        APs = 0
        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
        #results = calculate_gt(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
        #write_to_file(results)
        #write results to f   
    # Print AP for each class
    loss_to_file(loss_sample)
    pp.pprint(APs)
    print('\nMean Average Precision (mAP): %.3f' % mAP)
    
    attacks = [
            #SalemAttack(apply_softmax= False, k=2304, log_training=True),
            #MetricAttack(apply_softmax=False, batch_size=1),
            DetAttack(apply_softmax=False, batch_size=1)
            ]

    name = [
            #"SalemAttack", 
            "MetricAttack"
            "DetAttack"
            ]
    attack_list = []
    for i in range(len(attacks)):
        attack = attacks[i]
        attack.learn_attack_parameters(model_target, member_target, non_member_target)
        result = attack.evaluate(model_target, member_target, non_member_target)
        attack_list.append(result)
        print_attack_results(name[i], result)    

def write_to_file(results):
    file = args.split + '_dataset.pkl'
    with open(file, 'wb') as file:
        # 写入数据
        for result in results:
            for key in result.keys():
                value = result[key]                     
                result[key] = value.cpu().item() if torch.is_tensor(result[key]) else value
        pickle.dump(results, file)

def loss_to_file(loss):
    file = args.split + '_loss.pkl'
    with open(file, 'wb') as file:
        pickle.dump(loss,file)
        
if __name__ == '__main__':
    member_target_loader = torch.utils.data.DataLoader(member_target, batch_size=1, shuffle=False,
                                           collate_fn=collate_fn, num_workers=workers,
                                           pin_memory=True)  # note that we're passing the collate fun
    
    non_member_target_loader = torch.utils.data.DataLoader(non_member_target, batch_size=1, shuffle=False,
                                           collate_fn=collate_fn, num_workers=workers,
                                           pin_memory=True)  # note that we're passing the collate fun
    
    member_shadow_loader = torch.utils.data.DataLoader(member_shadow, batch_size=64, shuffle=False,
                                           collate_fn=collate_fn, num_workers=workers,
                                           pin_memory=True)  # note that we're passing the collate fun
    
    non_member_shadow_loader = torch.utils.data.DataLoader(non_member_shadow, batch_size=64, shuffle=False,
                                           collate_fn=collate_fn, num_workers=workers,
                                           pin_memory=True)  # note that we're passing the collate fun
    
    print(len(member_target))
    print(len(non_member_target))
    print(args.split)
    
    if args.split == "member":
        evaluate(member_target_loader, model_target)
    elif args.split == "non_member":
        evaluate(non_member_target_loader, model_target)
    else:
        raise ValueError("Invalid data_type: " + args.data_type)
        
