from tqdm import tqdm
from pprint import PrettyPrinter
import pickle
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from datasets import PascalVOCDataset
from utils import *
from utils_tools.utils import get_temp_calibrated_models
from attack import ThresholdAttack, SalemAttack, EntropyAttack, MetricAttack, DetAttack
from datasets_utils.dataset_tools import get_train_val_split, get_subsampled_dataset, print_attack_results, get_member_non_member_split, collate_fn
import torch
from models.ssd import SSD300, MultiBoxLoss
from utils_tools.loss import AttackLoss
# argparse
parser = argparse.ArgumentParser(description='PyTorch SSD Evaluation')
parser.add_argument('--checkpoint_target', default='./checkpoint/target_newssd300.pth.tar', type=str, help='Checkpoint path')
parser.add_argument('--checkpoint_shadow', default='./checkpoint/shadow_newssd300.pth.tar', type=str, help='Checkpoint path')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for evaluation')
parser.add_argument('--workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--keep_difficult', default=True, type=bool, help='Keep difficult ground truth objects in evaluation')
parser.add_argument('--device', default='cuda', type=str, help='Device used for inference')
parser.add_argument('--split', default='member', type=str, help='Data split to evaluate on')
parser.add_argument('--action', default='test', type=str, help='train or test')
parser.add_argument('--seed', default=42, type=int, help='seed')
parser.add_argument('--use_temp', action='store_true', help='use temperature scaling')
parser.add_argument('--temp_value', default=5, type=float, help='temperature value')
parser.add_argument('--dataset_name', default='voc07+12', type=str, help='voc07+12, voc07')
parser.add_argument('--data_folder', default='./data', type=str, help='Data folder')

args = parser.parse_args()

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()
# Parameters
data_folder = os.path.join(args.data_folder, args.dataset_name) # folder with data files

keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 1
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def evaluate(test_loader, model, model_target, model_shadow):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)
    
    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py
    loss_sample = []
    iou_sample = []
    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
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
            #loss = criterion(det_boxes, det_scores, det_labels, boxes, labels)  # scalar  
            #loss = criterion(det_boxes_batch, det_scores_batch, det_labels_batch, boxes, labels)  # scalar            
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar            
                      
            #print("loss: ", loss)
            loss_sample.append(loss.cpu().item())
            
            #APs, mAP,iou= calculate_attack(det_boxes_batch, det_labels_batch, det_scores_batch, boxes, labels, difficulties)
            
            #print('\nMean Average Precision (mAP): %.3f' % mAP)
            
        mAP = 0
        APs = 0
        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
        #results = calculate_gt(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
        #write_to_file(results)
        #write results to f   
    # Print AP for each class
    loss_to_file(iou_sample)
    pp.pprint(APs)
    print('\nMean Average Precision (mAP): %.3f' % mAP)
    
#def attack():
    if args.use_temp:
        model_target, model_shadow = get_temp_calibrated_models(
            target_model=model_target,
            shadow_model=model_shadow,
            non_member_target=non_member_target,
            non_member_shadow=non_member_shadow,
            temp_value=args.temp_value
        )

    attacks = [
            #SalemAttack(apply_softmax=False, batch_size=64, k=218300, log_training=True),
            #ThresholdAttack(apply_softmax= False, batch_size=64),
            MetricAttack(apply_softmax=False, batch_size=1),
            #DetAttack(apply_softmax=False, batch_size=1)
            #EntropyAttack(apply_softmax=False, batch_size=1)
            ]

    name = [
            #"SalemAttack", 
            "MetricAttack"
            #"DetAttack"
            ]
    attack_list = []
    for i in range(len(attacks)):
        attack = attacks[i]
        attack.learn_attack_parameters(model_shadow, member_shadow, non_member_shadow)
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
        evaluate(member_target_loader, model_target, model_target, model_shadow)
    elif args.split == "non_member":
        evaluate(non_member_target_loader, model_target, model_target, model_shadow)
    else:
        raise ValueError("Invalid data_type: " + args.split)
        