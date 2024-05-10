from tqdm import tqdm
from pprint import PrettyPrinter
import pickle
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from datasets import PascalVOCDataset
from utils import *
from utils_tools.utils import get_temp_calibrated_models
from attack import ThresholdAttack, SalemAttack, EntropyAttack, MetricAttack, DetAttack, FasterRCNNLossAttack
from datasets_utils.dataset_tools import get_train_val_split, get_subsampled_dataset, print_attack_results, get_member_non_member_split, collate_fn
from datasets_utils.dataset_rcnn import VOCDataset, TestDataset
from models.faster_rcnn import opt
from datasets_utils.dataset_rcnn import eval
from models.utils import array_tool as at
from example.trainer import FasterRCNNTrainer
from models.faster_rcnn_vgg16 import FasterRCNNVGG16

# argparse
parser = argparse.ArgumentParser(description='PyTorch SSD Evaluation')
parser.add_argument('--checkpoint_target', default='./checkpoints/fasterrcnn/fasterrcnn_target_epoch_17', type=str, help='Checkpoint path')
parser.add_argument('--checkpoint_shadow', default='./checkpoints/fasterrcnn/fasterrcnn_shadow_epoch_17', type=str, help='Checkpoint path')
parser.add_argument('--data_folder', default='./data', type=str, help='Data folder')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for evaluation')
parser.add_argument('--workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--keep_difficult', default=True, type=bool, help='Keep difficult ground truth objects in evaluation')
parser.add_argument('--device', default='cuda', type=str, help='Device used for inference')
parser.add_argument('--split', default='member', type=str, help='Data split to evaluate on')
parser.add_argument('--action', default='test', type=str, help='train or test')
parser.add_argument('--seed', default=42, type=int, help='seed')
parser.add_argument('--cuda', default=1, type=int, help='chose cuda')
parser.add_argument('--use_temp', action='store_true', help='use temperature scaling')
parser.add_argument('--temp_value', default=5, type=float, help='temperature value')

args = parser.parse_args()

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()
# Parameters
data_folder = './data'
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 1
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_target = args.checkpoint_target
checkpoint_shadow = args.checkpoint_shadow

# Load model checkpoint that is to be evaluated
checkpoint_target = torch.load(checkpoint_target)
checkpoint_shadow = torch.load(checkpoint_shadow)

model_target = FasterRCNNVGG16()
model_shadow = FasterRCNNVGG16()


model_target.load_state_dict(checkpoint_target['model'])
model_shadow.load_state_dict(checkpoint_shadow['model'])
  

model_target = model_target.to(device)
model_shadow = model_shadow.to(device)
# Switch to eval mode
model_target.eval()
model_shadow.eval()

train_dataset = VOCDataset(opt, data_name='VOC2007+2012', action='attack')

test_dataset = VOCDataset(opt, split='test', data_name='VOC2007', action='attack')

train_size = len(train_dataset) // 2
test_size = len(test_dataset) // 2

train_dataset = get_subsampled_dataset(train_dataset, dataset_size=train_size*2, proportion=None)
train_target, train_shadow = get_train_val_split(train_dataset, train_size, seed=args.seed, stratify=False, targets=None)

test_dataset = get_subsampled_dataset(test_dataset, dataset_size=test_size*2, proportion=0.5)
test_target, test_shadow= get_train_val_split(test_dataset, test_size, seed=args.seed, stratify=False, targets=None)


trainDataLoader_target = torch.utils.data.DataLoader(train_target, 
                              batch_size=1, 
                              shuffle=True, 
                              num_workers=num_workers)

trainDataLoader_shadow = torch.utils.data.DataLoader(train_shadow, 
                              batch_size=1, 
                              shuffle=True, 
                              num_workers=num_workers)

testDataLoade_target = torch.utils.data.DataLoader(test_target,
                                   batch_size=1,
                                   num_workers=num_workers,
                                   shuffle=False,
                                   )


testDataLoade_shadow = torch.utils.data.DataLoader(test_shadow,
                                   batch_size=1,
                                   num_workers=num_workers,
                                   shuffle=False,
                                   )
print("train target dataset: {}".format(len(train_target)))
print("test target dataset: {}".format(len(test_target)))


member_target, non_member_target = get_member_non_member_split(train_target, test_target, 2000)
member_shadow, non_member_shadow = get_member_non_member_split(train_shadow, test_shadow, 2000)


def evaluate(test_loader, model_target, model_shadow):
    # eval_result = eval(test_loader, model_target, test_num=opt.test_num)
    # print(eval_result)

    
    model_target = FasterRCNNTrainer(model_target).cuda()
    model_shadow = FasterRCNNTrainer(model_shadow).cuda()
    
    # for ii, (img, bbox, label, scale) in tqdm(enumerate(test_loader)):
    #     scale = at.scalar(scale)
    #     img, bbox, label = img.cuda().float(), bbox.cuda(), label.cuda()
    #     loss = model_target.get_loss(img, bbox, label, scale)
        
    attacks = [
            FasterRCNNLossAttack(apply_softmax=False, batch_size=1),
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
                                           num_workers=num_workers,
                                           pin_memory=True)  # note that we're passing the collate fun
    
    non_member_target_loader = torch.utils.data.DataLoader(non_member_target, batch_size=1, shuffle=False,
                                           num_workers=num_workers,
                                           pin_memory=True)  # note that we're passing the collate fun
    
    member_shadow_loader = torch.utils.data.DataLoader(member_shadow, batch_size=1, shuffle=False,
                                           num_workers=num_workers,
                                           pin_memory=True)  # note that we're passing the collate fun
    
    non_member_shadow_loader = torch.utils.data.DataLoader(non_member_shadow, batch_size=1, shuffle=False,
                                           num_workers=num_workers,
                                           pin_memory=True)  # note that we're passing the collate fun
    
    print(len(member_target))
    print(len(non_member_target))
    print(args.split)
    
    #eval_dataset = VOCDataset(opt, data_name='VOC2007', split='test', action='eval')
    test_dataset = TestDataset(opt, split='test', data_name='VOC2007')
    eval_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                           num_workers=num_workers,
                                           pin_memory=True)  # note that we're passing the collate fun
    
    if args.split == "member":
        evaluate(member_target_loader, model_target, model_shadow)
    elif args.split == "non_member":
        evaluate(non_member_target_loader, model_target, model_shadow)
    else:
        raise ValueError("Invalid data_type: " + args.split)
        