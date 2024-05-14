import matplotlib
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from models.faster_rcnn import opt
from datasets_utils.dataset_rcnn import Dataset, TestDataset, VOCDataset
from models.faster_rcnn_vgg16 import FasterRCNNVGG16
from example.trainer import FasterRCNNTrainer
from models.utils import array_tool as at
from models.utils.eval_tool import eval_detection_voc
import torch
from datasets_utils.dataset_tools import get_train_val_split, get_subsampled_dataset
import argparse

parser = argparse.ArgumentParser(description='Faster RCNN')
parser.add_argument('--model_type', default='target', type=str)
parser.add_argument('--dataset_name', default='VOC2007+2012', type=str, help='VOC2007+2012, voc07')
parser.add_argument('--train_size', default=8275, type=int)
parser.add_argument('--test_size', default=2476, type=int)
parser.add_argument('--load_path', default='checkpoints/fasterrcnn/fasterrcnn_shadow_epoch_17', type=str)
parser.add_argument('--epochs', default=None, type=int)
args = parser.parse_args()

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True 

torch.set_num_threads(2)

def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    opt._parse(kwargs)
    # # Custom dataloaders
    train_dataset = VOCDataset(opt, data_name=args.dataset_name, action='train')
    train_dataset_size = len(train_dataset)
    test_dataset = VOCDataset(opt, split='test', data_name="VOC2007", action='eval')

    train_size = len(train_dataset) // 2
    test_size = len(test_dataset) // 2
    train_dataset = get_subsampled_dataset(train_dataset, dataset_size=train_size*2, proportion=None)
    train_target, train_shadow = get_train_val_split(train_dataset, train_size, seed=opt.seed, stratify=False, targets=None)
    
    test_dataset = get_subsampled_dataset(test_dataset, dataset_size=test_size*2, proportion=0.5)
    test_target, test_shadow= get_train_val_split(test_dataset, test_size, seed=opt.seed, stratify=False, targets=None)
    
    
    trainDataLoader_target = torch.utils.data.DataLoader(train_target, 
                                  batch_size=1, 
                                  shuffle=True, 
                                  pin_memory=True,
                                  num_workers=opt.num_workers)
    
    trainDataLoader_shadow = torch.utils.data.DataLoader(train_shadow, 
                                  batch_size=1, 
                                  shuffle=True,
                                  pin_memory=True, 
                                  num_workers=opt.num_workers)
    
    testDataLoade_target = torch.utils.data.DataLoader(test_target,
                                       batch_size=1,
                                       num_workers=opt.num_workers,
                                       shuffle=False,
                                       pin_memory=False,
                                       )
    
    
    testDataLoade_shadow = torch.utils.data.DataLoader(test_shadow,
                                       batch_size=1,
                                       num_workers=opt.num_workers,
                                       shuffle=False,
                                       pin_memory=False,
                                       )

    print("train target dataset: {}".format(len(train_target)))
    print("test target dataset: {}".format(len(test_target)))

    if(args.dataset_name == 'VOC2007+2012'):
        decay_lr_at = [12000, 16000]
        max_iter = 18000     
        decay_lr_epoch = [it // (train_dataset_size // 16) for it in decay_lr_at] # iter to epoch
        epochs = max_iter // (train_dataset_size // 16)
    elif(args.dataset_name == 'VOC2007'):
        decay_lr_at = [4000, 5000]
        max_iter = 6000
        decay_lr_epoch = [it // (train_dataset_size // 16) for it in decay_lr_at] # iter to epoch
        epochs = max_iter // (train_dataset_size // 16)
    else:
        raise ValueError("dataset name is error")
    if (args.epochs is not None):
        epochs = args.epochs
        
    print("decay_lr_epoch: %s epochs: %s" % (decay_lr_epoch, epochs))
    faster_rcnn = FasterRCNNVGG16().cuda()
    
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    
    addition = args.model_type + '_epoch_' + str(epochs) + '_%s'%args.dataset_name
    if(args.model_type == "target"):
        trainDataLoader = trainDataLoader_target
        testDataLoader = testDataLoade_target
    elif(args.model_type == "shadow"):
        trainDataLoader = trainDataLoader_shadow
        testDataLoader = testDataLoade_shadow
    else:
        raise ValueError("Invalid type")
        
    #load model
    if args.load_path is not None:
        trainer.load(args.load_path)
        print('load pretrained model from %s' % args.load_path)
        
    best_map = 0
    lr_ = opt.lr
    start_epoch = 0
    for epoch in range(start_epoch, epochs):
        total_loss = []
        for ii, (img, bbox, label, scale) in enumerate(tqdm(trainDataLoader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox.cuda(), label.cuda()
            loss = trainer.train_step(img, bbox, label, scale)
            total_loss.append(loss.item())
            
        print("epoch: %d   loss: %f" % (epoch, sum(total_loss)/len(total_loss)))

        eval_result = eval(testDataLoader, trainer.faster_rcnn, test_num=10000)
        print(eval_result)
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
        if epoch in decay_lr_epoch:
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == epochs-1: 
            save_path = trainer.save(addition=addition, epoch=epoch+1)
            break
        
        print("eval result: ", best_map)

if __name__ == '__main__':
    train()
