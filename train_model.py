import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *
from datasets_utils.dataset_tools import get_train_val_split, get_subsampled_dataset, print_attack_results, get_member_non_member_split, collate_fn
import argparse
import copy


# Parameters
parser = argparse.ArgumentParser(description='PyTorch SSD Evaluation')
parser.add_argument('--checkpoint', default='./checkpoint/checkpoint_ssd300.pth.tar', type=str, help='Checkpoint path')
parser.add_argument('--seed', default=42, type=int, help='Data folder')
args = parser.parse_args()

# Data parameters
data_folder = './data'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 32  # batch size
iterations = 120000  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 200  # print training status every __ batches
lr = 1e-3  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True

#checkpoint = "./checkpoint/checkpoint_ssd300.pth.tar" # path to model checkpoint, None if none

checkpoint_target_file = "./checkpoint/target_newssd300.pth.tar"
checkpoint_shadow_file = "./checkpoint/shadow_newssd300.pth.tar"
train_zero = True

def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    if train_zero:
        start_epoch = 0
        model_target = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model_target.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
                    
        optimizer_target = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)
        criterion_target = MultiBoxLoss(priors_cxcy=model_target.priors_cxcy).to(device)

        model_shadow = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model_shadow.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
                    
        optimizer_shadow = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)
        
        criterion_shadow = MultiBoxLoss(priors_cxcy=model_shadow.priors_cxcy).to(device)
        
    else:
        checkpoint_target = torch.load(checkpoint_target_file)
        start_epoch = checkpoint_target['epoch'] + 1
        print('\nLoaded target checkpoint from epoch %d.\n' % start_epoch)
        model_target = checkpoint_target['model']
        optimizer_target = checkpoint_target['optimizer']
        criterion_target = MultiBoxLoss(priors_cxcy=model_target.priors_cxcy).to(device)
       
        checkpoint_shadow = torch.load(checkpoint_shadow_file)
        start_epoch = checkpoint_shadow['epoch'] + 1
        print('\nLoaded shadow checkpoint from epoch %d.\n' % start_epoch)
        model_shadow = checkpoint_shadow['model']
        optimizer_shadow = checkpoint_shadow['optimizer']
        criterion_shadow = MultiBoxLoss(priors_cxcy=model_shadow.priors_cxcy).to(device)
       

    # Move to default device
    model_target = model_target.to(device)
    model_shadow = model_shadow.to(device)
    

    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    
    test_dataset = PascalVOCDataset(data_folder,
                                     split='test',
                                     keep_difficult=keep_difficult)
    
    
    print("train_dataset: {}".format(len(train_dataset)))
    print("test_dataset: {}".format(len(test_dataset)))

    train_size = len(train_dataset) // 2
    test_size = len(test_dataset) // 2
    
    train_dataset = get_subsampled_dataset(train_dataset, dataset_size=train_size*2, proportion=None)
    train_target, train_shadow = get_train_val_split(train_dataset, train_size, seed=args.seed, stratify=False, targets=None)
    
    test_dataset = get_subsampled_dataset(test_dataset, dataset_size=test_size*2, proportion=0.5)
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
        
    
    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    epochs = iterations // (len(train_dataset) // 32)
    decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]

    # Epochs
    #epochs = 400
    print("Training target model")
    addition = 'target_400_'
    train(trainDataLoader_target, model_target, criterion_target, optimizer_target, epochs, addition)
    
    print("Training shadow model")
    addition = 'shadow_400_'
    train(trainDataLoader_shadow, model_shadow, criterion_shadow, optimizer_shadow, epochs, addition)
    
    
    
def train(train_loader, model, criterion, optimizer, epochs, addition):
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train_epoch(train_loader= train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, addition)


def train_epoch(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'lr {lr}\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses, lr=optimizer.param_groups[1]['lr']))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


if __name__ == '__main__':
    main()