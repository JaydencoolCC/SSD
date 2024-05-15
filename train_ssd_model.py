import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *
from datasets_utils.dataset_tools import get_train_val_split, get_subsampled_dataset, print_attack_results, get_member_non_member_split, collate_fn
import argparse
import copy
from opacus import PrivacyEngine


# Parameters
parser = argparse.ArgumentParser(description='PyTorch SSD Evaluation')
parser.add_argument('--checkpoint', default='./checkpoint/checkpoint_ssd300.pth.tar', type=str, help='Checkpoint path')
parser.add_argument('--seed', default=42, type=int, help='Data folder')
parser.add_argument('--model_type', default='target', type=str)
parser.add_argument('--label_smoothing', action='store_true', default=False, help='Use label smoothing')
parser.add_argument('--factor', default=0.1, type=float)
parser.add_argument('--dropout', default=None, type=float)
parser.add_argument("--disable_dp",action="store_true", default=False, help="Disable privacy training and just train with vanilla SGD")
parser.add_argument("--sigma", type=float, default=1.2, metavar="S", help="Noise multiplier")
parser.add_argument("--max_per_sample_grad_norm", type=float, default=1.0, metavar="C", help="Clip per-sample gradients to this norm")
parser.add_argument("--delta", type=float, default=1e-5, metavar="D", help="Target delta") 
parser.add_argument('--dataset_name', default='voc07', type=str, help='voc07+12, voc07')
parser.add_argument('--epochs', default=None, type=int)
parser.add_argument('--data_folder', default='./data', type=str)
args = parser.parse_args()

# Data parameters
data_folder = os.path.join(args.data_folder, args.dataset_name) # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 32  # batch size
iterations = 120000  # number of iterations to train
workers = 8  # number of workers for loading data in the DataLoader
print_freq = 200  # print training status every __ batches
lr = 1e-3  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True


checkpoint_file = args.checkpoint
train_zero = True

def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    if train_zero:
        start_epoch = 0
        if(args.dropout is not None):
            model = SSD300(n_classes=n_classes, dropout=args.dropout)
        else:
            model = SSD300(n_classes=n_classes)
            
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
                    
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)
        criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy, label_smoothing=args.label_smoothing, factor=args.factor).to(device)

    else:
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded target checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy, label_smoothing=args.label_smoothing, factor=args.factor).to(device)
    # Move to default device
    model = model.to(device)
    

    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    
    test_dataset = PascalVOCDataset(data_folder,
                                     split='test',
                                     keep_difficult=keep_difficult)
    

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
    
    
    print("train target dataset: {}".format(len(train_target)))
    print("test target dataset: {}".format(len(test_target)))
    
    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    if(args.dataset_name == "voc07"):
        iterations = 30000
        decay_lr_at = [20000, 25000]
    elif(args.dataset_name == "voc07+12"):
        decay_lr_at = [80000, 100000]
        iterations = 120000
        
    epochs = iterations // (len(train_dataset) // 32)
    decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]
    
    # Epochs
    if(args.epochs is not None):
        epochs=1
        
    print("Training %s model  epochs %d" % (args.model_type, epochs))
    print("decay_lr_at:", decay_lr_at)
    
    addition = args.model_type + "_epochs_" + str(epochs) + "_" + args.dataset_name
    if(args.label_smoothing):
        addition += "LS_" + str(args.factor) +"_"
    if(args.dropout):
        addition += "drop_" + str(args.dropout) +"_"  
    if(args.disable_dp):
        addition += "dp_" + str(args.delta)  
    if(args.model_type == "target"):
        trainDataLoader = trainDataLoader_target
    elif(args.model_type == "shadow"):
        trainDataLoader = trainDataLoader_shadow
    else:
        raise ValueError
    
    if args.disable_dp:
        print("Use DP to protect privacy")
        privacy_engine = PrivacyEngine()
        model, optimizer, trainDataLoader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=trainDataLoader,
            noise_multiplier=args.sigma,
            max_grad_norm=args.max_per_sample_grad_norm,
        )
    else:
        privacy_engine = None
            
    train(trainDataLoader, model, criterion, optimizer, epochs, addition, privacy_engine)
    
def train(train_loader, model, criterion, optimizer, epochs, addition, privacy_engine):
    file_path = "./checkpoint/ssd/"
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train_epoch(train_loader= train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              privacy_engine=privacy_engine)

        # Save checkpoint
        if(epoch == epochs-1):
            save_checkpoint(epoch+1, model, optimizer, addition, file_path)


def train_epoch(train_loader, model, criterion, optimizer, epoch, privacy_engine):
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

        #zero_grad
        optimizer.zero_grad()
        
        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar
        #print(loss)
        # Backward prop.
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        if privacy_engine is not None:
            epsilon = privacy_engine.get_epsilon(args.delta)
            print(
                    f"(ε = {epsilon:.2f}, δ = {args.delta})"
                )
            
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'lr {lr}'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses, lr=optimizer.param_groups[1]['lr']))
                
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


if __name__ == '__main__':
    main()
