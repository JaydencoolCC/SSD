import torch
from torch.utils.data import dataset, Subset, random_split
from sklearn.model_selection import train_test_split
import skimage
import torchvision

def get_train_val_split(data, train_size, seed=0, stratify=False, targets=None):
    if train_size > 0 and train_size < 1:
        training_set_length = int(train_size * len(data))
    elif train_size > 1:
        training_set_length = int(train_size)
    else:
        raise RuntimeError('Invalid argument for `size` given.')
    validation_set_length = len(data) - training_set_length
    if stratify:
        indices = list(range(len(data)))
        train_indices, validation_indices = train_test_split(indices, train_size=training_set_length,
                                                             test_size=validation_set_length, random_state=seed,
                                                             stratify=targets)
        train_set = Subset(data, train_indices)
        validation_set = Subset(data, validation_indices)
        return train_set, validation_set

    torch.manual_seed(seed)
    training_set, validation_set = random_split(data, [training_set_length, validation_set_length])

    return training_set, validation_set

def get_member_non_member_split(train_set: dataset, test_set: dataset, split_size: int):
    """
    Takes the train and test set and returns a subset of each set with the given number of samples.
    """
    # get the member subset of the target training data that can be used for finding a threshold and attacking the model
    member = get_subsampled_dataset(train_set, split_size)
    # get the non-member subset of the target test data that can be used for finding a threshold and attacking the model
    non_member = get_subsampled_dataset(test_set, split_size)

    return member, non_member

def get_subsampled_dataset(dataset, dataset_size=None, proportion=None, seed=0, stratify=False, targets=None):
    if dataset_size > len(dataset):
        raise ValueError('Dataset size is smaller than specified subsample size')
    if dataset_size is None:
        if proportion is None:
            raise ValueError('Neither dataset_size nor proportion specified')
        else:
            dataset_size = int(proportion * len(dataset))
    if stratify:
        indices = list(range(len(dataset)))
        if targets is None:
            targets = dataset.targets
        subsample_indices, _ = train_test_split(indices, train_size=dataset_size, random_state=seed, stratify=targets)
        subsample = Subset(dataset, subsample_indices)
        return subsample

    torch.manual_seed(seed)
    subsample, _ = random_split(dataset, [dataset_size, len(dataset) - dataset_size])
    return subsample



class AttackResult:
    attack_acc: float
    precision: float
    recall: float
    tpr: float
    tnr: float
    fpr: float
    fnr: float
    tp_mmps: float
    fp_mmps: float
    fn_mmps: float
    tn_mmps: float

    def __init__(
        self,
        attack_acc: float,
        precision: float,
        recall: float,
        auroc: float,
        aupr: float,
        fpr_at_tpr95: float,
        tpr: float,
        tnr: float,
        fpr: float,
        fnr: float,
        tp_mmps: float,
        fp_mmps: float,
        fn_mmps: float,
        tn_mmps: float
    ):
        self.attack_acc = attack_acc
        self.precision = precision
        self.recall = recall
        self.auroc = auroc
        self.aupr = aupr
        self.fpr_at_tpr95 = fpr_at_tpr95
        self.tpr = tpr
        self.tnr = tnr
        self.fpr = fpr
        self.fnr = fnr
        self.tp_mmps = tp_mmps
        self.fp_mmps = fp_mmps
        self.fn_mmps = fn_mmps
        self.tn_mmps = tn_mmps
        
def print_attack_results(attack_name: str, attack_results: AttackResult):
    """
    Takes the attack name and the attack result object and prints the results to the console.
    """
    print(
        f'{attack_name}: \n ' +  f'\tRecall: {attack_results.recall:.4f} \t accuracy: {attack_results.attack_acc:.4f} \t Precision: {attack_results.precision:.4f} ' +
        f'\t AUROC: {attack_results.auroc:.4f} \t AUPR: {attack_results.aupr:.4f} \t FPR@95%TPR: {attack_results.fpr_at_tpr95:.4f}' +
        f'\t FPR: {attack_results.fpr:.4f} \t TP MMPS: {attack_results.tp_mmps:.4f} ' +
        f'\t FP MMPS: {attack_results.fp_mmps:.4f} \t FN MMPS: {attack_results.fn_mmps:.4f} ' +
        f'\t TN MMPS: {attack_results.tn_mmps:.4f}'
    )


def collate_fn(batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each

def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    from models.faster_rcnn import opt
    
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = skimage.transform.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    normalize = pytorch_normalze
    return normalize(img)

def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img))
    return img.numpy()
