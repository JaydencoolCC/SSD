import os
import xml.etree.ElementTree as ET

import numpy as np

from PIL import Image


def read_image(path, dtype=np.float32, color=True):
    """Read an image from a file.

    This function reads an image from given file. The image is CHW format and
    the range of its value is :math:`[0, 255]`. If :obj:`color = True`, the
    order of the channels is RGB.

    Args:
        path (str): A path of image file.
        dtype: The type of array. The default value is :obj:`~numpy.float32`.
        color (bool): This option determines the number of channels.
            If :obj:`True`, the number of channels is three. In this case,
            the order of the channels is RGB. This is the default behaviour.
            If :obj:`False`, this function returns a grayscale image.

    Returns:
        ~numpy.ndarray: An image.
    """

    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))
    
class VOCBboxDataset:
    """Bounding box dataset for PASCAL `VOC`_.

    .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    The index corresponds to each image.

    When queried by an index, if :obj:`return_difficult == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.
    If :obj:`return_difficult == True`, this dataset returns corresponding
    :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
    that indicates whether bounding boxes are labeled as difficult or not.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`VOC_BBOX_LABEL_NAMES`.

    The array :obj:`difficult` is a one dimensional boolean array of shape
    :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
    If :obj:`use_difficult` is :obj:`False`, this array is
    a boolean array with all :obj:`False`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`

    Args:
        data_dir (string): Path to the root of the training data. 
            i.e. "/data/image/voc/VOCdevkit/VOC2007/"
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        year ({'2007', '2012'}): Use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
        return_difficult (bool): If :obj:`True`, this dataset returns
            a boolean array
            that indicates whether bounding boxes are labeled as difficult
            or not. The default value is :obj:`False`.

    """

    def __init__(self, data_dir, split='trainval', use_difficult=False, return_difficult=False, data_name='VOC2007'):
        
        image_id = []
        annotations_id = []
        if data_name == 'VOC2007' or data_name == 'VOC2012':
            data_file = os.path.join(data_dir, data_name)
            id_list_file = os.path.join(data_file, 'ImageSets/Main/{0}.txt'.format(split))            
            self.ids = [id_.strip() for id_ in open(id_list_file)]            
            for id in self.ids:
                image_id.append(os.path.join(data_file, 'JPEGImages', id + '.jpg'))
                annotations_id.append(os.path.join(data_file, 'Annotations', id + '.xml'))
                
        elif data_name == 'VOC2007+2012':
            data_file = os.path.join(data_dir, 'VOC2007')
            id_list_file = os.path.join(
                data_file, 'ImageSets/Main/{0}.txt'.format(split))
            self.ids = [id_.strip() for id_ in open(id_list_file)]
            for id in self.ids:
                image_id.append(os.path.join(data_file, 'JPEGImages', id + '.jpg'))
                annotations_id.append(os.path.join(data_file, 'Annotations', id + '.xml'))
                
            data_file = os.path.join(data_dir, 'VOC2012')
            id_list_file = os.path.join(
                data_file, 'ImageSets/Main/{0}.txt'.format(split))
            self.ids = [id_.strip() for id_ in open(id_list_file)]
            for id in self.ids:
                image_id.append(os.path.join(data_file, 'JPEGImages', id + '.jpg'))
                annotations_id.append(os.path.join(data_file, 'Annotations', id + '.xml'))
        else :
            raise ValueError('data_name should be VOC2007, VOC2012 or VOC2007+2012')

        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES
        self.image_id = image_id
        self.annotations_id = annotations_id
        

    def __len__(self):
        return len(self.image_id)

    def __getitem__(self, idx):
        """
        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        
        # id_ = self.ids[i]
        # anno = ET.parse(
        #     os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        anno = ET.parse(self.annotations_id[idx])
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object isdifficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=bool).astype(np.uint8)  # PyTorch don't support np.bool

        # Load a image
        # img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(self.image_id[idx], color=True)
        return img, bbox, label, difficult

VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')

