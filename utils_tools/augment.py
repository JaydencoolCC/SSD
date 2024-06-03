import numpy as np

def translate_bbox(bbox, img_shape, max_trans=10):
    """
    Translate the bounding box by a random offset within the specified range.

    Parameters:
    - bbox: A tuple (x_min, y_min, x_max, y_max) representing the bounding box.
    - img_shape: The shape of the image as (height, width).
    - max_trans: The maximum translation in pixels.

    Returns:
    - Translated bounding box as (x_min, y_min, x_max, y_max).
    """
    height, width = img_shape
    x_min, y_min, x_max, y_max = bbox

    tx = np.random.randint(-max_trans, max_trans)
    ty = np.random.randint(-max_trans, max_trans)

    new_x_min = max(0, x_min + tx)
    new_y_min = max(0, y_min + ty)
    new_x_max = min(width, x_max + tx)
    new_y_max = min(height, y_max + ty)

    return new_x_min, new_y_min, new_x_max, new_y_max

# 示例边界框 (x_min, y_min, x_max, y_max)
bbox = (50, 50, 150, 150)
img_shape = (200, 200)

# 进行平移扰动
new_bbox = translate_bbox(bbox, img_shape)

print("Original bbox:", bbox)
print("Translated bbox:", new_bbox)
