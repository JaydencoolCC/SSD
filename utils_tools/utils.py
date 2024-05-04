import torch
import pickle
import torch
import torch.nn as nn
from torch.utils.data import dataset
from typing import Optional, List
from defense.temperature_scaling import TemperatureScaling

def write_to_file(results, addition):
    file = addition + '_dataset.pkl'
    with open(file, 'wb') as file:
        # 写入数据
        for result in results:
            for key in result.keys():
                value = result[key]                     
                result[key] = value.cpu().item() if torch.is_tensor(result[key]) else value
        pickle.dump(results, file)

def get_temp_calibrated_models(
    target_model: nn.Module,
    shadow_model: nn.Module,
    non_member_target: dataset,
    non_member_shadow: dataset,
    temp_value: Optional[float] = None
):
    target_model_temp_calibrated = TemperatureScaling(target_model)
    shadow_model_temp_calibrated = TemperatureScaling(shadow_model)
    if temp_value is None:
        target_model_temp_calibrated.calibrate(non_member_target)
        shadow_model_temp_calibrated.calibrate(non_member_shadow)
    else:
        target_model_temp_calibrated.temperature = temp_value
        shadow_model_temp_calibrated.temperature = temp_value

    return target_model_temp_calibrated, shadow_model_temp_calibrated
