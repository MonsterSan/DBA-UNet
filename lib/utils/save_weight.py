import os
import torch


def save_weights(losses, miou, min_loss, best_miou, state_dict, weight_path):
    if losses.avg < min_loss:
        min_loss = losses.avg
        torch.save(state_dict, os.path.join(weight_path, 'lowest_loss.pth'))
    if miou > best_miou:
        best_miou = miou
        torch.save(state_dict, os.path.join(weight_path, 'best_miou.pth'))
    return min_loss, best_miou
