import torch

def set_total_sample(target, total_sample):
    # set class count
    for i in range(target.size(0)):
        total_sample[target[i]] += 1


def set_total_right(target, prediction, total_right):
    pred_class = torch.max(prediction, 1)[1].view(target.size()).data
    for i in range(pred_class.shape[0]):
        if pred_class[i] == target.data[i]:
            total_right[pred_class[i]] += 1

