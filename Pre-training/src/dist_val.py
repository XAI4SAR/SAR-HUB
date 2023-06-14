import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from torch import sigmoid
def image_val_TSX_OSU(dataloader, device,model,world_size, training = False):
    model.eval()
    # correct_samples = 0
    acc_num = 0.0
    data_num = 0.0
    val_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in dataloader:
            inputs = data['image'].to(device)
            labels = data['label'].to(device)
            # print(labels)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predict_top = torch.max(outputs,1)
            acc_num += torch.sum(torch.squeeze(predict_top) == labels.data).float()
            data_num += labels.size()[0]
            val_loss += loss
    if device != torch.device("cpu"):
                torch.cuda.synchronize(device)
    sum_num = reduce_value(acc_num, world_size, average=True)
    val_loss = reduce_value(val_loss, world_size, average=True)
    top1_acc = sum_num / data_num
    val_loss /= len(dataloader)
    # print('YES',top1_acc,val_loss)
    return top1_acc, val_loss
def reduce_value(value, world_size, average=True):
    world_size = world_size
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value) # 求和
        if average:  # 取平均
            value /= world_size

        return value
def accuracy(gt,pred):       
    gt = np.asarray(gt) #will round to the nearest even number
    pred = np.round(pred)
    return{
    'f1_samples': f1_score(gt, pred, average = 'samples', zero_division=1),
    'f1_macro': f1_score(gt, pred, average = 'macro', zero_division=1),
    'f1_micro': f1_score(gt,pred,average = 'micro', zero_division=1)}
    
def image_val_BEN(dataloader, device,model,world_size,training = False):
    model.eval()
    test_iter=0
    # correct_samples = 0
    acc_num = 0.0
    data_num = 0.0
    val_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    # criterion = Affinity_Loss(lambd = 0.1)
    # with torch.no_grad():
    with torch.no_grad():
        for data in dataloader:
            inputs = data['image'].to(device)
            labels = data['label'].to(device)
            # print(labels)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            pred = sigmoid(outputs).data.cpu().numpy()
            gt = labels.data.cpu().numpy()
            if test_iter==0:
                all_pred=pred
                all_gt=gt
            else:
                all_pred=np.vstack((all_pred,pred))
                all_gt  =np.vstack((all_gt,gt))

            test_iter+=1
            acc=accuracy(all_gt,all_pred)
            predicted_probs = np.asarray(all_pred)
            y_predicted = (predicted_probs >= 0.5).astype(np.float32)
            y_true = np.asarray(all_gt)
            prec = precision_score(y_true, y_predicted, average='micro')
            recal = recall_score(y_true, y_predicted, average='micro')
            data_num += labels.size()[0]
            val_loss += loss
    if device != torch.device("cpu"):
                torch.cuda.synchronize(device)

    val_loss = reduce_value(val_loss, world_size, average=True)

    val_loss /= len(dataloader)

    return acc, prec, recal, val_loss