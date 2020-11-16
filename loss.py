# Project hiatus
# loss
# file with all the losses used
# 13/11/2020
# Cédric BARON

import torch.nn as nn
import torch



def MeanSquareError(y_out, y, squared=True):
    """
    args: two pytorch tensors, one for the prediction and other for the target
    fun: returns a mean squared error as tensor
    """
    if squared:
        # computing loss on the matrixes
        loss_matrix = (y_out - y)**2
        
        # computing the mean of the matrix
        loss = torch.mean(loss_matrix)
    
    else:
        loss_matrix = torch.abs(y_out - y)
        loss = torch.mean(loss_matrix)
        
    return loss


def CrossEntropy(y_out, y):
    """
    args: two pytorch tensors, one for the prediction and other for the target
    fun: returns cross entropy loss as a tensor
    """
    
    criterion = nn.CrossEntropyLoss(reduction="none")
    loss = criterion(y_out, y)
    loss = loss.mean()
    
    return loss