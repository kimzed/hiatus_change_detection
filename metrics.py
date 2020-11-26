# Project hiatus
# script with metrics
# 26/11/2020
# Cédric BARON

# importing libraries
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve
from sklearn import metrics


class ConfusionMatrixBinary:
  def __init__(self, n_class, class_names):
    self.CM = np.zeros((n_class, n_class))
    self.n_class = n_class
    self.class_names = class_names
  
  def clear(self):
    self.CM = np.zeros((self.n_class, self.n_class))
    
  def add_batch(self, gt, pred):
    self.CM +=  confusion_matrix(gt, pred, labels = list(range(self.n_class)))
    
  def overall_accuracy(self):#percentage of correct classification
    return np.trace(self.CM) / np.sum(self.CM)

  def class_IoU(self, show = 1):
    ious = np.full(self.n_class, 0.)
    for i_class in range(self.n_class):
      error_matrix = [i for i in range(self.n_class) if i != i_class]
      ious[i_class] = self.CM[i_class, i_class] / (np.sum(self.CM[i_class, error_matrix]) + np.sum(self.CM[error_matrix, i_class]) + self.CM[i_class, i_class])
    if show:
      print('  |  '.join('{} : {:3.2f}%'.format(name, 100*iou) for name, iou in zip(self.class_names,ious)))
    #do not count classes that are not present in the dataset in the mean IoU
    return 100*np.nansum(ious) / (np.logical_not(np.isnan(ious))).sum()


def visualize_roc(y, pred, return_thresh = False):
    
    ## making the ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    # plot the roc curve for the model
    pyplot.plot(fpr, tpr, linestyle='--', label='Prediction')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()
    
    ## returning thresholds
    result = None
    
    if return_thresh:
        result = thresholds
    
    return result


def confusion_matrix_visualize(pred, y, thresh):
    """
    Computes the confusion matrix for binary values
    
    """
    
    # loading the confusion matrix 
    m = ConfusionMatrixBinary(2, ["no change", "change"])
    
    # putting into the confusion matrix
    m.add_batch(y, pred)
            
    # printing the result for one given threshold
    print("Threshold is "+str(thresh))
    print(m.CM)
    print('IoU : {:3.2f}%'.format(m.class_IoU()))
    print('Overall accuracy : {:3.2f}%'.format(m.overall_accuracy()*100))
    print('\n')
    
    return None

