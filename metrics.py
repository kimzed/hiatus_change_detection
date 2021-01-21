# Project hiatus
# script with metrics
# 26/11/2020
# Cédric BARON

# importing libraries
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
from sklearn.neighbors import KDTree
from scipy.special import digamma
import scipy.spatial as spatial
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import torch
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn import preprocessing

# importing functions from other files
import utils as fun


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
    """
    Function to perform AUC calculation, plots a ROC curve as well
    Can output the thresholds used
    """
    ## making the ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    auc = metrics.roc_auc_score(y, pred)
     # calculating the optimal threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    idx = np.argmax(gmeans)
    optimal_threshold = thresholds[idx]
    
    # plot the roc curve for the model
    pyplot.plot(fpr, tpr, linestyle='--', label='AUC: %1.2f opt_thresh: %2.2f' % ((auc), (optimal_threshold)))
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


def class_precision(binary_vec, y, classes):
    """
    Function to evaluate the precision per class for change/no change
    args are the bianary vect (predictions), the binary ground truth and the classes
    
    """
    # getting boolean vectors
    false_values = binary_vec != y
    true_values = binary_vec == y
    
    # converting to numpy
    classes = np.array(classes)
    
    ## getting the number of true values for every class
    true1 = classes[true_values]
    true1 = np.count_nonzero(true1 == 1)
    true2 = classes[true_values]
    true2 = np.count_nonzero(true2 == 2)
    true3 = classes[true_values]
    true3 = np.count_nonzero(true3 == 3)
    
    ## getting the number of false values for every class
    false1 = classes[false_values]
    false1 = np.count_nonzero(false1 == 1)
    false2 = classes[false_values]
    false2 = np.count_nonzero(false2 == 2)
    false3 = classes[false_values]
    false3 = np.count_nonzero(false3 == 3)
    
    # getting the percentage of correctly predicted values
    precision1 = true1 / (true1 + false1)
    precision2 = true2 / (true2 + false2)
    precision3 = true3 / (true3 + false3)
    
    # printing the result
    print("Precision for class one is {:3.2f} ".format(precision1))
    print("Precision for class two is {:3.2f} ".format(precision2))
    print("Precision for class three is {:3.2f} ".format(precision3))
    print("\n")
    
    return None


def NMI_continuous_discrete(labels_discrete, data_continuous, nb_classes, labels, classes_idx):
    """
    Function to compute the normalised mutual information
    """
    
    # number of samples
    N = len(labels_discrete)
    
    # loading the kd tree
    tree = spatial.cKDTree(data_continuous) 
    
    # variable to store the score
    MI = 0
    
    # number of neighbours (actually k-1)
    k = 5
    
    for i in range(len(labels)):
        
        # loading the number of pixels from this class
        Nxi = nb_classes[i]
        
        # loading the class
        label_class = labels[i]  
        
        # index to get class continuous data
        idx_class = 0
        
        # loading the index matrix for the class
        class_idx = classes_idx[i]
        
        # loading the values corresponding to the class
        data_class = data_continuous[class_idx]
        
        # loading the tree for this class
        tree_class = KDTree(data_class)
        
        # looping through our data
        for i in range(len(labels_discrete)):
            
            # checking if the sample has the correct class
            if labels_discrete[i] == label_class:
                
                # getting the distance for the nearest neighbours
                dist, ind = tree_class.query(data_class[idx_class][None,:], k=k)
                
                # getting max distance
                dist_max = np.max(dist)
                
                # getting the number of samples within the distance
                ind = tree.query_ball_point(data_continuous[i], dist_max)
                Mi = len(ind)
                
                # updating index for data_class 
                idx_class += 1
                
                # calculating the MI
                MI += digamma(N) - digamma(Nxi) + digamma(k-1) - digamma(Mi)
                
    # averaging to get the NMI
    NMI_avg = MI / N
    
    return NMI_avg


def svm_accuracy_estimation(data, labels, cv=False):
    
    ## linear svm with the mns
    # loading the data
    dataset = fun.train_val_dataset(data, labels)
    tensor_train = torch.tensor(dataset['train'])
    tensor_val = torch.tensor(dataset['val'])
    tensor_gt_val = torch.tensor(dataset['gt_val'])
    
    
    if len(list(tensor_train.shape)) == 1:
        
        tensor_train = torch.tensor(dataset['train'])[:,None]
        tensor_val = torch.tensor(dataset['val'])[:,None]
        tensor_gt_val = torch.tensor(dataset['gt_val'])[:,None]
    
    scaler = preprocessing.StandardScaler()
    
    tensor_train = scaler.fit_transform(tensor_train)
    tensor_val = scaler.fit_transform(tensor_val)
    
    # loading the model, ovo is one against all, C is the soft margin
    svclassifier = SVC(kernel='linear', decision_function_shape='ovr', C=0.01,
                       class_weight="balanced")
    
    # training the model
    svclassifier.fit(tensor_train, dataset['gt_train'])
    
    # predicting the labels
    pred_label = svclassifier.predict(tensor_val)
    
    # printing  results
    conf_mat = confusion_matrix(tensor_gt_val, pred_label)
    class_report = classification_report(tensor_gt_val, pred_label)
    
    # performing a cross validation (optional)
    if cv:
        # prepare the cross-validation procedure
        cv = KFold(n_splits=10, random_state=1, shuffle=True)
        
        # performing a k fold validation
        scores_cv = cross_val_score(svclassifier, tensor_val, tensor_gt_val,
                                cv=cv, scoring='f1_macro')
    else:
        scores_cv=None
    
    return conf_mat, class_report, scores_cv




def svm_accuracy_estimation_2(data_train, data_test, labels_train, labels_test, cv=False):
    
    ## linear svm with the mns
    # loading the data
    tensor_train = torch.tensor(data_train)
    tensor_val = torch.tensor(data_test)
    tensor_gt_val = torch.tensor(labels_test)
    
    if len(list(tensor_train.shape)) == 1:
        
        tensor_train = torch.tensor(data_train)[:,None]
        tensor_val = torch.tensor(data_test)[:,None]
        tensor_gt_val = torch.tensor(labels_test)[:,None]
    
    scaler = preprocessing.StandardScaler()
    
    tensor_train = scaler.fit_transform(tensor_train)
    tensor_val = scaler.fit_transform(tensor_val)
    
    # loading the model, ovo is one against all, C is the soft margin
    svclassifier = SVC(kernel='linear', decision_function_shape='ovr', C=0.01,
                       class_weight="balanced")
    
    # training the model
    svclassifier.fit(tensor_train, labels_train)
    
    # predicting the labels
    pred_label = svclassifier.predict(tensor_val)
    
    # printing  results
    conf_mat = confusion_matrix(tensor_gt_val, pred_label)
    class_report = classification_report(tensor_gt_val, pred_label)
    
    # performing a cross validation (optional)
    if cv:
        # prepare the cross-validation procedure
        cv = KFold(n_splits=10, random_state=1, shuffle=True)
        
        # performing a k fold validation
        scores_cv = cross_val_score(svclassifier, tensor_val, tensor_gt_val,
                                cv=cv, scoring='f1_macro')
    else:
        scores_cv=None
    
    return conf_mat, class_report, scores_cv

