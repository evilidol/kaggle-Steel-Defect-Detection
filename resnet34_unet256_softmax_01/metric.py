from common  import *

from sklearn import metrics as sklearn_metrics


##################################################################################################3

def remove_small_one(predict, min_size):
    H,W = predict.shape
    num_component, component = cv2.connectedComponents(predict.astype(np.uint8))
    predict = np.zeros((H,W), np.bool)
    for c in range(1,num_component):
        p = (component==c)
        if p.sum()>min_size:
            predict[p] = True
    return predict

def remove_small(predict, min_size):
    for b in range(len(predict)):
        for c in range(4):
            predict[b,c] = remove_small_one(predict[b,c], min_size[c])
    return predict

##################################################################################################3
def compute_eer(fpr,tpr,threshold):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    fnr = 1-tpr
    abs_diff  = np.abs(fpr-fnr)
    min_index = np.argmin(abs_diff)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return eer, threshold[min_index]

def compute_metric_label(truth_label, predict_label):
    t = truth_label.reshape(-1,4)
    p = predict_label.reshape(-1,4)

    #           num_truth, num_predict, correct, recall, precision
    # (all) neg
    # (all) pos
    #      neg1
    #      pos1
    #     ...
    ts = np.array([
        [1-t.reshape(-1),t.reshape(-1)],
        [1-t[:,0],t[:,0]],
        [1-t[:,1],t[:,1]],
        [1-t[:,2],t[:,2]],
        [1-t[:,3],t[:,3]],
    ]).reshape(-1)
    ps = np.array([
        [1-p.reshape(-1),p.reshape(-1)],
        [1-p[:,0],p[:,0]],
        [1-p[:,1],p[:,1]],
        [1-p[:,2],p[:,2]],
        [1-p[:,3],p[:,3]],
    ]).reshape(-1)

    result = []
    for tt,pp in zip(ts, ps):
        num_truth   = tt.sum()
        num_predict = pp.sum()
        num_correct = (tt*pp).sum()
        recall      = num_correct/num_truth
        precision   = num_correct/num_predict
        result.append([num_truth, num_predict, num_correct, recall, precision])


    ## from kaggle probing ...
    kaggle_pos = np.array([ 128,43,741,120 ])
    kaggle_neg_all = 6172
    kaggle_all     = 1801*4

    recall_neg_all = result[0][3]
    recall_pos = np.array([
        result[3][3],
        result[5][3],
        result[7][3],
        result[9][3],
    ])

    kaggle = []
    for dice_pos in [1.00, 0.75, 0.50]:
        k = (recall_neg_all*kaggle_neg_all + sum(dice_pos*recall_pos*kaggle_pos))/kaggle_all
        kaggle.append([k,dice_pos])

    return kaggle, result

def print_metric_label(kaggle, result):
    text  = ''
    text += '* image level metric *\n'
    text += '             num_truth, num_predict,  num_correct,   recall, precision\n'
    text += ' (all) neg       %4d        %4d           %4d       %0.2f    %0.2f  \n'%(*result[0],)
    text += ' (all) pos       %4d        %4d           %4d       %0.2f    %0.2f  \n'%(*result[1],)
    text += '\n'
    text += '       neg1      %4d        %4d           %4d       %0.2f    %0.2f  \n'%(*result[2],)
    text += '       pos1      %4d        %4d           %4d       %0.2f    %0.2f  \n'%(*result[3],)
    text += '\n'
    text += '       neg2      %4d        %4d           %4d       %0.2f    %0.2f  \n'%(*result[4],)
    text += '       pos2      %4d        %4d           %4d       %0.2f    %0.2f  \n'%(*result[5],)
    text += '\n'
    text += '       neg3      %4d        %4d           %4d       %0.2f    %0.2f  \n'%(*result[6],)
    text += '       pos3      %4d        %4d           %4d       %0.2f    %0.2f  \n'%(*result[7],)
    text += '\n'
    text += '       neg4      %4d        %4d           %4d       %0.2f    %0.2f  \n'%(*result[8],)
    text += '       pos4      %4d        %4d           %4d       %0.2f    %0.2f  \n'%(*result[9],)
    text += '\n'

    text += 'kaggle = %0.5f @ dice%0.3f\n'%(kaggle[0][0],kaggle[0][1])
    text += '       = %0.5f @ dice%0.3f\n'%(kaggle[1][0],kaggle[1][1])
    text += '       = %0.5f @ dice%0.3f\n'%(kaggle[2][0],kaggle[2][1])
    text += '\n'

    return text

def compute_roc_label(truth_label, probability_label):
    t = truth_label.reshape(-1,4)
    p = probability_label.reshape(-1,4)

    auc=[]
    result=[]
    for c in [0,1,2,3]:
        a = sklearn_metrics.roc_auc_score(t[:,c], p[:,c])
        auc.append(a)

        fpr, tpr, threshold = sklearn_metrics.roc_curve(t[:,c], p[:,c])
        result.append([fpr, tpr, threshold])

    return auc, result

def print_roc_label(auc, result):
    text  = ''
    text += '%s\n'%(str(auc))
    text +='\n'
    for c,(fpr, tpr, threshold) in enumerate(result):
        text += 'class%d\n'%(c+1)
        text += 'bin\tfpr\ttpr\n'
        for f,t,b in zip(fpr, tpr, threshold):
            text += '%0.3f\t%0.3f\t%0.3f\n'%(b,f,t)
        text +='\n'


    return text


#---







##################################################################################################3
