
# coding: utf-8

# This code is to designed to create a visualization of confsuion matrix displaying multiclass 
# for this you need to copy the numpy array of prediction and actual class label and paste it where the array are displayed
# 
# for more automated approach the results of predictions and actual label can be captured by dataframe using pandas 
# by doing this we do not need to copy the labels and prediction manually...

# In[1]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]

y_actu = [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,
        3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,
        5,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,
        6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,
        8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11,
       11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13,
       13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15,
       15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
       17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18,
       18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20,
       20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22,
       22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23,
       23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25,
       25, 25, 25, 25, 25]

y_pred = [20,  0,  0,  0, 20,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  2,  8,  2,  2, 24,  2,  2,  2,  2,  2,  3,  3,  3,  3,
        3,  3,  3,  3,  3,  3,  4, 14, 16,  4, 14,  4,  4,  4,  4,  4,  1,
        1,  1,  5,  5,  1,  5,  5,  5,  1,  6, 21,  6,  6,  6, 11, 11,  2,
        2, 11,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,
        8,  8,  8,  8,  8,  1,  1,  1,  1,  1, 14,  9,  9,  1,  9, 22,  8,
        8, 12, 12, 22, 22, 22, 10, 10,  6,  6, 19, 11, 11,  2,  2,  2,  2,
       11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 15, 11, 11, 13, 13, 13,
       13, 15, 13, 13, 14,  4, 14,  4,  4, 14, 14, 14, 16, 14, 15, 13, 15,
       15, 15, 15, 13, 15, 15, 13, 16, 16, 16, 16, 16,  9, 16, 16, 16, 16,
       17, 17, 17, 17, 17, 17, 17, 17, 17,  3, 18, 18, 18, 18,  1, 18, 18,
       18, 18, 18, 13, 13, 13, 13, 13, 19, 19, 19, 19, 19, 20, 20, 20, 20,
       20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 13, 13, 13, 13, 21, 22,
       22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 22, 23, 23,
       23, 23, 24, 24,  2,  2, 24, 11, 11, 11, 11, 15, 25, 25, 25, 25, 25,
       25, 25, 25, 25, 25]

cm=confusion_matrix(y_actu, y_pred)


# In[2]:


import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
   
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(25, 16))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# In[21]:


plot_confusion_matrix(cm,labels)


# In[26]:


import matplotlib.pyplot as plt
plt.savefig('foo.png')

