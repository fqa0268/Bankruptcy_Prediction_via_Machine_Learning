import numpy as np
import pandas as pd
from sklearn import metrics
import itertools
from sklearn.metrics import recall_score, precision_score , accuracy_score
from sklearn.model_selection import cross_val_predict, cross_val_score
import matplotlib.pyplot as plt

class Helper():
    def __init__(self):
        pass

    def plot_confusion_matrix(self, cm, classes):
        """
        This function prints and plots the confusion matrix.
        """

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            # Plot coordinate system has origin in upper left corner
            # -  coordinates are (horizontal offset, vertical offset)
            # -  so cm[i,j] should appear in plot coordinate (j,i)
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
    
    def Evaluate_Model(self, name, clf, X_train, y_train, k = 3, display = True):
        """
        Return 3 scorses (Accuracy, Recall, Precision), and optionally display the scores and
        confusion matrix.
        """
        cross_val_scores = cross_val_score(clf, X_train, y_train, cv=k)
        y_pred = cross_val_predict(clf, X_train, y_train, cv=k, method="predict")
        accuracy_s = accuracy_score(y_train, y_pred)
        recall_s = recall_score(y_train, y_pred, pos_label=1, average="binary")
        precision_s = precision_score(y_train, y_pred, pos_label=1, average="binary")
        if display:
            print("Model: {m:s} avg cross validation score={s:3.2f}\n".format(m=name, s=cross_val_scores.mean()))
            print('Accuracy Score={s:3.2f}'.format(s=accuracy_s))
            print('Recall Score={s:3.2f}'.format(s=recall_s))
            print('Precision_Score={s:3.2f}'.format(s=precision_s))

            confusion_mat = metrics.confusion_matrix(y_train, y_pred)
            self.plot_confusion_matrix(confusion_mat, [0,1])
        return accuracy_s, recall_s, precision_s

    def record_in_summary(self, summary_table, name, accuracy_s, recall_s, precision_s):
        """
        Record a result in a table for convenient comparison.
        """
        row = pd.DataFrame({'Model':name, 'Accuracy':accuracy_s,
                            'Recall':recall_s, 'Precision':precision_s}, index = [0])
        summary_table = pd.concat([summary_table, row], ignore_index = True)
        return summary_table
