from sklearn import metrics


def pred_metrics(labels_test, labels_pred):
    labels_test = labels_test.astype(int)
    labels_pred[labels_pred > 0.5] = 1
    labels_pred[labels_pred <= 0.5] = 0
    labels_pred = labels_pred.astype(int)
    mtrcs = {}
    mtrcs['confusion'] = metrics.confusion_matrix(labels_test, labels_pred)
    mtrcs['accuracy'] = metrics.accuracy_score(labels_test, labels_pred)
    mtrcs['roc_auc'] = metrics.roc_auc_score(labels_test, labels_pred)
    mtrcs['f1_score'] = metrics.f1_score(labels_test, labels_pred)
    return mtrcs
