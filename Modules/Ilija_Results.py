import warnings

from sklearn.metrics import confusion_matrix


def results_dict_from_confusion_matrix_with_ratios(y_test, y_pred, y_train):
    """ Returns a dictionary of metrics for the actual and predicted test classes

    :param y_test: pandas.Series or np.ndarray
        actual classes for test samples
    :param y_pred: pandas.Series or np.ndarray
        predicted classes for test samples
    :param y_train: pandas.Series or np.ndarray
        train classes for calculating class balance mostly

    :return: dict
        dictionary of calculated metrics and sample counts
    """

    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

    support_dict = dict(y_test.value_counts())
    support_1 = support_dict.get(1)
    support_0 = support_dict.get(0)
    ratio_support = round(support_0 / support_1, 4)

    if y_train is not None:
        training_dict = dict(y_train.value_counts())
        train_1 = training_dict.get(1)
        train_0 = training_dict.get(0)
        ratio_training = round(train_0 / train_1, 4)
    else:
        warnings.warn("Pass on training y for balance ratio between classes")
        ratio_training = "None"
        train_1 = "None"
        train_0 = "None"

    if (FN == 0 and TP == 0) or (TN == 0 and FP == 0):
        return dict(
            Accuracy="Unanimous", f1_macro="Unanimous",
            f1_1="Unanimous", f1_0="Unanimous",
            Precision="Unanimous", Recall="Unanimous",
            NPV="Unanimous", Specificity="Unanimous",
            TP="Unanimous", FP="Unanimous", FN="Unanimous", TN="Unanimous",
            training_1="Unanimous", training_0="Unanimous", ratio_training="Unanimous",
            support_1="Unanimous", support_0="Unanimous", ratio_support="Unanimous",
        )

    ACC = round((TP + TN) / (TP + TN + FP + FN), 2)

    if (TP + FN) != 0:
        REC = round(TP / (TP + FN), 4)
    else:
        REC = "None"

    if (TN + FP) != 0:
        SPC = round(TN / (TN + FP), 4)
    else:
        SPC = "None"

    if (TP + FP) != 0:
        PREC = round(TP / (TP + FP), 4)
    else:
        PREC = "None"

    if (TN + FN) != 0:
        NPV = round(TN / (TN + FN), 4)
    else:
        NPV = "None"

    if REC == "None" or PREC == "None":
        F1_1 = "None"
    elif (REC + PREC) != 0:
        F1_1 = round((2 * REC * PREC) / (REC + PREC), 4)
    else:
        F1_1 = "None"

    if SPC == "None" or NPV == "None":
        F1_0 = "None"
    elif (SPC + NPV) != 0:
        F1_0 = round((2 * SPC * NPV) / (SPC + NPV), 4)
    else:
        F1_0 = "None"

    if F1_1 == "None" or F1_0 == "None":
        f1_macro = "None"
    else:
        f1_macro = round((F1_1 + F1_0) / 2, 4)

    return dict(
        Accuracy=ACC, f1_macro=f1_macro,
        f1_1=F1_1, f1_0=F1_0,
        Precision=PREC, Recall=REC,
        NPV=NPV, Specificity=SPC,
        TP=int(TP), FP=int(FP), FN=int(FN), TN=int(TN),
        training_1=train_1, training_0=train_0, ratio_training=ratio_training,
        support_1=support_1, support_0=support_0, ratio_support=ratio_support,
    )


def results_dict_from_confusion_matix (y_test, y_pred, y_train):
    """ Returns a dictionary of metrics for the actual and predicted test classes

    :param y_test: pandas.Series or np.ndarray
        actual classes for test samples
    :param y_pred: pandas.Series or np.ndarray
        predicted classes for test samples
    :param y_train: pandas.Series or np.ndarray
        train classes for calculating class balance mostly

    :return: dict
        dictionary of calculated metrics and sample counts
    """

    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

    support_dict = dict(y_test.value_counts())
    support_1 = support_dict.get(1)
    support_0 = support_dict.get(0)
    ratio_support = round(support_0 / support_1, 4)

    if y_train is not None:
        training_dict = dict(y_train.value_counts())
        train_1 = training_dict.get(1)
        train_0 = training_dict.get(0)
        ratio_training = round(train_0 / train_1, 4)
    else:
        warnings.warn("Pass on training y for balance ratio between classes")
        ratio_training = "None"
        train_1 = "None"
        train_0 = "None"

    if (FN == 0 and TP == 0) or (TN == 0 and FP == 0):
        return dict(
            Accuracy="Unanimous", f1_macro="Unanimous",
            f1_1="Unanimous", f1_0="Unanimous",
            Precision="Unanimous", Recall="Unanimous",
            NPV="Unanimous", Specificity="Unanimous",
            TP="Unanimous", FP="Unanimous", FN="Unanimous", TN="Unanimous",
            training_1="Unanimous", training_0="Unanimous", ratio_training="Unanimous",
            support_1="Unanimous", support_0="Unanimous", ratio_support="Unanimous",
        )

    ACC = round((TP + TN) / (TP + TN + FP + FN), 2)

    if (TP + FN) != 0:
        REC = round(TP / (TP + FN), 4)
    else:
        REC = "None"

    if (TN + FP) != 0:
        SPC = round(TN / (TN + FP), 4)
    else:
        SPC = "None"

    if (TP + FP) != 0:
        PREC = round(TP / (TP + FP), 4)
    else:
        PREC = "None"

    if (TN + FN) != 0:
        NPV = round(TN / (TN + FN), 4)
    else:
        NPV = "None"

    if REC == "None" or PREC == "None":
        F1_1 = "None"
    elif (REC + PREC) != 0:
        F1_1 = round((2 * REC * PREC) / (REC + PREC), 4)
    else:
        F1_1 = "None"

    if SPC == "None" or NPV == "None":
        F1_0 = "None"
    elif (SPC + NPV) != 0:
        F1_0 = round((2 * SPC * NPV) / (SPC + NPV), 4)
    else:
        F1_0 = "None"

    if F1_1 == "None" or F1_0 == "None":
        f1_macro = "None"
    else:
        f1_macro = round((F1_1 + F1_0) / 2, 4)

    return dict(
        Accuracy=ACC, f1_macro=f1_macro,
        f1_1=F1_1, f1_0=F1_0,
        Precision=PREC, Recall=REC,
        NPV=NPV, Specificity=SPC,
        TP=int(TP), FP=int(FP), FN=int(FN), TN=int(TN),
        training_1=train_1, training_0=train_0, ratio_training=ratio_training,
        support_1=support_1, support_0=support_0, ratio_support=ratio_support,
    )