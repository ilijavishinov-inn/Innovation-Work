from sklearn.model_selection import GroupShuffleSplit

def find_best_group_split(dataframe, target_feature, group_by_feature, balance_focus="train"):
    """
    :param dataframe: pandas.Dataframe
        dataframe to split for max balance
    :param target_feature: string
        name of the target feature of the dataset
    :param group_by_feature: string
        name of the feature on which to group sets, preventing data_1-24h leakage when needed
    :param balance_focus: string
        {'train', 'test'}

    :return: pandas.Dataframe, pandas.Dataframe, float, float
         returns train, test,
    """

    min_train_diff = 1
    min_test_diff = 1
    min_train_indices = list()
    min_test_indices = list()
    # using GroupShuffleSplit to generate 10 splits (the golden rule) and find the best split for our goal
    for train_indices, test_indices in GroupShuffleSplit(test_size=0.20,
                                                         n_splits=10,
                                                         random_state=42
                                                         ).split(dataframe.drop(target_feature, axis=1),
                                                                 dataframe[target_feature],
                                                                 groups=dataframe[group_by_feature]):

        train, test = dataframe.iloc[train_indices].copy(), dataframe.iloc[test_indices].copy()

        #
        vc_train = dict(train[target_feature].value_counts())
        n_train = vc_train.get(0) + vc_train.get(1)
        zero_train, target_train = vc_train.get(0) / n_train, vc_train.get(1) / n_train

        vc_test = dict(test[target_feature].value_counts())
        n_test = vc_test.get(0) + vc_test.get(1)
        zero_test, target_test = vc_test.get(0) / n_test, vc_test.get(1) / n_test

        if len(min_train_indices) == 0 and len(min_test_indices) == 0:
            min_train_diff = abs(zero_train - target_train)
            min_test_diff = abs(zero_test - target_test)
            min_train_indices = train_indices
            min_test_indices = test_indices
        elif balance_focus == 'train' and abs(zero_train - target_train) < min_train_diff:
            min_train_diff = abs(zero_train - target_train)
            min_test_diff = abs(zero_test - target_test)
            min_train_indices = train_indices
            min_test_indices = test_indices
        elif balance_focus == 'test' and abs(zero_test - target_test) < min_test_diff:
            min_train_diff = abs(zero_train - target_train)
            min_test_diff = abs(zero_test - target_test)
            min_train_indices = train_indices
            min_test_indices = test_indices

    train_best, test_best = dataframe.iloc[min_train_indices].copy(), dataframe.iloc[min_test_indices].copy()

    return train_best, test_best, min_train_diff, min_test_diff