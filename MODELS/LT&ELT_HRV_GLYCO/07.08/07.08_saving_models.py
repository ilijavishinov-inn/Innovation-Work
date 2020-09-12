globals().clear()
import sys
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import confusion_matrix
from imblearn.pipeline import Pipeline as imblearn_Pipeline
import json
import warnings
from datetime import datetime
from imblearn.under_sampling import ClusterCentroids
import pickle

######################################################################################################################################################
num_iter = 1

data_directory_id = '1-24h'
model_description = f'scaling_before_pca_i{num_iter}'
results_explanatory_suffix = f'{model_description}_{str(datetime.now().date())}_{str(datetime.now().hour)}h{str(datetime.now().minute)}m'

# JUPYTER NOTEBOOK
path = ''

# PYCHARM
# path = '/MODELS/07.08/'

data_directory = f"{path}data_{data_directory_id}"
results_directory = f"{path}results_data_{data_directory_id}_{results_explanatory_suffix}"
results_write_path = f'{results_directory}/{data_directory_id}_{results_explanatory_suffix}.xlsx'


# # GOOGLE COLLABORATORY
# from google.colab import drive
#
# current_date = '06.08'
# drive.mount('/content/gdrive')
# data_directory = f'/content/gdrive/My Drive/Colab Notebooks/{current_date}/data_{data_directory_id}'
# results_directory = f'/content/gdrive/My Drive/Colab Notebooks/{current_date}/results_data_{data_directory_id}_{results_explanatory_suffix}'
# results_write_path = f'{results_directory}/results_data_{data_directory_id}_{results_explanatory_suffix}.xlsx'
# collab_flag = 1

######################################################################################################################################################

if not os.path.exists(results_directory):
    os.makedirs(results_directory)
else:
    answer = str(input("The directory already exists. Are you sure you want to continue? [y/n]"))
    if answer == 'y':
        pass
    else:
        sys.exit()

results = list()
workflow_dict = dict()

classifiers = ['lr', 'rfc']
# classifiers = ["svc", "lr", "gpc", "rfc", "gbc", 'abc']
datasets = ['td', 'td_nl']


def results_dict_from_confusion_matrix(_y_test, _y_pred, _y_train):
    """ Returns a dictionary of metrics for the actual and predicted test classes

    :param _y_test: pandas.Series or np.ndarray
        actual classes for test samples
    :param _y_pred: pandas.Series or np.ndarray
        predicted classes for test samples
    :param _y_train: pandas.Series or np.ndarray
        train classes for calculating class balance mostly

    :return: dict
        dictionary of calculated metrics and sample counts
    """

    TN, FP, FN, TP = confusion_matrix(_y_test, _y_pred).ravel()

    support_dict = dict(_y_test.value_counts())
    support_1 = support_dict.get(1)
    support_0 = support_dict.get(0)
    ratio_support = round(support_0 / support_1, 4)

    if _y_train is not None:
        training_dict = dict(_y_train.value_counts())
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


def partial_polynomial_transform(dataframe, feature_names=None, degree=2):
    """Transform a dataframe such that all polynomial combinations of custom selected features of less than or equal
    custom degree are included in the dataframe.

    :param dataframe: pandas.DataFrame
        Dataframe to be transformed
    :param feature_names: list of strings
        Features (list of column names) to be polynomialy expanded
    :param degree: int, default = 2
        Largest degree of the resulting polynomials

    :return: padnas.DataFrame
    """

    # if no feature names are passed on, assign a list of all the column names in the frame
    if feature_names is None or not feature_names:
        feature_names = dataframe.columns

    # if _feature_names is a string, i.e. only one element, there's no point in executing transformation
    if isinstance(feature_names, str):
        return dataframe

    # features which are not transformed and concatenated at the end
    holdout_features = [x for x in dataframe.columns if x not in feature_names]

    if holdout_features:
        holdout_dataframe = dataframe[holdout_features]
    # else case is for the compiler, an empty list for holdout_dataframe will not be used
    else:
        holdout_dataframe = []

    # polynomial transformation
    prepolynomial_dataframe = dataframe[feature_names]
    polynomial_transformer = PolynomialFeatures(degree=degree)
    polynomial_ndarray = polynomial_transformer.fit_transform(prepolynomial_dataframe)
    polynomial_feature_names = polynomial_transformer.get_feature_names(input_features=feature_names)

    polynomial_dataframe = pd.DataFrame(data=polynomial_ndarray,
                                        columns=polynomial_feature_names)

    # recombine the holdout and the transformed frames
    if holdout_features:
        transformed_dataframe = pd.concat([polynomial_dataframe, holdout_dataframe], axis=1)
    else:
        transformed_dataframe = polynomial_dataframe

    return transformed_dataframe


def remove_outliers(dataframe, feature_names=None):
    """ Remove outliers from a dataframe. Outliers are considered data_1-24h points which deviate more than 3 sigmas from the mean
    after computing the z-score.

    :param dataframe: pandas.DataFrame
        Dataframe from which outliers are removed
    :param feature_names: list of strings, default="all"
        Cirteria features (list of column names) on which the outlier removal is based

    :return: pandas.DataFrame
    """

    if feature_names is None or not feature_names:
        feature_names = dataframe.columns

    frame = dataframe[feature_names]

    z_scores = stats.zscore(frame)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    dataframe_with_removed_outliers = dataframe[filtered_entries]

    return dataframe_with_removed_outliers, abs_z_scores


def add_transformed_features(dataframe, feature_names=None,
                             log_transform=False,
                             boxcox_transform=False,
                             root_transform=False):
    """ Add transformed features to a dataframe

    :param dataframe: pandas.DataFrame
        Input dataframe
    :param feature_names: list of strings or string
        features to be transformed to the
    :param log_transform: bool
    :param boxcox_transform: bool
    :param root_transform: bool

    :return: transformed dataframe
    """
    if feature_names is None:
        feature_names = dataframe.columns

    if isinstance(feature_names, str):
        feature_names = list(feature_names)

    for feature_name in feature_names:

        if log_transform:
            log_transformed = np.log1p(dataframe[feature_name])
            dataframe[feature_name + "_log"] = log_transformed

        if root_transform:
            root_transformed = np.sqrt(dataframe[feature_name])
            dataframe[feature_name + "_root"] = root_transformed

        if boxcox_transform:
            box_cox_transformed, _ = stats.boxcox(dataframe[feature_name])
            dataframe[feature_name + "_boxcox"] = box_cox_transformed

    return dataframe


# day = str(datetime.now().day) if datetime.now().day > 9 else f'0{datetime.now().day}'
# month = str(datetime.now().month) if datetime.now().month > 9 else f'0{datetime.now().month}'
# current_date = f'{day}.{month}'

file_names = [f'hrv_{x}h.csv' for x in [24, 12, 10, 8, 6, 4, 3, 2, 1]]

for file_name in file_names:

    for dataset in datasets:

        file_path = data_directory + "/" + file_name

        records_initial = pd.read_csv(file_path)

        records_initial = records_initial[records_initial["HbA1C(%)"].notnull()]
        records_initial.drop(["Dataset Name", "Interval"], axis=1, inplace=True)

        hba1c = records_initial["HbA1C(%)"]
        records_initial.drop("HbA1C(%)", axis=1, inplace=True)

        records_initial.drop(["ULF", "VLF", "LF", "HF", "LF/HF"], axis=1, inplace=True)
        if dataset == "td":
            records_initial.drop(["SD1", "SD2", "SD1/SD2"], axis=1, inplace=True)

        # Encoding End feature
        records_initial.replace(inplace=True,
                                to_replace={"End": {1: "night", 2: "night", 3: "night", 4: "night",
                                                    5: "night", 6: "night", 7: "morning", 8: "morning",
                                                    9: "morning", 10: "morning", 11: "morning", 12: "morning",
                                                    13: "afternoon", 14: "afternoon", 15: "afternoon", 16: "afternoon",
                                                    17: "afternoon", 18: "afternoon", 19: "evening", 20: "evening",
                                                    21: "evening", 22: "evening", 23: "evening", 24: "evening",
                                                    0: "evening"}}
                                )
        # >>
        workflow_dict["Encode End"] = f'night, morning, afternoon, evening'

        records_initial.replace(inplace=True,
                                to_replace={"Regulation_Class": {"B": 1,
                                                                 "G": 0}}
                                )

        # Removing null values
        num_rows_before_drop_null = records_initial.shape[0]
        records_initial.dropna(axis=0, inplace=True)
        num_rows_removed_null = num_rows_before_drop_null - records_initial.shape[0]
        # >>
        workflow_dict["Missing Values"] = f'All {num_rows_removed_null} rows with missing values are removed'

        # Outlier detection
        outlier_columns = [x for x in records_initial.columns if x not in ["Patient_ID", "End", "Regulation_Class"]]

        records_positive = records_initial.copy()
        for outlier_column in outlier_columns:
            records_positive = records_positive[records_positive[outlier_column] > 0]

        num_rows_before_outlier_removal = records_positive.shape[0]
        records_no_outliers, abs_z_scores_debug = remove_outliers(records_positive, outlier_columns)
        num_rows_removed_outliers = num_rows_before_outlier_removal - records_no_outliers.shape[0]
        # >>
        workflow_dict["Outliers"] = f'{num_rows_removed_outliers} rows with outliers are removed'

        # Transforming categorical features to numerical
        records_encoded = pd.get_dummies(records_no_outliers, columns=["End"])

        # Adding transformed features
        records_encoded = add_transformed_features(records_encoded, outlier_columns,
                                                   log_transform=True,
                                                   boxcox_transform=True,
                                                   root_transform=True
                                                   )
        # >>
        workflow_dict["Transformations"] = f'Log, Sqrt, BoxCox'

        # Splitting dataset into train and test
        split_balancing = 'test'
        train, test, min_train_diff, min_test_diff = find_best_group_split(dataframe=records_encoded,
                                                                           target_feature="Regulation_Class",
                                                                           group_by_feature="Patient_ID",
                                                                           balance_focus=split_balancing)
        # >>
        workflow_dict['Split'] = f'Balancing {split_balancing} set'

        train_id = train["Patient_ID"]

        X_train, y_train = train.drop(columns=["Regulation_Class", "Patient_ID"], axis=1), train["Regulation_Class"]
        X_test, y_test = test.drop(columns=["Regulation_Class", "Patient_ID"], axis=1), test["Regulation_Class"]

        # >>
        workflow_dict['intersect of patients in train and test'] = list(set(X_train['Patient_ID'].unique()).intersection(set(X_test['Patient_ID'].unique())))

        # >>
        workflow_dict["Features"] = str(X_train.columns)

        for classifier in classifiers:

            # Write results after training every model
            # <<
            pd.DataFrame(results).to_excel(results_write_path)

            if classifier == "lr":
                # Pipelined model
                pipeline_logistic_regression = imblearn_Pipeline(steps=[
                    ('cc', ClusterCentroids(random_state=42, n_jobs=-1)),
                    ('standard_scaler', StandardScaler()),
                    ('pca', PCA(svd_solver='auto', iterated_power='auto')),
                    ('logistic_regression', LogisticRegression(solver='saga',
                                                               penalty='elasticnet',
                                                               warm_start=True,
                                                               n_jobs=-1,
                                                               random_state=42,
                                                               verbose=10))
                ])

                params = [
                    {
                        "pca__n_components": np.linspace(10, 68, num=36, dtype=int),
                        # "logistic_regression__solver": ["saga"],
                        "logistic_regression__max_iter": [100],
                        # "logistic_regression__penalty": ["elasticnet"],
                        "logistic_regression__C": np.linspace(0.01, 1000, num=200) ** 2,
                        "logistic_regression__l1_ratio": np.linspace(0, 1, num=200),
                        # "logistic_regression__warm_start": [True],
                        # "logistic_regression__n_jobs": [-1],
                        "logistic_regression__intercept_scaling": np.linspace(1, 100, num=100),
                        "logistic_regression__verbose": [10]
                    }
                ]

                randomized_search = RandomizedSearchCV(pipeline_logistic_regression,
                                                       params,
                                                       scoring="f1",
                                                       # cv=10
                                                       n_iter=num_iter,
                                                       cv=GroupShuffleSplit(random_state=42,
                                                                            test_size=0.2,
                                                                            n_splits=10,
                                                                            ).split(np.zeros(X_train.shape[0]),
                                                                                    y_train,
                                                                                    groups=train_id)
                                                       )
                # Fit model
                start_time = datetime.now()
                randomized_search.fit(X_train, y_train)
                end_time = datetime.now()
                training_time = end_time - start_time
                # >>
                model_path = f'{results_directory}/LR_{classifier}_{file_name}_{dataset}.sav'
                pickle.dump(randomized_search.best_estimator_, open(model_path, 'wb'))

                workflow_dict["model"] = str(randomized_search.best_estimator_)
                # >>
                time_dict = dict(
                    n_iter=num_iter,
                    training_time=str(training_time), start_time=str(start_time.time())[:-7],
                    end_time=str(end_time.time())[:-7],
                )

                # Predict
                y_pred = randomized_search.best_estimator_.predict(X_test)
                # >>
                scores_and_samples_dict = results_dict_from_confusion_matrix(y_test, y_pred, y_train)

                # >>
                first_columns_dict = dict(
                    Dataset=dataset,
                    Algorithm=classifier,
                    File_Name=file_name,
                )

                # Create results row and append to all results :list, later converted to dataframe
                results_row_dict = dict(**first_columns_dict, **scores_and_samples_dict, **time_dict, **workflow_dict)
                results.append(results_row_dict)

                # JSON model detailed model information
                # >>
                current_estimator = dict()
                current_estimator['file_name'] = file_name
                current_estimator['dataset'] = dataset
                current_estimator['model'] = str(randomized_search.best_estimator_)
                current_estimator['coefficients'] = str(randomized_search.best_estimator_['logistic_regression'].coef_)
                current_estimator['intercept'] = str(
                    randomized_search.best_estimator_['logistic_regression'].intercept_)
                current_estimator['best_parameters'] = str(randomized_search.best_params_)
                current_estimator["best_score"] = str(randomized_search.best_score_)

                # Create json dictionary with all informations about the model
                workflow_dict.pop('model')
                json_dict = dict(**current_estimator, **workflow_dict)

                # <<
                json_file = json.dumps(json_dict)
                json_write_path = f'{results_directory}/model_{file_name}_{dataset}_{classifier}.json'
                f = open(json_write_path, "w")
                f.write(json_file)
                f.close()

                # <<
                cv_scores_write_path = f'{results_directory}/cvResults_model_{file_name}_{dataset}_{classifier}.xlsx'
                pd.DataFrame(randomized_search.cv_results_).to_excel(cv_scores_write_path)

                # log
                testing_report = classification_report(y_test, y_pred, output_dict=True)
                print(pd.DataFrame().from_dict(testing_report).T)

                # Test samples with actual and predicted class
                tested_dataframe = X_test.copy()
                tested_dataframe['y_test'] = y_test
                tested_dataframe['y_pred'] = y_pred
                # >>
                testing_dataframe_write_path = f'{results_directory}/testing_frame_{file_name}_{dataset}_{classifier}.xlsx'
                pd.DataFrame(tested_dataframe).to_excel(testing_dataframe_write_path)

            # Random Forest Classifier
            if classifier == "rfc":
                pipeline_logistic_regression = imblearn_Pipeline(steps=[
                    ('cc', ClusterCentroids(random_state=42, n_jobs=-1)),
                    ('standard_scaler', StandardScaler()),
                    ('pca', PCA(svd_solver='auto', iterated_power='auto')),
                    ('random_forest', RandomForestClassifier(n_jobs=-1,
                                                             verbose=10,
                                                             random_state=42))
                ])

                params = [
                    {
                        "pca__n_components": np.linspace(10, 68, num=36, dtype=int),
                        "random_forest__n_estimators": list(np.linspace(1, 24, 24, dtype=int) ** 2),
                        "random_forest__max_depth": np.append(np.array(None),
                                                              np.linspace(1, 100, 100, dtype=int)).tolist(),
                        "random_forest__min_samples_split": np.linspace(2, 24, 23, dtype=int).tolist(),
                        "random_forest__min_samples_leaf": np.linspace(1, 24, 24, dtype=int).tolist(),
                        "random_forest__max_features": [None, "auto", "log2"],
                    }
                ]

                randomized_search = RandomizedSearchCV(pipeline_logistic_regression,
                                                       params,
                                                       scoring="f1",
                                                       n_iter=num_iter,
                                                       cv=GroupShuffleSplit(random_state=42,
                                                                            test_size=0.2,
                                                                            n_splits=10,
                                                                            ).split(np.zeros(X_train.shape[0]),
                                                                                    y_train,
                                                                                    groups=train_id)
                                                       )

                # Fit model
                start_time = datetime.now()
                randomized_search.fit(X_train, y_train)
                end_time = datetime.now()
                training_time = end_time - start_time
                # >>
                model_path = f'{results_directory}/RF_{classifier}_{file_name}_{dataset}.sav'
                pickle.dump(randomized_search.best_estimator_, open(model_path, 'wb'))

                workflow_dict["model"] = str(randomized_search.best_estimator_)
                # >>
                time_dict = dict(
                    n_iter=num_iter,
                    training_time=str(training_time), start_time=str(start_time.time())[:-7],
                    end_time=str(end_time.time())[:-7],
                )

                # Predict
                y_pred = randomized_search.best_estimator_.predict(X_test)
                # >>
                scores_and_samples_dict = results_dict_from_confusion_matrix(y_test, y_pred, y_train)

                # >>
                first_columns_dict = dict(
                    Dataset=dataset,
                    Algorithm=classifier,
                    File_Name=file_name,
                )

                # Create results row and append to all results :list, later converted to dataframe
                results_row_dict = dict(**first_columns_dict, **scores_and_samples_dict, **time_dict, **workflow_dict)
                results.append(results_row_dict)

                # >>
                current_estimator = dict()
                current_estimator['file_name'] = file_name
                current_estimator['dataset'] = dataset
                current_estimator['model'] = str(randomized_search.best_estimator_)
                current_estimator['feature_importances'] = str(
                    randomized_search.best_estimator_['random_forest'].feature_importances_)
                current_estimator['best_parameters'] = str(randomized_search.best_params_)
                current_estimator["best_score"] = str(randomized_search.best_score_)

                # Create json dictionary with all informations about the model
                workflow_dict.pop('model')
                json_dict = dict(**current_estimator, **workflow_dict)

                # <<
                json_file = json.dumps(json_dict)
                json_write_path = f'{results_directory}/model_{file_name}_{dataset}_{classifier}.json'
                f = open(json_write_path, "w")
                f.write(json_file)
                f.close()

                # <<
                cv_scores_write_path = f'{results_directory}/cvResults_model_{file_name}_{dataset}_{classifier}.xlsx'
                pd.DataFrame(randomized_search.cv_results_).to_excel(cv_scores_write_path)

                # log
                testing_report = classification_report(y_test, y_pred, output_dict=True)
                print(pd.DataFrame().from_dict(testing_report).T)

                # Test samples with actual and predicted class
                tested_dataframe = X_test.copy()
                tested_dataframe['y_test'] = y_test
                tested_dataframe['y_pred'] = y_pred
                # >>
                testing_dataframe_write_path = f'{results_directory}/testing_frame_{file_name}_{dataset}_{classifier}.xlsx'
                pd.DataFrame(tested_dataframe).to_excel(testing_dataframe_write_path)

# <<abs
pd.DataFrame(results).to_excel(results_write_path)

# %%




