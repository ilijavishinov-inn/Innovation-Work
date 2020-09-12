import warnings

from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np


def feature_engineering_end(dataframe, feature_name, method = None):
    """
    :param dataframe:
    :param feature_name:
    :param method:

    :return:
    """

    if method is None:
        warnings.warn("Give the type of feature engineering as an argument")
        return None

    if method == 'maen':
        dataframe.replace(
            to_replace = {feature_name: {1: "night", 2: "night", 3: "night", 4: "night",
                                         5: "night", 6: "night", 7: "morning", 8: "morning",
                                         9: "morning", 10: "morning", 11: "morning", 12: "morning",
                                         13: "afternoon", 14: "afternoon", 15: "afternoon",
                                         16: "afternoon",
                                         17: "afternoon", 18: "afternoon", 19: "evening", 20: "evening",
                                         21: "evening", 22: "evening", 23: "evening", 24: "evening",
                                         0: "evening"}},
            inplace = True,
        )

    if method == 'cyclical':

        dataframe[f'{feature_name}_sin'] = np.sin( dataframe[feature_name] * ( 2. * np.pi / 24 ) )
        dataframe[f'{feature_name}_cos'] = np.cos( dataframe[feature_name] * ( 2. * np.pi / 24 ) )
        dataframe.drop(feature_name, axis = 1, inplace = True)

    return dataframe


def remove_null_values(dataframe, return_num_removed = False):

    num_rows_before_drop_null = dataframe.shape[0]

    dataframe.dropna(axis = 0, inplace = True)

    num_rows_removed_null = num_rows_before_drop_null - dataframe.shape[0]

    if return_num_removed:
        return dataframe, num_rows_removed_null
    else:
        return dataframe


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


def remove_outliers(dataframe,  method = 'z-score', feature_names=None, return_num_removed = False):
    """ Remove outliers from a dataframe. Outliers are considered data_1-24h points which deviate more than 3 sigmas from the mean
    after computing the z-score.

    :param dataframe: pandas.DataFrame
        Dataframe from which outliers are removed
    :param method: {'z-score', 'iqr'}, default = 'z-score'

    :param feature_names: list of strings, default="all"
        Cirteria features (list of column names) on which the outlier removal is based

    :return: pandas.DataFrame
    """

    num_rows_before_outlier_removal = dataframe.shape[0]

    if feature_names is None or not feature_names:
        feature_names = dataframe.columns

    if method == 'iqr':

        frame = dataframe[feature_names]

        q1 = frame.quantile(0.25)
        q3 = frame.quantile(0.75)
        iqr = (q1 - q3).abs()

        dataframe_without_outliers = dataframe[~ ((frame < (q1 - 1.5*iqr)) | (frame > (q3 + 1.5*iqr))).any(axis = 1)]

        num_rows_removed_outliers = num_rows_before_outlier_removal - dataframe_without_outliers.shape[0]

        if return_num_removed:
            return dataframe_without_outliers, num_rows_removed_outliers
        else:
            return dataframe_without_outliers

    if method == 'z-score':

        frame = dataframe[feature_names]

        z_scores = stats.zscore(frame)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        dataframe_with_removed_outliers = dataframe[filtered_entries]

        return dataframe_with_removed_outliers, abs_z_scores


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
