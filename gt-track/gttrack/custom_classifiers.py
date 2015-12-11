# -*- coding:utf-8 -*-
__author__ = 'huangpu'
'''
sklearn分类器线上代码,根据项目需求仅对部分功能做实现
输入: sklearn 分类器属性 tuple
输出: 1:机器 0: 人
'''

import numpy as np
import msgpack

# (GBDT x 3 + GBDT) -> LR 增强模型
class GBDTsLREnsembleClassifier():
    def __init__(self, ensemble_clf_model):
        with open(ensemble_clf_model, "rb") as file_handler_in:
            self.classifiers, self.manual_feature_list = msgpack.load(file_handler_in)
            # interp GBDT x 3: x, v, a | manual GBDT | Output: LR
            self.interp_gbdts = [CustomGradientBoostingClassifier(arg) for arg in self.classifiers[:3]]
            self.manual_gbdt = CustomGradientBoostingClassifier(self.classifiers[3])
            self.lr = CustomLogisticRegression(self.classifiers[4])

    def predict(self, track):
        interp_features = (track.interp_300_x, track.interp_300_v, track.interp_300_a)

        manual_features = [getattr(track, feature_name) for feature_name in self.manual_feature_list]
        first_layer_output = []
        for interp_gbdt, interp_feature in zip(self.interp_gbdts, interp_features):
            first_layer_output += [dt.predict(interp_feature) for dt in interp_gbdt.estimators_]
        first_layer_output += [dt.predict(manual_features) for dt in self.manual_gbdt.estimators_]
        return self.lr.predict(first_layer_output)


# GBDT
class CustomGradientBoostingClassifier():
    def __init__(self, args):
        self.estimators_ = [CustomDecisionTree(arg) for arg in args]

# LR
class CustomLogisticRegression():
    def __init__(self, (coef_, intercept_)):
        if type(coef_) != np.ndarray:
            coef_ = np.array(coef_)
        self.coef_ = coef_
        self.intercept_ = intercept_

    def predict(self, x):
        if type(x) == np.ndarray:
            x = np.array(x)
        prob = np.sum(x * self.coef_) + self.intercept_
        return 1 if prob > 0 else 0

# DT
class CustomDecisionTree():
    def __init__(self, (feature, value, n_node_samples, threshold, children_left, children_right)):
        self.feature = feature
        self.value = value
        self.n_node_samples = n_node_samples
        self.threshold = threshold
        self.children_left = children_left
        self.children_right = children_right

    def predict(self, x):
        node_index, res = 0, 1
        while True:
            feature_index = self.feature[node_index]
            if node_index == -1 or feature_index < 0:
                res = self.value[node_index][0][0]
                break
            threshold = self.threshold[node_index]
            feature_value = x[feature_index]
            if feature_value <= threshold:
                node_index = self.children_left[node_index]
            else:
                node_index = self.children_right[node_index]
        return res
