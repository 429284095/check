# -*- coding:utf-8 -*-
import numpy as np
import msgpack
import base64
import json
import feature
import re
from custom_classifiers import *

chromeRe = re.compile(r'Chrome/(\d*)\.')


def is_old_ie(ua):
    if "Windows NT 5.1" in ua and "MSIE" in ua:
        return True
    if "MSIE 8.0" in ua:
        return True
    return False


def is_xp_chrome(ua):
    return "Windows NT 5.1" in ua and "Chrome" in ua


def is_mobile(ua):
    return "Mobile" in ua or "Android" in ua


def is_old_chrome(ua):
    if "Chrome" in ua and not is_xp_chrome(ua):
        try:
            version = int(chromeRe.findall(ua)[0])
        except:
            # print ua
            # print chromeRe.findall(ua)
            return False
        if version <= 32:
            return True
    return False


def is_common(ua):
    return not is_mobile(ua) and not is_xp_chrome(ua) and not is_old_chrome(ua)


def is_chrome(ua):
    if ua is None:
        return False
    return 'Chrome' in ua


def is_ie8(ua):
    if ua is None:
        return False
    return "MSIE 8.0" in ua \
           or "MSIE 7.0" in ua \
           or "MSIE 6.0" in ua


def is_ie9(ua):
    if ua is None:
        return False
    return "MSIE 9.0" in ua \
           or "MSIE 10.0" in ua \
           or "MSIE 11" in ua \
           or "Trident/7.0" in ua


def is_mobile(ua):
    if ua is None:
        return False
    return "Android" in ua or "iPhone" in ua


def is_other(ua):
    if ua is None:
        return False
    if not is_ie8(ua) and not is_ie9(ua) and not is_mobile(ua) and not is_chrome(ua):
        if "WinHttp" in ua:
            return True
    return False


def parse_nparray(arr_str):
    dtype, arr_str, shape = json.loads(arr_str)
    return np.fromstring(base64.decodestring(arr_str), dtype).reshape(shape)


def value_mapping(val, cut_values, nbin):
    if val < cut_values[0]:
        return 0.
    for i in range(len(cut_values) - 1):
        if cut_values[i] <= val < cut_values[i + 1]:
            return i / float(nbin)
    if val >= cut_values[-1]:
        return (len(cut_values) - 2) / float(nbin)


class RandomForest(object):
    def __init__(self, ensemble_clf_model, one_class_svm_model):
        self.ensemble = GBDTsLREnsembleClassifier(ensemble_clf_model)
        if one_class_svm_model is None:
            print "one_class_svm_model missing!"
            return

        self.ocs = {}
        for name, svm_model_file in one_class_svm_model.iteritems():
            with open(svm_model_file, 'rb') as f:
                self.ocs[name] = {}
                self.ocs[name]['support_vector'] = np.load(f)
                self.ocs[name]['alpha'] = np.load(f)
                self.ocs[name]['gamma'] = np.load(f)
                self.ocs[name]['b'] = np.load(f)
                self.ocs[name]['ocs_feature'] = np.load(f).tolist()
                # print self.ocs[name]['ocs_feature']

    def predict_proba(self, track):
        res = self.ensemble.predict(track)
        # 反转 ensemble 结果: people: 0, robot: 1 -> people: 1, robot: 0
        return 0 if res > 0.5 else 1

    def ocs_predict(self, name, track):
        data = np.array([getattr(track, feature_name) for feature_name in self.ocs[name]['ocs_feature']])
        return np.sign(np.dot(self.ocs[name]['alpha'],
                              self.rbf(self.ocs[name]['support_vector'], data, gamma=self.ocs[name]['gamma'])) +
                       self.ocs[name]['b'])

    def rbf(self, x, z, gamma=1.0):
        distance = x - z
        square_distance = map(lambda x: np.dot(x, x), distance)
        square_distance = np.array(square_distance)
        return np.exp(square_distance * -gamma)

    def predict(self, track):
        category = "people"
        id_list = [
            "32b06e5c9122046cb516c92a0362ca01",  # yy
            "1c9a08e1df76f5200ea621909c36bddf",  # sina-show
            "94cf4ec5d71cb4eeb94d2f44ba9b1086",  # 17173
            "f86a44b3c5da47e6c6e9d0581f3cf076",  # juxiangyou
            "703d2936f03e6235c9cb6383420f162c",  # baozou
            "396327e49cf278d0ae59127b8c900453",  # poco
            "82a5344b58cc712926bc694dda62c764",  # ceair
            '50b04f39a4906c9b8cf28a72fa11f488',  # 178
            '1e808b9127450447ada1bc34affe013a',  # heilongjiang
            '65a7fafe61945517c4f2c58165f4906e',  # 76shequ
            '7a5d83647ca6c3e1d57af217a2e34e20',  # huodongxing
            '5d8e96f7b149559b0c80d44d2b0c0cc1',  # jingchu
            '22ed6691b280f038f62336baa4785f5c',  # youxigou
        ]
        captcha_id = getattr(track, 'captcha_id')
        if captcha_id not in id_list:
            probability = self.predict_proba(track)
            if probability >= 0.5:
                category = "people"
            else:
                category = "RF"
            return category

        # 针对ocs样本也要启用RF, Eg: youxigou
        if not self.predict_proba(track) >= 0.5:
            return "RF"

        ua = getattr(track, 'ua')

        if is_common(ua):
            # use new platform like this temporarily, use all new platform when ready
            ocs_result = self.ocs_predict('common', track)
            if ocs_result == -1:
                category = 'ocs_common'
        elif is_old_chrome(ua):
            ocs_result = self.ocs_predict('old_chrome', track)
            if ocs_result == -1:
                category = 'ocs_old_chrome'
        elif is_xp_chrome(ua):
            ocs_result = self.ocs_predict('xp_chrome', track)
            if ocs_result == -1:
                category = 'ocs_xp_chrome'
        else:
            if is_chrome(ua):
                ocs_result = self.ocs_predict('chrome', track)
                if ocs_result == -1:
                    category = 'ocs_chrome'
            if is_ie9(ua):
                ocs_result = self.ocs_predict('ie9', track)
                if ocs_result == -1:
                    category = 'ocs_ie9'
            if is_ie8(ua):
                ocs_result = self.ocs_predict('ie8', track)
                if ocs_result == -1:
                    category = 'ocs_ie8'
            if is_other(ua):
                ocs_result = self.ocs_predict('ie9', track)
                if ocs_result == -1:
                    category = 'ocs_other'

        return category
