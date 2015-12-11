# -*- coding:utf-8 -*-
__author__ = 'poo'

import cPickle as pickle
import numpy as np


# np.seterr(all='print')

def load_models():
    combine_clf = pickle.load(open("../../model/gbdt_manual_split.lr"))
    mrf = pickle.load(open("../../model/manual_diff_feature.rf"))
    RFs = [pickle.load(open("../../model/gbdt_rf%d.clf" % i)) for i in range(3)]
    return RFs, mrf, combine_clf


RFs, mrf, combine_clf = load_models()

import sys

sys.path.append("./")
from feature import RequestAttr
import json

from ast import literal_eval as make_tuple


def to_json(request):
    # try to parse json
    try:
        return json.loads(request)
    except ValueError:
        try:
            return make_tuple(request)
        except ValueError:
            return None
        return None


def load_tracks():
    f = "../../data/people/people"
    # f = "../../data/people/append/common/people_16"
    # f = "../data/robot_diff/三个数据/mixture/1027_jingchu_xaa"
    # f= "../../data/robot/20150503_133000-yy-step-hour-1_passed_ie9"
    # f = "../../data/robot/20150317_000000-6cn-step-hour-24_extracted"
    return [to_json(l) for l in open(f).readlines()[:14000]]


from time import time
import msgpack
from RF import RandomForest


def load_features():
    n, manual_feature_list = msgpack.load(open("gtutest/gbdts_lr_clf.pkg"))
    return manual_feature_list


def ensemble_clf_test():
    feature_names = load_features()
    clf = RandomForest("gtutest/gbdts_lr_clf.pkg", None)
    global RFs, mrf, combine_clf
    cnt, p_cnt, l_p_cnt = 0, 0, 0
    ds = load_tracks()
    s_time = time()
    for td in ds:

        track = RequestAttr(td)
        if len(track.trackdata_origin) < 3: continue
        if cnt > 12000: break
        cnt += 1
        try:
            # 线上模型
            p_cnt += clf.predict_proba(track)

            # 线下模型
            c_features, m_features = [], [getattr(track, feature_name) for feature_name in feature_names]
            for rf, f in zip(RFs, (track.interp_300_x, track.interp_300_v, track.interp_300_a)):
                c_features += [t[0].predict(f) for t in rf.estimators_]
            c_features += [t[0].predict(np.array([m_features])) for t in mrf.estimators_]
            res = combine_clf.predict(np.array(c_features).T)[0]
            l_p_cnt += 0 if res > 0.5 else 1
        except:
            pass
            # print td


    e_time = time()
    print s_time, e_time, e_time - s_time, (e_time - s_time) / float(cnt + 1e-4)
    print "CNT: %d P_CNT: %d Score: %f" % (cnt, p_cnt, p_cnt / float(cnt + 1e-4))
    print "CNT: %d L_P_CNT: %d Score: %f" % (cnt, l_p_cnt, l_p_cnt / float(cnt + 1e-4))


if __name__ == '__main__':
    # calcu_foo_test()
    ensemble_clf_test()
