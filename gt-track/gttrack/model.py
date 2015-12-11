import os
import cPickle as pickle
import numpy as np


def load_model(path):
    return 0, 0  # do not use pkl
    estimator = None
    with file(path) as fid:
        estimator = pickle.load(fid)
        feat_sel_name = pickle.load(fid)

    # print feat_sel_name
    estimator.verbose = True
    estimator.n_jobs = 1
    return estimator, feat_sel_name


# PKL_DIR = os.path.dirname(__file__) + '/pkl'
PKL_DIR = '/www/gt-server-new/gttrack'

randomforest_model = load_model(PKL_DIR + '/RandomForest.pkl')





