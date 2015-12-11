#!coding:utf8
""" Track feature define class, use functions in calcu module to calculate
"""

from calcu import *
from request import Feature, Request


class RequestAttr(Request):
    def __init__(self, req={}):
        super(RequestAttr, self).__init__(req)
        self.passtime = Feature(self, calcu_passtime, False)
        self.n = Feature(self, points_count, False)

        self.mousedown_xpos = Feature(self, lambda track: track.trackdata_origin[0][0])
        self.mousedown_ypos = Feature(self, lambda track: track.trackdata_origin[0][1])
        self.mousedown_time = Feature(self, lambda track: track.trackdata_origin[0][2])

        self.mouseup_xpos = Feature(self, lambda track: track.trackdata_origin[-1][0])
        self.mouseup_ypos = Feature(self, lambda track: track.trackdata_origin[-1][1])
        self.mouseup_time = Feature(self, lambda track: track.trackdata_origin[-1][2])

        self.n12 = Feature(self, calcu_n12, False)  # TO-DO
        self.n23 = Feature(self, calcu_n23, False)  # TO-DO

        self.s = Feature(self, calcu_s, False)

        # region [v,a,aa,w]*[1,x,y]*[max,min,avg,stdev,mean,median]
        self.v = Feature(self, calcu_v, False)
        self.vx = Feature(self, calcu_vx, False)
        self.vy = Feature(self, calcu_vy, False)

        self.a = Feature(self, calcu_a, False)
        self.ax = Feature(self, calcu_ax, False)
        self.ay = Feature(self, calcu_ay, False)

        self.aa = Feature(self, calcu_aa, False)
        self.aax = Feature(self, calcu_aax, False)
        self.aay = Feature(self, calcu_aay, False)

        self.w = Feature(self, calcu_w, False)
        self.aw = Feature(self, calcu_aw, False)
        self.aaw = Feature(self, calcu_aaw, False)

        self.normal()  # calculate all stat value
        # endregion

        # -------------新加入特征--2014-09-18


        self.w0 = Feature(self, lambda track: track.w[0])
        self.wn = Feature(self, lambda track: track.w[-1])

        self.same_v_count = Feature(self, find_same_v_long)

        # # # DZ # # # #
        self.STCurve_k = Feature(self, calcu_STCurve_k, group='randomforest')

        self.x_entropy = Feature(self, calcu_x_entropy, group='forbidden')
        self.y_entropy = Feature(self, calcu_y_entropy, group='forbidden')
        self.t_entropy = Feature(self, calcu_t_entropy, group='forbidden')

        self.entropy = Feature(self, calcu_entropy, group='simplerule')

        self.passtime_steps_ratio = Feature(self, calcu_passtime_steps_ratio)
        self.sumX_steps_ratio = Feature(self, calcu_sumX_steps_ratio)

        self.vx_skewness = Feature(self, calcu_vx_skewness)
        self.x_skewness = Feature(self, calcu_x_skewness)
        self.t_skewness = Feature(self, calcu_t_skewness)
        self.y_skewness = Feature(self, calcu_y_skewness)
        self.w_skewness = Feature(self, calcu_w_skewness)

        self.vx_kurtosis = Feature(self, calcu_vx_kurtosis)
        self.x_kurtosis = Feature(self, calcu_x_kurtosis)
        self.t_kurtosis = Feature(self, calcu_t_kurtosis)
        self.y_kurtosis = Feature(self, calcu_y_kurtosis)
        self.w_kurtosis = Feature(self, calcu_w_kurtosis)

        self.x_value_type = Feature(self, calcu_x_value_type)
        self.y_value_type = Feature(self, calcu_y_value_type)
        self.t_value_type = Feature(self, calcu_t_value_type)

        ##########one_class_svm#################
        self.x_entropy_normed = Feature(self, calcu_x_entropy_normed)
        self.t_entropy_normed = Feature(self, calcu_t_entropy_normed)
        self.x_entropy_e = Feature(self, calcu_x_entropy_e)
        self.t_entropy_e = Feature(self, calcu_t_entropy_e)
        self.tail = Feature(self, calcu_tail)
        self.first_t = Feature(self, calcu_first_t)
        self.non_zero_y = Feature(self, calcu_non_zero_y)
        self.sum_x = Feature(self, calcu_sum_x)
        self.point_len = Feature(self, calcu_point_len)
        self.sequence_percent = Feature(self, calcu_sequence_percent)
        self.sequence_num = Feature(self, calcu_sequence_num)
        self.x_vibration = Feature(self, calcu_x_vibration)
        self.most_sequence_average = Feature(self, calcu_most_sequence_average)

        ###################new feature####################
        self.t_max = Feature(self, calcu_t_max)
        self.t_tail = Feature(self, calcu_t_tail)

        self.x_continous_scores = Feature(self, calcu_x_continous_scores)
        self.y_continous_scores = Feature(self, calcu_y_continous_scores)
        self.y_max_continous = Feature(self, calcu_y_max_continous)

        self.t_peEntropy_2 = Feature(self, calcu_t_peEntropy_2)
        self.x_peEntropy_345 = Feature(self, calcu_x_peEntropy_345)
        self.x_2bin_peEntropy234 = Feature(self, calcu_x_2bin_peEntropy234)
        self.t_2bin_peEntropy234 = Feature(self, calcu_t_2bin_peEntropy234)

        self.vx_2bin_peEntropy = Feature(self, calcu_vx_2bin_peEntropy)
        self.xv_var = Feature(self, calcu_xv_var)

        self.strange_t = Feature(self, calcu_strange_t)
        self.strange_x = Feature(self, calcu_strange_x)
        self.strange_vx = Feature(self, calcu_strange_vx)
        #################### end ###########################

        #################### ZhangYing Start #####################
        self.t_var = Feature(self, calcu_t_var)
        self.x_var = Feature(self, calcu_x_var)
        self.vx_var = Feature(self, calcu_vx_var)
        self.reverse_ax = Feature(self, calcu_reverse_ax)
        self.max_ax = Feature(self, calcu_max_ax)
        self.ax_entropy = Feature(self, calcu_ax_entropy)
        self.vx_cov = Feature(self, calcu_vx_cov)
        self.vx_cov_reverse = Feature(self, calcu_vx_cov_reverse)
        self.x_cov_reverse = Feature(self, calcu_x_cov_reverse)
        self.x_filter = Feature(self, calcu_x_filter)
        self.y_filter = Feature(self, calcu_y_filter)
        self.t_filter = Feature(self, calcu_t_filter)
        self.t_var = Feature(self, calcu_t_var)
        self.x_var = Feature(self, calcu_x_var)
        self.vx_var = Feature(self, calcu_vx_var)
        self.reverse_ax = Feature(self, calcu_reverse_ax)
        self.max_ax = Feature(self, calcu_max_ax)
        self.ax_entropy = Feature(self, calcu_ax_entropy)
        self.vx_cov = Feature(self, calcu_vx_cov)
        self.vx_cov_reverse = Feature(self, calcu_vx_cov_reverse)
        self.x_cov_reverse = Feature(self, calcu_x_cov_reverse)
        self.x_filter = Feature(self, calcu_x_filter)
        self.y_filter = Feature(self, calcu_y_filter)
        self.t_filter = Feature(self, calcu_t_filter)
        ################### ZhangYing End   ######################

        #################### HuangPu start #######################
        self.interp_300_x = Feature(self, interp_300_x)
        self.interp_300_v = Feature(self, interp_300_v)
        self.interp_300_a = Feature(self, interp_300_a)
        ################### HuangPu end ##########################


    @classmethod
    def get_cls_fields(cls, feature_grp=['default']):
        return cls().get_all_field(feature_grp)

    def normal(self):
        """
        the feature v,a,aa,w list
        generates a serial other features,
        use this tool we can reduce many code
        :return:
        :rtype:
        """

        features = ['v', 'vx', 'vy', 'a', 'ax', 'ay', 'aa', 'aax', 'aay', 'aw', 'aaw']
        stat = ['_max', '_min', '_avg', '_stdev', '_mean', '_median', '_entrophy']
        # stat = ['_max', '_min', '_avg', '_stdev','_mean','_median']
        for f in features:
            for s in stat:
                feature_name = f + s
                exec 'self.%s = Feature(self, lambda track: getattr(tools, "%s")(track.%s))' % (feature_name, s, f)
        # # # DZ # # # #
        K = 2
        features = ['v', 'vx', 'vy', 'w', 'x', 'y', 't']
        stat = ['_acf']
        for f in features:
            for s in stat:
                for k in range(1, K + 1):
                    feature_name = '%s%s_%d' % (f, s, k)
                    exec 'self.%s = Feature(self, lambda track: getattr(tools, "%s")(track.%s, %d))' % (
                    feature_name, s, f, k)
