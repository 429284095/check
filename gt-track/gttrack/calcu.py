#!coding:utf8
"""
Calcu feature function
"""
from __future__ import division
import numpy as np
FLOAT_EPSILON = 0.00000001
import collections
import tools
import math
import warnings
import operator
warnings.filterwarnings('error')

######################################################################################
##################### math calcu function ############################################

def calcu_entropy_func(nparray):
    counts = collections.Counter(nparray)
    counts = np.array(counts.values(),dtype = np.float)
    prob = counts / len(nparray)
    return (-prob * np.log2(prob)).sum()

def calcu_skewness(nparray):
    stdev = np.std(nparray)
    mean = np.mean(nparray)
    mean_element_pow3 = np.mean(nparray ** 3)
    skewness = (mean_element_pow3 - 3*mean*(stdev**2) - mean **3) / (stdev + FLOAT_EPSILON)**3
    return skewness

def calcu_kurtosis(nparray):
    stdev = np.std(nparray)
    mean = np.mean(nparray)
    mean_4 = np.mean((nparray - mean) ** 4)
    kurtosis = mean_4 / (stdev + FLOAT_EPSILON)**4 - 3
    return kurtosis

def calcu_w_func(nparray_x,nparray_y):
    s = np.sqrt(nparray_x**2 + nparray_y**2)
    x1 = nparray_x[:-1]
    x2 = nparray_x[1:]
    y1 = nparray_y[:-1]
    y2 = nparray_y[1:]
    s1 = s[:-1]
    s2 = s[1:]
    w = (x1 * x2 + y1 * y2) / (s1 * s2 + FLOAT_EPSILON)
    try:
        return_value = np.arccos(w)
    except:
        print w.tolist()
    return return_value


def calcu_w(track):
    return calcu_w_func(track.x, track.y)

######################## end #######################################################

##################### calcu entropy #######################################
def calcu_x_entropy(track):
    return calcu_entropy_func(track.x)

def calcu_y_entropy(track):
    return calcu_entropy_func(track.y)

def calcu_t_entropy(track):
    return calcu_entropy_func(track.t)

def calcu_entropy(track):
    option = {
        'x_entropy': [0, 4.26, 15],
        'y_entropy': [0, 3.13, 10],
        't_entropy': [0, 4.56, 15],
    }

    ration = 1
    sumidx = 0
    for k,v in option.iteritems():
        feat_val = getattr(track, k)
        idx = min(v[2]-1, max(0,int((feat_val - v[0]) * v[2] / (v[1] - v[0]))))
        sumidx = sumidx * ration + idx
        ration = v[2]

    return sumidx
################################# end #######################################

########################## calcu skewness and kurtosis ####################################
def calcu_x_skewness(track):
    result = calcu_skewness(track.x)
    return max(-5,min(10,result))

def calcu_y_skewness(track):
    result = calcu_skewness(track.y)
    return max(-5,min(10,result))

def calcu_t_skewness(track):
    result = calcu_skewness(track.t)
    return max(-5,min(10,result))

def calcu_vx_skewness(track):
    result = calcu_skewness(track.vx)
    return max(-5,min(10,result))

def calcu_w_skewness(track):
    result = calcu_skewness(track.w)
    return max(-5,min(10,result))

def calcu_x_kurtosis(track):
    result = calcu_kurtosis(track.x)
    return max(-10,min(80,result))

def calcu_y_kurtosis(track):
    result = calcu_kurtosis(track.y)
    return max(-10,min(80,result))

def calcu_t_kurtosis(track):
    result = calcu_kurtosis(track.t)
    return max(-10,min(80,result))

def calcu_vx_kurtosis(track):
    result = calcu_kurtosis(track.vx)
    return max(-10,min(80,result))

def calcu_w_kurtosis(track):
    result = calcu_kurtosis(track.w)
    return max(-10,min(80,result))
##################################### end #################################################

def calcu_passtime_steps_ratio(track):
    result = np.sum(track.t) / (len(track.t) * 100)
    return max(0,min(result,1.5))

def calcu_sumX_steps_ratio(track):
    result = np.sum(track.x) / len(track.x)
    return max(0,min(result,20))

def calcu_x_value_type(track):
    return len(set(track.x))

def calcu_y_value_type(track):
    return len(set(track.y))

def calcu_t_value_type(track):
    return len(set(track.t))

def calcu_vx_mean_var(track):
    cumsum = np.cumsum(track.vx)
    index = np.array([i+1 for i in range(len(track.vx))], dtype=np.float)
    cum_mean = cumsum / index
    return max(0,min(np.var(cum_mean),0.5))

def calcu_x_median_entropy(track):
    median_array = np.array([np.median(track.x[:i+1]) for i in range(len(track.x))])
    return calcu_entropy_func(median_array)

def calcu_y_median_entropy(track):
    median_array = np.array([np.median(track.y[:i+1]) for i in range(len(track.y))])
    return calcu_entropy_func(median_array)

def calcu_t_median_entropy(track):
    median_array = np.array([np.median(track.t[:i+1]) for i in range(len(track.t))])
    return calcu_entropy_func(median_array)

##################################### end #################################################



def calcu_passtime(track):
    return np.sum(track.t)


def points_count(track):
    return len(track.trackdata)


def calcu_s(track):
    s = np.sqrt(track.x ** 2 + track.y ** 2)
    return s
    #return [math.sqrt(track.x[i] ** 2 + track.y[i] ** 2) for i in range(0, track.n)]


def calcu_n12(track):
    return (int)(round(track.n / 3))  # 3分位点


def calcu_n23(track):
    return (int)(round(track.n * 2 / 3))

#######################################################################################################################
def calcu_index_negtive(track):
    count = 0
    for i in range(0, track.n):
        if (track.x[i] < 0 and track.y[i] < 0):
            count += 1
    return count / track.n


def calcu_v_maxmin_ratio(track):
    return (track.v_max + 1) / (track.v_min + 1)


def calcu_v_max_index(track):
    return track.v.index(track.v_max) + 1


def calcu_v_min_index(track):
    return track.v.index(track.v_min) + 1


def calcu_v1(track):
    return sum(track.s[0:track.n12]) / sum(track.t[0:track.n12])  # 第一段平均速度


def calcu_v2(track):
    return sum(track.s[track.n12 - 1:track.n23]) / sum(track.t[track.n12 - 1:track.n23])


def calcu_v3(track):
    return sum(track.s[track.n23 - 1:]) / sum(track.t[track.n23 - 1:])


def calcu_ratio_v1v2(track):
    return (track.v1 + FLOAT_EPSILON) / (track.v2 + FLOAT_EPSILON)  # 防止分母为0,V不为负


def calcu_ratio_v3v2(track):
    return (track.v3 + 1) / (track.v2 + 1)


def calcu_v(track):
    #return [(float(track.s[i]) / track.t[i]) for i in range(1, track.n - 1)]
    v = track.s / (track.t + FLOAT_EPSILON)
    return v
    #return [(float(track.s[i]) / track.t[i]) for i in range(0, track.n)]


def calcu_vx(track):
    vx = np.fabs(track.x / track.t)
    return vx
    #return [abs(float(track.x[i]) / track.t[i]) for i in range(0, track.n)]


def calcu_vy(track):
    vy = np.fabs(track.y / track.t)
    return vy
    #return [abs(float(track.y[i]) / track.t[i]) for i in range(0, track.n)]


def calcu_a(track):
    a = track.v[1:] - track.v[:-1] / track.t[:-1]
    return a
    #return [(float(v[i + 1] - v[i]) / track.t[i]) for i in range(0, len(v) - 1)]

def calcu_ax(track):
    ax = track.vx[1:] - track.vx[:-1] / track.t[:-1]
    return ax
    #vx = track.vx
    #return [(float(vx[i + 1] - vx[i]) / track.t[i]) for i in range(0, len(vx) - 1)]
    #return [(float(vx[i + 1] - vx[i]) / track.t[i]) for i in range(0, len(vx) - 1)]


def calcu_ay(track):
    ay = track.vy[1:] - track.vy[:-1] / track.t[:-1]
    return ay
    '''
    vy = track.vy
    return [(float(vy[i + 1] - vy[i]) / track.t[i]) for i in range(0, len(vy) - 1)]
    '''

def calcu_aa(track):
    aa = track.a[1:] - track.a[:-1] / track.t[1:-1]
    return aa
    '''
    a = track.a
    return [(float(a[i + 1] - a[i]) / track.t[i]) for i in range(0, len(a) - 1)]
    '''

def calcu_aax(track):
    aax = track.ax[1:] - track.ax[:-1] / track.t[1:-1]
    #print aax, 'aax'
    return aax
    '''
    ax = track.ax
    return [(float(ax[i + 1] - ax[i]) / track.t[i]) for i in range(0, len(ax) - 1)]
    '''


def calcu_aay(track):
    aay = track.ay[1:] - track.ay[:-1] / track.t[1:-1]
    return aay
    ay = track.ay
    return [(float(ay[i + 1] - ay[i]) / track.t[i]) for i in range(0, len(ay) - 1)]

#######################################################################################################################



def calcu_time_last_3(track):
    return np.sum(track.t[-3:])


def calcu_endtime_last_3time_ratio(track):
    return track.t[-1] / track.time_last_3


def calcu_max3time_sum(track):
    return tools.sum_abs_biggest_3_value(track.t)


def calcu_time_last_3_ratio(track):
    return track.time_last_3 / track.passtime


def calcu_time_first_3_ratio(track):
    return np.sum(track.t[:3]) / track.passtime


def calcu_sinangel(track):
    return np.sum(np.fabs(track.x)) / np.sum(track.s)
    '''
    if np.sum(track.s) == 0:
        for x,y in zip(track.x, track.y):
            print(x, y)
    else:
        return sum([abs(i) for i in track.x]) / sum(track.s)
    '''


def calcu_stop_points_percentage(track):
    '''
    cnt = 0
    for i in range(track.n):
        if abs(track.x[i]) <= 1 and abs(track.y[i]) <= 1:
            cnt = cnt + 1
    '''
    cnt = np.sum(np.logical_and(np.fabs(track.x) <= 1, np.fabs(track.y) <= 1))
    return cnt / track.n


def calcu_curve_y_std(track):
    return np.std(track.curve_x_y_t['curve_y'])


def calcu_curve_y_std123(track):
    curve_y = track.curve_x_y_t['curve_y']
    curve_y_std1 = np.std(curve_y[:track.n12])
    curve_y_std2 = np.std(curve_y[track.n12:track.n23])
    curve_y_std3 = np.std(curve_y[track.n23:])
    return np.std(np.array([curve_y_std1, curve_y_std2, curve_y_std3]))

#add normalization by Liuzhongyu in 2014-12-03
def calcu_x_trend_occupancy(track):
    x = track.x
    x1 = x[1:]
    x2 = x[:-1]
    return np.sum(x1 * x2 < 0) / float(track.n)
    '''
    x_trend_cnt = 1
    for i in range(2, track.n):
        if track.x[i] * track.x[i - 1] < 0:
            x_trend_cnt += 1
    return float(x_trend_cnt)/(track.n-1)
    '''

#add normalization by Liuzhongyu in 2014-12-03
def calcu_y_trend_occupancy(track):
    y = track.y
    y1 = y[1:]
    y2 = y[:-1]
    return np.sum(y1 * y2 < 0) / float(track.n)
    '''
    y_trend_cnt = 1
    for i in range(2, track.n):
        if track.y[i] * track.y[i - 1] < 0:
            y_trend_cnt += 1
    return float(y_trend_cnt)/(track.n-1)
    '''


def calcu_curve_x_y_t(track):
    curve_x_y_t = {
        'curve_x': np.array([]),
        'curve_y': np.array([]),
        'curve_t': np.array([]),
    }
    # curve_x = [0]
    # curve_y = [0]
    # curve_t = [0]
    for i in range(1, track.n):
        curve_x_y_t['curve_x'] = np.append( curve_x_y_t['curve_x'], np.sum(track.x[:i]) )
        curve_x_y_t['curve_y'] = np.append( curve_x_y_t['curve_y'], np.sum(track.y[:i]) )
        curve_x_y_t['curve_t'] = np.append( curve_x_y_t['curve_t'], np.sum(track.t[:i]) )
    return curve_x_y_t


def calcu_sum_abs_curve_y(track):
    return np.sum( np.abs(track.curve_x_y_t['curve_y']) )

def calcu_lchange(track):
    move = float(np.sum(track.s))
    distance = np.sqrt(sum(track.x) ** 2 + sum(track.y) ** 2)
    return move / (distance + FLOAT_EPSILON)


def find_same_v_long(track): ############################# need optimization to save memory
    # aother method to save memory
    v = track.v
    max = 0
    count = 0
    for i in range(1, len(v)):
        if v[i] == v[i - 1]:
            count += 1
        else:
            if count > max:
                max = count
            count = 0
    if count > max:
        max =  count
    return max
    '''
    v = track.v
    flag = range(len(v))
    temp = [0 for i in range(len(v))]
    for i in range(1, len(v)):
        if v[i] == v[i - 1]:
            flag[i] = flag[i - 1]
    for i in flag:
        temp[i] += 1
    return max(temp)
    '''


def calcu_thita(track):
    #thita = []

    s1 = track.s[:-1]
    s2 = track.s[1:]

    xx = track.x[:-1] + track.x[1:]
    yy = track.y[:-1] + track.y[1:]
    #xx = [track.x[i] + track.x[i + 1] for i in range(0, track.n - 1)]
    #yy = [track.y[i] + track.y[i + 1] for i in range(0, track.n - 1)]

    s3 = np.sqrt(xx ** 2 + yy ** 2)

    cos_data = (s1 ** 2 + s2 **2 - s3 ** 2) / (2 * s1 * s2)
    np.clip(cos_data, -1, 1)
    thita = np.degrees(np.arccos(cos_data))

    '''
    for i in range(0, len(s1)):
        cos_data = (s1[i] ** 2 + s2[i] ** 2 - s3[i] ** 2) / (2 * s1[i] * s2[i])
        cos_data = max([-1, cos_data])
        cos_data = min([1, cos_data])
        thita.append(math.degrees(math.acos(cos_data)))
    '''
    return thita


def calcu_thita_mean(track):
    return np.mean(track.thita)
    thita_mean = 0
    if track.n == 1:
        thita_mean = 0
    else:
        thita_mean = tools._mean(track.thita)
    return thita_mean


def calcu_thita_std(track):
    return np.std(track.thita)
    thita_std = 0
    if track.n == 1:
        thita_std = 0
    else:
        thita_std = tools._std(track.thita)
    return thita_std


def calcu_omiga_mean(track):
    if track.n == 1:
        omiga_mean = 0
    else:
        omiga_mean = np.sum(track.thita) / np.sum(track.t)
    return omiga_mean


def calcu_aw(track):
    return (track.w[1:] - track.w[:-1]) / track.t[1:-1]


def calcu_aaw(track):
    return (track.aw[1:] - track.aw[:-1]) / track.t[2:-1]


def calcu_x_1_pcc(track):
    x = np.fabs(track.x)
    x1 = x[::2]
    x2 = x[1::2]
    if track.n % 2 == 1:
        x2 = np.append(x2, np.mean(x2))
    return np.mean(   (x1 - np.mean(x1)) * (x2 - np.mean(x2))   ) / (np.std(x1) * np.std(x2))


# # # # DZ # # # #
def calcu_STCurve_k(track):
    x = track.x
    y = track.y
    t = track.t
    diff_x=np.diff(x)
    diff_y=np.diff(y)
    diff_t=np.diff(t)
    A = y[1:] * diff_t - diff_y * t[1:]
    B = t[1:] * diff_x - diff_t * x[1:]
    C = x[1:] * diff_y - diff_x * y[1:]
    k = ( A**2 + B**2 + C**2 )**0.5 / ( x[1:]**2 + y[1:]**2 + t[1:]**2 )**1.5
    diff_k = np.diff(k)
    max_k = np.nonzero(np.logical_and(diff_k[0:-1] > 0, diff_k[1:] < 0))[0]
    try:
        max_k = max_k[np.argmax(diff_k[max_k])]
        max_k += 2
        max_k /= np.float(track.n + 1)
    except:
        max_k = 1

    return max_k


# # # # LZY # # # #
def calcu_xt_pcc(track):
    x = track.x
    t = track.t
    return np.mean(   (x - np.mean(x)) * (t - np.mean(t))   ) / (np.std(x) * np.std(t))
    return tools._pcc(x,t)

#########one_class_svm#################
def calcu_x_entropy_normed(track):
    x = track.x
    counts = collections.Counter(x)
    n_class = len(counts.keys())
    counts = np.array(counts.values(),dtype = np.float)
    prob = counts / len(x)
    if n_class <= 1:
        return 0
    lg_prob = map(lambda x: math.log(x,n_class),prob)
    lg_prob = np.array(lg_prob)
    return (-prob * lg_prob).sum()

def calcu_t_entropy_normed(track):
    t = track.t
    counts = collections.Counter(t)
    n_class = len(counts.keys())
    counts = np.array(counts.values(),dtype = np.float)
    prob = counts / len(t)
    if n_class <= 1:
        return 0
    lg_prob = map(lambda t: math.log(t,n_class),prob)
    lg_prob = np.array(lg_prob)
    return (-prob * lg_prob).sum()

def calcu_x_entropy_e(track):
    x = track.x
    counts = collections.Counter(x)
    n_class = len(counts.keys())
    counts = np.array(counts.values(),dtype = np.float)
    prob = counts / len(x)
    if n_class <= 1:
        return 0
    lg_prob = map(lambda x: math.log(x),prob)
    lg_prob = np.array(lg_prob)
    return (-prob * lg_prob).sum()

def calcu_t_entropy_e(track):
    t = track.t
    counts = collections.Counter(t)
    n_class = len(counts.keys())
    counts = np.array(counts.values(),dtype = np.float)
    prob = counts / len(t)
    if n_class <= 1:
        return 0
    lg_prob = map(lambda t: math.log(t),prob)
    lg_prob = np.array(lg_prob)
    return (-prob * lg_prob).sum()

def merge_zero(track):
    new_track = []
    for i in range(len(track)):
        n = track[i][:]
        if n[0] == 0 and n[1] == 0:
            try:
                new_track[-1][2] += n[2]
            except:
                new_track.append(n)
        else:
            new_track.append(n)
    return new_track


def calcu_non_zero_y(track):
    try:
        y = np.array(track.trackdata_origin)[1:][:, 1]
    except:
        return 1

    count = collections.Counter(y)

    if len(y) == 0:
        return 1
    return min(1 - float(count[0]) / len(y), 0.7)


def calcu_tail(track):
    track_origin = track.trackdata_origin[1:]
    track_origin = merge_zero(track_origin)
    if len(track_origin) == 0:
        return 0
    track_origin = np.array(track_origin)
    t = track_origin[-6:][:,2]
    tail_num = max(t) - min(t)
    if tail_num == 0:
        return 0
    return math.log(tail_num,1.5)

def calcu_sum_x(track):
    return sum(track.x)

def calcu_point_len(track):
    track_origin = track.trackdata_origin[:]
    track_origin = filter(lambda x: (x[0],x[1]) != (0,0),track_origin)
    return min(len(track_origin) ** 2, 10000)

def calcu_first_t(track):
    t = track.t
    if len(t) == 0 or t[0] <= 0:
        return 0

    return math.log(t[0], 1.5)


def calcu_sequence_percent(track):
    s_list = {}
    t = np.array(track.trackdata_origin)[1:][:,2]
    for i in range(len(t)):
        in_list = False
        for s in s_list:
            if s - 3 < t[i] < s + 3:
                s_list[s] += 1
                in_list = True
                break
        if not in_list:
            s_list[t[i]] = 1
    sorted_s_list = sorted(s_list.items(), key=operator.itemgetter(1), reverse=True)

    length = min(6, len(sorted_s_list))
    total = 0
    for i in range(length):
        total += sorted_s_list[i][1]

    return max(total / float(len(t)), 0.6)

def calcu_sequence_num(track):
    s_list = {}
    t = np.array(track.trackdata_origin)[1:][:,2]
    for i in range(len(t)):
        in_list = False
        for s in s_list:
            if s - 3 < t[i] < s + 3:
                s_list[s] += 1
                in_list = True
                break
        if not in_list:
            s_list[t[i]] = 1
    sorted_s_list = sorted(s_list.items(), key=operator.itemgetter(1), reverse=True)

    length = len(t) * 0.9
    total = 0
    for i in range(len(sorted_s_list)):
        total += sorted_s_list[i][1]
        if total >= length:
            return min(i, 20)

def calcu_x_vibration(track):
    x_list = np.array(track.trackdata_origin)[1:][:, 0]

    if len(x_list) == 0:
        return 0

    total = 0
    for i in range(len(x_list) - 1):
        if x_list[i] - x_list[i + 1] != 0:
            total += math.log(abs(x_list[i] - x_list[i + 1]) ** 3)

    return min(float(total) / len(x_list), 5)

def calcu_most_sequence_average(track):
    t = np.array(track.trackdata_origin)[1:][:, 2]
    s = sorted(t, reverse=True)
    length = int(len(s) * 0.4)

    most = s[0:length]

    if len(most) == 0:
        return -1

    average = (float(sum(most)) / len(most)) ** 3

    if average == 0:
        return 0

    return math.log(average)

######################### not drag track's feature################
def calcu_t_max(track):
    return np.log(np.max(track.t))


def calcu_t_tail(track):
    return np.log(track.trackdata_origin[-1][2])
########################## end ####################################


################### continous feature ##############################
def calcu_x_continous_scores(track):
    freq = calcu_continous_part(track.x)
    return sum(freq) / float(len(track.x))

def calcu_y_continous_scores(track):
    freq = calcu_continous_part(track.y)
    return sum(freq) / float(len(track.y))

def calcu_y_max_continous(track):
    freq = calcu_continous_part(track.y)
    if not freq:
        return 0
    return np.max(np.array(freq))
#################### end ##############################


###################### shape feature###################
def calcu_t_peEntropy_2(track):
    return calcu_permutation_entropy(track.t, 2)


def calcu_x_peEntropy_345(track):
    x3 = calcu_permutation_entropy(track.x, 3)
    x4 = calcu_permutation_entropy(track.x, 4)
    x5 = calcu_permutation_entropy(track.x, 5)
    return (x3 + x4 + x5) / 3

def calcu_x_2bin_peEntropy234(track):
    x_bin = calcu_bin(track.x, 2)
    x2 = calcu_permutation_entropy(x_bin, 2)
    x3 = calcu_permutation_entropy(x_bin, 3)
    x4 = calcu_permutation_entropy(x_bin, 4)
    return (x2 + x3 + x4) / 3

def calcu_t_2bin_peEntropy234(track):
    t_bin = calcu_bin(track.t, 2)
    t2 = calcu_permutation_entropy(t_bin, 2)
    t3 = calcu_permutation_entropy(t_bin, 3)
    t4 = calcu_permutation_entropy(t_bin, 4)
    return (t2 + t3 + t4) / 3
#################### end #############################


################# velocity feature####################
def calcu_vx_2bin_peEntropy(track):
    vx = track.x / (track.t + 0.01)
    vx_bin = calcu_bin(vx, 2)
    vx2 = calcu_permutation_entropy(vx_bin, 2)
    vx6 = calcu_permutation_entropy(vx_bin, 6)
    return (vx2 + vx6) / 2


def calcu_xv_var(track):
    x = track.x
    t = track.t
    t = t[np.where(x != 0)]
    x = x[np.where(x != 0)]
    return np.log(np.var(np.abs(t / x)))
###################### end ###########################


###############outlier feature########################
def calcu_strange_t(track):
    return strange_points(track.t, 0.4)


def calcu_strange_x(track):
    return strange_points(track.x, 0.6)


def calcu_strange_vx(track):
    t = track.t
    x = track.x
    vx = x / (t + 0.01)
    return strange_points(vx, 0.9)
################# end ##################################


################### dependent function ###################
def calcu_continous_part(array):
    freq = []
    cnt = 1
    for i in range(len(array)-1):
        if array[i + 1] == array[i]:
            cnt += 1
            continue
        if cnt >= 2:
            freq.append(cnt)
    return freq


def calcu_atom_shape(val1, val2):
    if val1 < val2:
        return 0
    if val1 == val2:
        return 1
    if val1 > val2:
        return 2


def calcu_shape(array):
    n = len(array)
    result = []
    for i in range(n-1):
        result.append(calcu_atom_shape(array[i], array[i+1]))
    return result

def calcu_shape_index(shape):
     shape_len = len(shape)
     index = 0
     for i in range(shape_len):
         index += (shape[i] * (3 ** (shape_len-i-1)))
     return index


def calcu_permutation_entropy(array, n):
    array_len = len(array)
    shape_index = []
    for i in range(array_len-n+1):
        shape_index.append(calcu_shape_index(calcu_shape(array[i:i+n])))
    return calcu_entropy_func(shape_index)


def value_mapping(val, cut_values, nbin):
    if val < cut_values[0]:
        return 0.
    for i in range(len(cut_values)-1):
        if cut_values[i] <= val < cut_values[i+1]:
            return i / float(nbin)
    if val >= cut_values[-1]:
        return (len(cut_values)-2) / float(nbin)


def calcu_cut_values(array, nbin):
    sort_array = sorted(array)
    single_bin_lens = len(array) / nbin
    if single_bin_lens == 0:
        return sort_array
    cut_values = []
    cut_values.append(sort_array[0])
    for i in range(len(array)):
        if (i+1) % single_bin_lens == 0:
            cut_values.append(sort_array[i])
        else:
            continue
    return cut_values


def calcu_bin(array, nbin):
    cut_values = calcu_cut_values(array, nbin)
    array_bin = [value_mapping(val, cut_values, nbin) for val in array]
    return array_bin

def strange_points(array, ratio=1):
    overflow = np.abs(array - np.average(array)) - ratio * np.std(array)
    return np.count_nonzero(np.where(overflow > 0, 1, 0)) / float(len(array))
######################### end ##############################


######################### Append ZhangYing #############################
def calcu_dist(x1, x2):
    x1 = np.array(x1)[1:-1]
    x2 = np.array(x2)[1:-1]
    if np.equal(x1, x2).all():
        return 0
    mini = (x1 - x2) * (x1 - x2)
    dist_sig = np.sum(mini) ** 0.5
    dist_sig = np.exp(- dist_sig / 2.)
    if dist_sig > 100:
        dist_sig = 100
    return dist_sig


def calcu_t_var(track):
    start_index = track.n * 0.8
    test_t = track.t[int(start_index) - 1:]
    var_val = np.var(test_t) + 1e-4
    val = np.log(var_val * 10)
    if val > 16:
        val = 16
    return max(0, val)


def calcu_x_var(track):
    vx = np.round(track.vx, 2) * 100
    start_index = len(vx) * 0.8
    test_x = track.x[int(start_index) - 1:]
    var_val = np.var(test_x) + 1e-4
    val = np.log(var_val * 100)
    if val > 10:
        val = 10
    return max(0, val)


def calcu_vx_var(track):
    vx = np.round(track.vx, 2) * 100
    start_index = len(vx) * 0.8
    test_vx = vx[int(start_index) - 1:]

    var_val = np.var(test_vx) + 1e-4
    val = np.log(var_val * 10)
    if val > 16:
        val = 16
    return max(0, val)


def get_ax(t, vx):
    ax = np.zeros_like(vx)
    for i in range(len(vx)):
        if i == 0:
            ax[i] = vx[i] / (t[i] + 0.0001)
        else:
            ax[i] = (vx[i] - vx[i - 1]) / (t[i] + 0.0001)
    return ax


def get_ax_reverse(vx, t):
    ax = np.zeros_like(vx)
    for i in range(len(vx)):
        if i == 0:
            ax[i] = vx[i] / (t[i] + 0.0001)
        else:
            ax[i] = (vx[i] - vx[i - 1]) / (t[i] + 0.0001)
    return ax


def calcu_reverse_ax(track):
    vx = np.round(track.vx, 2) * 100
    ax = get_ax_reverse(track.t, vx)
    var_val = np.var(ax) + 1e-4
    return max(0, np.log(var_val))


def calcu_max_ax(track):
    # val = np.max(np.abs(track.ax))
    # val = np.log(val * 10)
    # if val > 12:
    #     val = 12
    # return max(0, val)

    # x = np.array(trackdata, dtype=np.float)[1:-1, 0]
    # t = np.array(trackdata, dtype=np.float)[1:-1, 2]
    # vx = x / (t + 0.0001)
    vx = np.round(track.vx, 2) * 100
    ax = get_ax(track.t, vx)

    val = np.max(np.abs(ax)) + 1e-4
    val = np.log(val * 10)
    if val > 12:
        val = 12
    return max(0, val)


def calcu_ax_entropy(track):
    vx = np.round(track.vx, 2) * 100
    ax = get_ax(track.t, vx)
    return calcu_entropy_func(np.abs(ax))


def cal_cov(x, y):
    np.cov(x, y)
    if len(x) != len(y):
        print('not same length')
        raise (Exception)
    x = np.array(x, dtype=np.float)
    y = np.array(y, dtype=np.float)
    cov = np.sum((x - np.mean(x)) * (y - np.mean(y))) / (len(x) - 1)
    return cov


def calcu_vx_cov(track):
    # vx = abs(track.x / (track.t + 0.0001))
    vx = abs(track.vx)
    vx = np.round(vx, 2) * 100
    ax = get_ax(track.t, vx)

    # print ax
    predict_vx = np.zeros_like(vx)
    for i in range(len(vx)):
        if i == 0:
            predict_vx[i] = 0
        else:
            predict_vx[i] = vx[i - 1] + ax[i - 1] * track.t[i]
    vx_dist = cal_cov(predict_vx[2:], vx[1: -1])
    vx_dist = np.int(vx_dist) + 1e-4

    val = max(0, np.log(abs(vx_dist)))
    if val > 25:
        val = 25
    return val


def calcu_vx_cov_reverse(track):
    vx = np.round(track.vx, 2) * 100
    ax = get_ax_reverse(track.t, vx)
    predict_vx = np.zeros_like(vx)
    for i in range(len(vx)):
        if i == 0:
            predict_vx[i] = 0
        else:
            predict_vx[i] = vx[i - 1] + ax[i - 1] * track.t[i]

    vx_dist = cal_cov(predict_vx[2:], vx[1: -1]) + 1e-4

    val = max(0, np.log(abs(vx_dist)))
    if val > 25:
        val = 25
    return val


def calcu_x_cov_reverse(track):
    vx = np.round(track.vx, 2) * 100
    ax = get_ax_reverse(track.t, vx)

    predict_x = np.zeros_like(track.x)
    for i in range(len(vx)):
        if i == 0:
            predict_x[i] = 0
        else:
            predict_x[i] = vx[i - 1] * track.t[i] + (ax[i - 1] * (track.t[i] ** 2)) / 2.

    x_dist = cal_cov(predict_x[2:], track.x[1: -1]) + 1e-4

    val = max(0, np.log(abs(x_dist)))
    if val > 28:
        val = 28
    return val


# 三点合一求方差
def calcu_x_filter(track):
    denoise_x = np.zeros_like(track.x)
    for i in range(track.n):
        if i == 0 or i == track.n - 1:
            denoise_x[i] = track.x[i]
        else:
            denoise_x[i] = (denoise_x[i - 1] + denoise_x[i] + denoise_x[i + 1]) / 3.
    var_val = np.var(track.x - denoise_x) * 10 + 1e-4
    var_val = np.log(var_val)
    if var_val > 12:
        var_val = 12
    return max(0, var_val)


def calcu_y_filter(track):
    denoise_y = np.zeros_like(track.y)
    for i in range(track.n):
        if i == 0 or i == track.n - 1:
            denoise_y[i] = track.y[i]
        else:
            denoise_y[i] = (denoise_y[i - 1] + denoise_y[i] + denoise_y[i + 1]) / 3.
    var_val = np.var(track.y - denoise_y) * 1000 + 1e-4
    var_val = np.log(var_val)
    if var_val > 12:
        var_val = 12
    return max(0, var_val)


def calcu_t_filter(track):
    denoise_t = np.zeros_like(track.t)
    for i in range(track.n):
        if i == 0 or i == track.n - 1:
            denoise_t[i] = track.t[i]
        else:
            denoise_t[i] = (denoise_t[i - 1] + denoise_t[i] + denoise_t[i + 1]) / 3.
    var_val = np.var(track.t - denoise_t) + 1e-4
    var_val = np.log(var_val)
    if var_val > 12:
        var_val = 12
    return max(0, var_val)

###################### end ##########################

###################### huangpu #########################
'''
基于T 300点采样
输入: 轨迹
输出: 采样list
'''
def interp_300_x(track):
    C, trackdata = np.array([1000, 100, 300 - 1]), np.array(track.trackdata_origin)
    # 位移->绝对坐标
    for i in range(trackdata.shape[0])[1:]:
        trackdata[i] = trackdata[i - 1] + trackdata[i]
    final_dot = np.array(trackdata[-1], np.float)
    final_dot[final_dot == 0] = 1
    delta = C / final_dot
    align_track = np.array(trackdata * delta).round()
    t_aligned, t_filled = align_track.T[-1], range(C[-1] + 1)
    x_filled = np.interp(t_filled, t_aligned, align_track.T[0])
    return x_filled

def interp_300_v(track):

    sub_arr = np.roll(track.interp_300_x, 1)
    return (track.interp_300_x - sub_arr)[1:]


def interp_300_a(track):
    sub_arr = np.roll(track.interp_300_v, 1)
    return (track.interp_300_v - sub_arr)[1:]

###################### end ############################
