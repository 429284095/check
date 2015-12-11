#!coding:utf8
import copy
import math
import numpy as np
# constant defined to avoid divide 0
FLOAT_SAFE = 0.00001


def _avg(arr):
    if len(arr)==0:
        return 0#TODO

    return _dolog(float(sum([abs(i) for i in arr])) / len(arr))


def _max(arr):
    return _dolog(max([abs(i) for i in arr]))


def _min(arr):
    return _dolog(min([abs(i) for i in arr]))


def _mean(arr):
    if len(arr) == 0:#TODO
        return _dolog(sum(arr) / FLOAT_SAFE)
    return _dolog(sum(arr) / (len(arr)))


def _stdev(arr):
    if len(arr)==0:
        return 0#TODO
    avg = _avg(arr)
    return _dolog(math.sqrt(sum([(abs(i) - avg) ** 2 for i in arr]) / len(arr)))


def _median(arr):
    if len(arr)==0:
        return 0#TODO

    data = copy.deepcopy(arr)
    data.sort()
    n = len(data)
    pos = n >> 1
    if (n % 2 == 0):
        return (float((data[pos - 1] + data[pos]) / 2))
    else:
        return data[pos]


def sum_abs_biggest_3_value(list_data):
    """
    Sum abs of  the biggest 3 data
    :param list_data:
    :return: :rtype:
    """
    data = copy.deepcopy(list_data)
    data.sort()

    return sum_abs_list(data[-3:])


def sum_abs_list(list_data):
    """
    绝对值求和
    :param list_data:
    """
    sum_abs = 0.0
    for item in list_data:
        sum_abs += abs(item)
    return sum_abs


def _dolog(val):
    return val

    sig = 1 if val > 0 else -1
    return sig * 1000 * math.log(abs(val) + 1)

# # # # DZ # # # #
def _acf(data,k):   
    import numpy as np
    data = np.array(data, dtype=np.float)
    data = abs(data)
    # print x
    # print vx
    K = 3
    # closure
    def acf(v):
        N  = len(v)
        v -= np.mean(v) # demean
        def acf_k(k):
          try:
            return sum(v[1:N-k]*v[k+1:N])/(sum(v*v)+np.spacing(1)) #eps
          except Exception,e:
            print e
        return acf_k
    
    acf_data = acf(data)
    return acf_data(k)


####Liuzhongyu####

def _pcc(list_1,list_2):
    array_1 = np.array(list_1)
    array_2 = np.array(list_2)
    array_1_mean = np.mean(array_1)
    array_2_mean = np.mean(array_2)
    array_1_std = np.std(array_1)
    array_2_std = np.std(array_2)
    combine_array = (array_1 - array_1_mean) * (array_2 - array_1_mean)
    if array_1_std == 0 or array_2_std == 0:
        return 1.0
    return np.mean(combine_array) / (array_1_std * array_2_std)


# # # #  DZ  # # # #
def _entrophy(data):
    data = np.array(data, dtype=np.float)
    bin_width = 0.1
    bin_edges = np.arange(data.min(),data.max()+bin_width,bin_width)
    hist, bin_edges = np.histogram(data, bins=bin_edges, density=True)
    zero_idx = np.nonzero(hist==0)[0]

    # in case of warnings of log(0), change to log(1) in 
    # accordance with textbook definition of '0log0=0'
    hist[zero_idx] = 1 
    Q = -1 * hist * np.log(hist)
    return sum(Q)

