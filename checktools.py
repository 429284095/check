import math
import re
from decode import decodedatacompress

def check_rule(track):
    flag = 'people'

    if track.mousedown_xpos == 0 or track.mousedown_ypos == 0 or track.mousedown_time != 0:
        return 'robot'
    if 0.2 <= track.t_max <= 0.22 and 0.8 <= track.t_peEntropy_2 <= 1.0:
        return 'robot'
    yy = "32b06e5c9122046cb516c92a0362ca01"
    captcha_id = getattr(track, 'captcha_id')
    if captcha_id == yy:
        if 3.7 <= track.t_entropy <= 4.7 and 5 <= track.sequence_num <= 10 and track.w_kurtosis <= 3\
        and track.t_value_type >= 20:
            return 'robot'
        if track.passtime_steps_ratio <= 0.18 and track.y_entropy >= 1.2:
            return 'robot'

    shenhudong = "38d583f968b30c5b0e222daac9e0ac10"
    if captcha_id == shenhudong:
        if 2.5 <= track.t_tail <= 4.:
            return 'robot'
        if 3.8 <= track.t_max <= 4.3 or track.xv_var <= 3.:
            return 'robot'

    weifeng = '19a9ffd48e51bb2a9205941d92a4bf5f'
    if captcha_id == weifeng:
        if track.t_max < 0.02 and track.t_tail <= 0.04 and track.xv_var < 0.02:
            return 'robot'

    jingchu = "5d8e96f7b149559b0c80d44d2b0c0cc1"
    if captcha_id == jingchu:
        if track.x_continous_scores <= 2. and track.x_peEntropy_345 <= 3. and track.y_entropy >= 2.:
            return 'robot'
        if track.t_max <= 4. and track.xv_var <= 5. and track.x_2bin_peEntropy234 >= 1.8:
            return 'robot'
        if track.t_peEntropy_2 <= 1.4 and track.y_entropy <= 0.1 and track.t_max <= 5.4:
            return 'robot'
        if track.t_max <= 4.3 and track.xv_var <= 6.:
            return 'robot'

    zhongan = '7a7c4ae5b8a4ef318b86c2836536fdea'
    if captcha_id == zhongan:
        if track.y_entropy >= 1.5 and track.t_tail <= 5.:
            return 'robot'

    banciyuan = 'c5610598ef686ccbdae067c42a9effb7'
    if captcha_id == banciyuan:
        if track.y_entropy <= 0.1 and track.x_entropy >= 3.5 and track.x_2bin_peEntropy234 >= 2.3:
            return 'robot'

    youxigou = '22ed6691b280f038f62336baa4785f5c'
    if captcha_id == youxigou:
        if track.strange_t >= 0.76 and 4.3 <= track.t_max <= 4.7 and track.x_peEntropy_345 <= 2.7:
            return 'robot'
        if track.x_peEntropy_345 <= 2.7 and track.y_entropy >= 1.4 and track.xv_var <= 3.9:
            return 'robot'
        if 1.9 < track.x_entropy < 3.1 and track.y_max_continous > 124 and track.t_peEntropy_2 > 1.35 and track.x_continous_scores > 12:
            return 'robot'
        if track.x_entropy <= 1.5 and 2.2 <= track.t_entropy <= 3.2 and track.x_continous_scores >= 7.:
            return 'robot'

    yinyuetai = 'e013d6b51053dde2b36b86470c1f156f'
    if captcha_id == yinyuetai:
        if track.y_entropy >= 2.2 and track.x_peEntropy_345 <= 3.:
            return 'robot'

    youxidaquan = '5f87024c5664f8631779d7fd2625fbcd'
    if captcha_id == youxidaquan:
        if track.x_entropy >= 1.6 and 2.1 <= track.t_tail <= 3.4:
            return 'robot'
        if track.x_entropy <= 1.2 and track.x_peEntropy_345 <= 2.6:
            return 'robot'

    xinshouka = '50b04f39a4906c9b8cf28a72fa11f488'
    if captcha_id == xinshouka:
        if track.strange_x <= 0.05 and track.t_entropy >= 4.7 and track.x_continous_scores >= 10:
            return 'robot'

    xiuse = 'c89a27ba59dfd5b94194f43e452137d2'
    if captcha_id == xiuse:
        if 5.4 <= track.t_max <= 5.7 and 5.1 <= track.t_tail <= 5.3 and 0.05 <= track.y_entropy <= 0.35:
            return 'robot'

    chengshizhongguo = '0f21fedd58caf38d6c414cdf29840b40'
    if captcha_id == chengshizhongguo:
        if track.y_entropy >= 2.2 and track.x_peEntropy_345 <= 3.:
            return 'robot'

    wangxiao = 'fb4a018d7f7dcb27ce99a4c1d06826d3'
    if captcha_id == wangxiao:
        if track.x_peEntropy_345 <= 2.7 and track.t_max < 4. and track.t_peEntropy_2 <= 0.9:
            return 'robot'

def track_check(requestattr):
    start = time.time()

    if check_rule(requestattr) == 'robot':
        return 'rule'

    category = RF.predict(requestattr)
    trackcheck_time = time.time()-start
    key = datetime.datetime.now().strftime("%Y-%m-%d-%H")+"-trackcheck"
    add_performancemessage(key, trackcheck_time)

def y_change(data):
    account = 0
    trackdata = data['trackdata']
    x = set()
    for i in trackdata:
        x.add(i[0])
        if i[1] == 0:
            account += 1
        if abs(i[1]) == 1:
            account += 0.3
    if ((account / 0.95) > len(trackdata)) and (account > 8) and (len(x) <= 5):
        return "y error"

def lchange_passtime_check(data):
    track_passtime = data['passtime']
    trackdata = data['trackdata']
    passtime = 0
    delta_x, delta_y = 0, 0
    move = 0
    for i in range(1, len(trackdata)):
        rec = trackdata[i]
        move += math.sqrt(rec[0] * rec[0] + rec[1] * rec[1])
        delta_x += rec[0]
        delta_y += rec[1]
        passtime += rec[2]

    distance = math.sqrt(delta_x * delta_x + delta_y * delta_y)
    lchange = float(move) / (distance + 0.00001)

    if passtime < 1000 and lchange < 1.010:
        return "v2ex attack"

def decode_response(challenge_id, string):
    if len(string) > 100:
        return 0
    shuzi = (1, 2, 5, 10, 50)
    chongfu = []
    key = {}
    count = 0
    for i in challenge_id:
        if i in chongfu:
            continue
        else:
            value = shuzi[count % 5]
            chongfu.append(i)
            count += 1
            key.update({i: value})
    res = 0
    for i in string:
        res += key.get(i, 0)
    return res

def translation(input):
    r = input
    rule_ip = '\d+\.\d+\.\d+\.\d'
    rule_time = '(?<=\[).+(?=\])'
    rule_challenge = '(?<=challenge=).+?(?=&)'
    rule_passtime = '(?<=passtime=).+?(?=&)'
    rule_imgload = '(?<=imgload=).+?(?=&)'
    rule_a = '(?<=a=).+?(?=&)'
    rule_userresponse = '(?<=userresponse=).+?(?=&)'

    ip = re.search(rule_ip, r).group()   
    time = re.findall(rule_time, r)[0]
    challenge = re.findall(rule_challenge, r)[0]
    passtime = re.findall(rule_passtime, r)[0]
    imgload = re.findall(rule_imgload, r)[0]
    userresponse=re.findall(rule_userresponse,r)[0]
    a = re.findall(rule_a, r) 
    a ="".join(tuple(a))
    trackdata = decodedatacompress(a)
    pyl = decode_response(challenge, userresponse)

    data = {
        'challenge' : challenge,
        'ip' : ip,
        'time' : time,
        'userresponse' : userresponse,
        'captcha_id' : '50b04f39a4906c9b8cf28a72fa11f488',
        'passtime' : passtime, 
        'trackdata' : trackdata, 
        'imgload' : imgload,
        'offset' : pyl,
                       }
    return data
