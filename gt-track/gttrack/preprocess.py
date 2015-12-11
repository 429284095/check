def filter(trackdata):        
    if len(trackdata) < 6:         #if sample point is a small number,we'll drop it
        return []
    try:
        if len([i for i in trackdata if i[2] < 0]) > 0:
            return []
    except:
        return []

    return trackdata[1:-1]


def merge_data(trackdata, step):
    new_track = []
    sum_block_dist = 0
    index_start, index_end = (0,1)
    for i in trackdata:
        sum_block_dist += i[0]
        if abs(sum_block_dist) > 3:
            sum_x = sum([trackdata[idx][0] for idx in range(index_start, index_end)])
            sum_y = sum([trackdata[idx][1] for idx in range(index_start, index_end)])
            sum_t = sum([trackdata[idx][2] for idx in range(index_start, index_end)])
            new_track.append([sum_x, sum_y, sum_t])

            index_start, index_end = index_end, index_end + 1
            sum_block_dist = 0
        else:
            index_end += 1
    return new_track