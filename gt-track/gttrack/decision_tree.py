def decision_tree(trackdata):
    if trackdata.x_entropy <= 0.3 and trackdata.y_entropy <= 0.2:
        return 'robot'
    if trackdata.t_entropy <= 1.0 and trackdata.y_entropy <= 0.1:
        return 'robot'
    if trackdata.y_entropy >= 1.7:
        return 'robot'
    # return 'people'
    if trackdata.v_median <= 0.157728:
        if trackdata.vx_min <= 0.049383:
            if trackdata.nonzero_percent_y <= 0:
                if trackdata.w0 <= 0.785398:
                    if trackdata.mouseup_time <= 5:
                        if trackdata.mouseup_time <= 4: return 'people'
                        else:
                            if trackdata.w_acf_2 <= 0.041459: return 'robot'
                            else: return 'people'
                    else:
                        if trackdata.t_acf_1 <= 0.430963: return 'people'
                        else:
                            if trackdata.x_acf_2 <= -0.00906: return 'robot'
                            else: return 'people'
                else:
                    if trackdata.wn <= 0.785398:
                        if trackdata.mouseup_time <= 120:
                            if trackdata.last_change_x <= 272:
                                if trackdata.t_acf_1 <= 0.026371: return 'robot'
                                else: return 'people'
                            else: return 'robot'
                        else:
                            if trackdata.v_acf_1 <= 0.171086:
                                if trackdata.v_acf_2 <= 0.082895: return 'people'
                                else: return 'robot'
                            else: return 'people'
                    else: return 'people'
            else:
                if trackdata.nonzero_percent_y <= 0.847826:
                    if trackdata.mouseup_time <= 92: return 'people'
                    else:
                        if trackdata.t_acf_2 <= 0.520637: return 'people'
                        else:
                            if trackdata.x_acf_1 <= 0.005611:
                                if trackdata.last_change_x <= 273: return 'robot'
                                else: return 'people'
                            else: return 'people'
                else:
                    if trackdata.x_acf_2 <= 0.71629:
                        if trackdata.vx_min <= 0.03367: return 'people'
                        else:
                            if trackdata.last_change_x <= 277: return 'people'
                            else: return 'robot'
                    else: return 'robot'
        else:
            if trackdata.last_change_x <= 271: return 'people'
            else:
                if trackdata.y_acf_2 <= 0.014814: return 'robot'
                else:
                    if trackdata.vx_median <= 0.114379: return 'robot'
                    else: return 'people'
    else:
        if trackdata.x_acf_1 <= 0.261018:
            if trackdata.w0 <= 0:
                if trackdata.last_change_x <= 243: return 'people'
                else:
                    if trackdata.last_change_x <= 349:
                        if trackdata.nonzero_percent_y <= 0.16568:
                            if trackdata.mouseup_time <= 513:
                                if trackdata.sinangel <= 0.968039:
                                    if trackdata.x_acf_2 <= 0.00942:
                                        if trackdata.x_acf_2 <= -0.039607: return 'people'
                                        else:
                                            if trackdata.x_acf_1 <= -0.010526: return 'robot'
                                            else:
                                                if trackdata.same_v_count <= 5: return 'people'
                                                else: return 'robot'
                                    else: return 'people'
                                else:
                                    if trackdata.time_last_3_ratio <= 0.291498:
                                        if trackdata.vy_acf_1 <= 0.017682:
                                            if trackdata.same_v_count <= 1:
                                                if trackdata.t_acf_1 <= -0.024514: return 'robot'
                                                else:
                                                    if trackdata.t_acf_2 <= 0.006065: return 'robot'
                                                    else: return 'people'
                                            else:
                                                if trackdata.mousedown_xpos <= 14: return 'robot'
                                                else:
                                                    if trackdata.v_median <= 0.292857:
                                                        if trackdata.x_acf_1 <= 0.027859: return 'robot'
                                                        else:
                                                            if trackdata.t_acf_1 <= -0.08992: return 'robot'
                                                            else:
                                                                if trackdata.vx_min <= 0.032258: return 'people'
                                                                else: return 'robot'
                                                    else:
                                                        if trackdata.ratio_v1v2 <= 0.252553: return 'robot'
                                                        else: return 'people'
                                        else:
                                            if trackdata.x_acf_1 <= 0.010015: return 'robot'
                                            else: return 'people'
                                    else:
                                        if trackdata.v_min <= 0.000707: return 'robot'
                                        else:
                                            if trackdata.sinangel <= 0.970772: return 'robot'
                                            else: return 'people'
                            else: return 'people'
                        else:
                            if trackdata.mouseup_time <= 105:
                                if trackdata.vx_acf_2 <= 0.520708: return 'people'
                                else:
                                    if trackdata.mousedown_xpos <= 42:
                                        if trackdata.time_last_3_ratio <= 0.044459: return 'robot'
                                        else: return 'people'
                                    else: return 'robot'
                            else:
                                if trackdata.v_min <= 0.011136: return 'people'
                                else:
                                    if trackdata.x_acf_2 <= 0.012193:
                                        if trackdata.vx_acf_2 <= -0.133548:
                                            if trackdata.nonzero_percent_y <= 0.203593: return 'robot'
                                            else: return 'people'
                                        else: return 'robot'
                                    else:
                                        if trackdata.time_last_3_ratio <= 0.081001:
                                            if trackdata.w_acf_1 <= 0.324367: return 'robot'
                                            else:
                                                if trackdata.v_min <= 0.047619: return 'people'
                                                else: return 'robot'
                                        else: return 'people'
                    else: return 'people'
            else:
                if trackdata.vx_acf_2 <= 0.745871:
                    if trackdata.mousedown_xpos <= 0: return 'robot'
                    else:
                        if trackdata.mouseup_time <= 199: return 'people'
                        else:
                            if trackdata.x_acf_1 <= 0.000846:
                                if trackdata.vx_median <= 0.25266:
                                    if trackdata.mouseup_time <= 495:
                                        if trackdata.y_acf_1 <= -0.062554:
                                            if trackdata.nonzero_percent_y <= 0.689655:
                                                if trackdata.vx_acf_1 <= 0.199095:
                                                    if trackdata.same_v_count <= 4: return 'people'
                                                    else: return 'robot'
                                                else: return 'robot'
                                            else: return 'robot'
                                        else:
                                            if trackdata.same_v_count <= 5:
                                                if trackdata.vx_min <= 0.008448:
                                                    if trackdata.same_v_count <= 3:
                                                        if trackdata.sinangel <= 0.98859: return 'people'
                                                        else: return 'robot'
                                                    else:
                                                        if trackdata.time_last_3_ratio <= 0.325203: return 'robot'
                                                        else: return 'people'
                                                else: return 'robot'
                                            else: return 'robot'
                                    else:
                                        if trackdata.same_v_count <= 6: return 'people'
                                        else: return 'robot'
                                else: return 'people'
                            else:
                                if trackdata.sinangel <= 0.900576: return 'robot'
                                else: return 'people'
                else: return 'robot'
        else:
            if trackdata.mousedown_xpos <= 1:
                if trackdata.v_median <= 0.551084:
                    if trackdata.y_acf_1 <= 0.245532:
                        if trackdata.time_last_3_ratio <= 0.107822: return 'robot'
                        else: return 'people'
                    else: return 'people'
                else: return 'robot'
            else:
                if trackdata.mousedown_xpos <= 59:
                    if trackdata.vx_median <= 0.072078:
                        if trackdata.vx_acf_2 <= 0.667491: return 'people'
                        else: return 'robot'
                    else:
                        if trackdata.sinangel <= 0.890205:
                            if trackdata.time_last_3_ratio <= 0.052277:
                                if trackdata.time_last_3_ratio <= 0.043887:
                                    if trackdata.ratio_v1v2 <= 5.703567:
                                        if trackdata.same_v_count <= 4: return 'people'
                                        else:
                                            if trackdata.nonzero_percent_y <= 0.545455: return 'robot'
                                            else: return 'people'
                                    else: return 'robot'
                                else:
                                    if trackdata.vx_median <= 0.262275:
                                        if trackdata.t_acf_1 <= 0.010448: return 'robot'
                                        else: return 'people'
                                    else: return 'robot'
                            else: return 'people'
                        else:
                            if trackdata.wn <= 1.961403:
                                if trackdata.w0 <= 1.132196:
                                    if trackdata.mouseup_time <= 14: return 'people'
                                    else:
                                        if trackdata.x_acf_2 <= 0.332278:
                                            if trackdata.t_acf_1 <= -0.031676:
                                                if trackdata.nonzero_percent_y <= 0.175182:
                                                    if trackdata.vx_min <= 0.021053:
                                                        if trackdata.vy_acf_2 <= 0.080883: return 'people'
                                                        else:
                                                            if trackdata.mouseup_time <= 332: return 'robot'
                                                            else: return 'people'
                                                    else:
                                                        if trackdata.last_change_x <= 269: return 'people'
                                                        else:
                                                            if trackdata.vx_median <= 0.491667: return 'robot'
                                                            else: return 'people'
                                                else:
                                                    if trackdata.last_change_x <= 259: return 'people'
                                                    else:
                                                        if trackdata.vx_median <= 0.196774:
                                                            if trackdata.t_acf_2 <= -0.071155: return 'robot'
                                                            else:
                                                                if trackdata.vy_acf_2 <= 0.130867: return 'people'
                                                                else: return 'robot'
                                                        else: return 'people'
                                            else:
                                                if trackdata.last_change_x <= 256: return 'people'
                                                else:
                                                    if trackdata.ratio_v1v2 <= 0.268691:
                                                        if trackdata.y_acf_2 <= 0.228091:
                                                            if trackdata.vx_acf_1 <= 0.398422: return 'people'
                                                            else:
                                                                if trackdata.v_acf_2 <= -0.053859: return 'robot'
                                                                else:
                                                                    if trackdata.sinangel <= 0.996947: return 'people'
                                                                    else:
                                                                        if trackdata.v_min <= 0.039604:
                                                                            if trackdata.wn <= 0.466842:
                                                                                if trackdata.wn <= 0.026188:
                                                                                    if trackdata.ratio_v1v2 <= 0.232708:
                                                                                        if trackdata.vy_acf_2 <= -0.065059: return 'robot'
                                                                                        else: return 'people'
                                                                                    else: return 'robot'
                                                                                else: return 'people'
                                                                            else: return 'robot'
                                                                        else: return 'people'
                                                        else:
                                                            if trackdata.last_change_x <= 260: return 'people'
                                                            else: return 'robot'
                                                    else: return 'people'
                                        else: return 'people'
                                else:
                                    if trackdata.wn <= 0.858117: return 'people'
                                    else:
                                        if trackdata.ratio_v1v2 <= 0.485218:
                                            if trackdata.last_change_x <= 280: return 'people'
                                            else:
                                                if trackdata.v_acf_2 <= 0.689612:
                                                    if trackdata.vx_median <= 0.256579:
                                                        if trackdata.time_last_3_ratio <= 0.086655: return 'robot'
                                                        else:
                                                            if trackdata.vx_median <= 0.180851: return 'people'
                                                            else: return 'robot'
                                                    else:
                                                        if trackdata.v_acf_2 <= 0.644918: return 'people'
                                                        else: return 'robot'
                                                else: return 'people'
                                        else:
                                            if trackdata.vy_acf_2 <= 0.42505: return 'people'
                                            else:
                                                if trackdata.x_acf_2 <= 0.389286: return 'robot'
                                                else: return 'people'
                            else:
                                if trackdata.wn <= 2.089942:
                                    if trackdata.w0 <= 0.942: return 'people'
                                    else: return 'robot'
                                else:
                                    if trackdata.w_acf_2 <= -0.015441:
                                        if trackdata.w0 <= 1.954266:
                                            if trackdata.last_change_x <= 221:
                                                if trackdata.t_acf_2 <= 0.05683: return 'robot'
                                                else: return 'people'
                                            else:
                                                if trackdata.w_acf_2 <= -0.017099: return 'people'
                                                else: return 'robot'
                                        else: return 'robot'
                                    else: return 'people'
                else: return 'robot'
