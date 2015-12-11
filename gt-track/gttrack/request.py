#!coding:utf8
from __future__ import division
import calcu
import preprocess
import numpy as np


class Feature(object):
    """
    Feature is a class for lazy calcu feature of track by define calcu function
    """

    def __init__(self, track, calcu_func=None, show=True, group='default', *args, **kwargs):
        self.track = track
        self.calcu_func = calcu_func  # the pointer of the calculator function
        self.show = show  # control weather this feature is display or output to the csv file
        self.group = group

    def calcu(self, *args, **kwargs):
        #print 'calculate'
        if not hasattr(self, 'res'):
            self.res = self.calcu_func(self.track, *args, **kwargs)
        return self.res


class Request(object):
    base_data = ('ip','referer','new_user','request_time','imgload','drag_count','res','challenge','captcha_id')
    def __init__(self, req={}):
        """
        The parent Object of Track
        Init the track object,and convert the track
        from  [[x,y,t]..]
        to [x,..],[y,..],[t,..]
        and make sure all the value type is float type
        :param req:
        :type req:
        :return:
        :rtype:
        """
        self.fields = []
        self.dictdata = {}
        self.trackdata_origin = req.get('trackdata',[])
        self.ua = req.get('UA', "")
        self.captcha_id = req.get('captcha_id', "")
        self.trackdata = self.preprocess_data(self.trackdata_origin)

        self.valid = self.check(self.trackdata)
        for i in self.base_data:
            setattr(self,i,req.get(i))

        if self.valid:
            self.trackdata = np.array(self.trackdata,dtype = np.float)
            self.x = self.trackdata[:, 0]
            self.y = self.trackdata[:, 1]
            self.t = self.trackdata[:, 2]

    def preprocess_data(self,trackdata):
        trackdata = preprocess.filter(trackdata)
        return trackdata
        

    def check(self, trackdata):
        """
        1. the length check
        2. the time check
        :param trackdata:
        :type trackdata:
        :return:
        :rtype:
        """
        if len(trackdata) < 5:
            return False

        for i in trackdata:
            if i[2] <= 0:
                return False
                
        if sum([i[0] for i in trackdata]) == 0:
            return False

        return True

   
    def __getattribute__(self, name, *args, **kwargs):
        """
        overload the __getattribute__ function
        and return the calculate result feature
        :param name:
        :type name:
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        attribute = object.__getattribute__(self, name)
        if isinstance(attribute, Feature):
            return getattr(attribute, 'calcu')(*args, **kwargs)
        else:
            return attribute

    def __str__(self):
        """
        convert all the attributes in to csv heads,reload the __str__ function
        :return:
        :rtype:
        """
        fields = self.get_all_field()
        return ','.join([str(getattr(self, i)) for i in fields])

        # return json.dumps(res)

    def getdict(self, features):
        return dict([[i, getattr(self, i)] for i in features])

    def get_all_field(self, feature_grp=['default']):
        """
        add all the feature name(the key not the value)
        which is need to show to the field list
        and return
        filter the feature elements
        :return:
        :rtype:
        """
        if self.fields:
            return self.fields

        fields = []
        for i in dir(self):
            attribute = object.__getattribute__(self, i)
            if isinstance(attribute, Feature) and attribute.show:
                if attribute.group in feature_grp:
                    fields.append(i)
        self.fields = fields
        return fields

    def get_field_name(self):
        """
        join all the fields together to output csv
        :return:
        :rtype:
        """
        return ','.join(self.get_all_field())

    def get_all_result(self):
        res = {}
        for i in self.get_all_field():
            res.update({i: getattr(self, i)})
        return res

