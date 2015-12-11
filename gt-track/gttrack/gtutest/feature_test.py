#!coding=utf8
"""
简单的单元测试
"""

import unittest
from gttrack.gtutest import data


class TestGtTrack(unittest.TestCase):
    def setUp(self):
        self.track_data = data.track_data
        pass

    def tearDown(self):
        # print 'tearDown'
        # print self.case_doc
        pass

    def test_passtime(self):
        """
        测试检pass time
        :return:
        """
        server_pass_time = sum((i[2] for i in self.track_data))
        self.assertEquals(server_pass_time, data.validate_pass_time)