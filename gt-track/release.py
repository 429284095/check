#!coding:utf8
"""
先占个坑，能够在本地自动打tag，后续慢慢实现全自动化的。
todo :处理完毕后，再git push上服务器
"""

from git import *
from setup import *
# current_version = '15.01.05.1'

if __name__ == '__main__':
    repo = Repo('./')
    tags = repo.tags

    print 'last tag is:' + str(tags[-1])
    # for tag in tags:
    # print tag

    # 自动本地打tag
    repo.create_tag(current_version)

    tags = repo.tags
    print 'current tag is:' + str(tags[-1])

    # origin = repo.remotes.origin
    # print origin.url
    # origin.fetch()
