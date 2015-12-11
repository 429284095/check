from checktools import  *
from gttrack.feature import RequestAttr

Nginx = '222.82.208.218 -  [02/Dec/2015:11:31:02 +0800]  GET /ajax.php?challenge=5a389b0998e29fb80e5a1a8877efec8db2&userresponse=ff9ffffff2dd8c1&passtime=998&imgload=249&a=K-.--,.(!!Jtstyy(yytyy(~y(sststsssssssssy!)ty!)yz(sssttsssssss(!!($)68(9(9(8(9(899L89L9L9LL9L8(9(9(8(9(8(9(9(8(98M8aL$,T&callback=geetest_1449027070044 HTTP/1.1 - 200 113 http://fahao.eeyy.com/fahao/78103.html Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/31.0.1650.63 Safari/537.36 0.024'

data = translation(Nginx)
redata = RequestAttr(data)
print redata

result = [
	y_change(data),
	lchange_passtime_check(data),
	check_rule(redata)
	]
print result







