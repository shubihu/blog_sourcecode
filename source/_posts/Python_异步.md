---
title: Python 异步
date: 2021-08-11 16:52:10
index_img: /img/article/async-sync.png
categories:
    - Python
tags:
    - Python
comment: 'valine'
---
## Python中异步、同步、多进程及多线程的比较
<!-- more -->
```
from urllib import request
from urllib import parse
from urllib.request import urlopen
import json
import pandas as pd
# 用于多进程
from multiprocessing import Process
# 用于多线程
from threading import Thread
# 用于协程+异步
import aiohttp
import asyncio
"""
aiohttp:异步发送POST请求
"""
async def city_rule_asy():
    data = {"key": ""}
    myPostUrl = "http://api.chinadatapay.com/government/traffic/2299"
    async with aiohttp.ClientSession() as session:
        async with session.post(myPostUrl, data=data) as res:
            # print(res.status)
            return json.loads(await res.text())
def run():
    tasks = []
    for i in range(5):
        task = asyncio.ensure_future(city_rule_asy())
        tasks.append(task)
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(asyncio.gather(*tasks))
    with open('city_rules.txt', 'a+') as fw:
        for i in response['data']:
           for j in i['cities']:
                fw.write(f"{j['city']}\t{j['engine']}\t{j['prefix']}\t{j['vin']}\t{j['model']}\n")
#### ============================================ ###
def city_rule():
    myPostUrl = "http://api.chinadatapay.com/government/traffic/2299"
    data = {"key": ""}
    params = parse.urlencode(data).encode('utf-8')  # 提交类型不能为str，需要为byte类型
    req = request.Request(myPostUrl, params)
    response = json.loads(urlopen(req).read().decode())
    with open('city_rules.txt', 'a+') as fw:
        for i in response['data']:
           for j in i['cities']:
                fw.write(f"{j['city']}\t{j['engine']}\t{j['prefix']}\t{j['vin']}\t{j['model']}\n")
## 单进程单线程同步
def single_process():
    for i in range(5):
        city_rule()
# 多进程并行
def mul_process():
    processes = []
    for i in range(5):
        p = Process(target=city_rule, args=())     # 一个参数 args=(prameter,)
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
# 多线程并发
def mul_thead():
    threads = []
    for i in range(5):
        t = Thread(target=city_rule, args=())
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
if __name__ == '__main__':
    # 异步
    run()
    # 同步
    single_process()
    # 多进程
    mul_process()
    #多线程
    mul_thead()
```