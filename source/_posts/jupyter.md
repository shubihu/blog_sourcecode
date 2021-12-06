---
title: JupyterNotebook远程云服务器
date: 2021-11-30 16:08:28
index_img: /img/article/jupyter.png
categories:
    - Python
tags:
    - jupyter
comment: 'valine'
---
## 搭建Jupyter Notebook远程云服务器
<!-- more -->
##### 安装Jupyter
```
pip install Jupyter
jupyter notebook --generate-config
```
##### 设置密码用于设置服务器配置，以及登录Jupyter。打开Python终端，输入以下：
```
In [1]: from IPython.lib import passwd
In [2]: passwd()
Enter password: 
Verify password: 
Out[2]: '这里是密码'
```
##### 设置服务器配置文件
```
vim ~/.jupyter/jupyter_notebook_config.py
```
在末尾增加以下几行配置信息
```
c.NotebookApp.ip = '*' #所有绑定服务器的IP都能访问，若想只在特定ip访问，输入ip地址即可
c.NotebookApp.port = 8888 #将端口设置为自己喜欢的吧，默认是8888
c.NotebookApp.open_browser = False #我们并不想在服务器上直接打开Jupyter Notebook，所以设置成False
c.NotebookApp.notebook_dir = '/root/jupyter_projects' #这里是设置Jupyter的根目录，若不设置将默认root的根目录，不安全
c.NotebookApp.allow_root = True # 为了安全，Jupyter默认不允许以root权限启动jupyter 
```
##### 启动Jupyter 远程服务器
```
jupyter notebook
```
至此，Jupyter远程服务器以搭建完毕。在本地浏览器上，输入 ip地址:8888，将会打开远程Jupyter。接下来就可以像在本地一样使用服务器上的Jupyter。
