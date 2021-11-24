---
title: MongoDB基础
date: 2021-10-18 09:54:34
index_img: /img/article/mongodb.jpg
categories:
    - 数据库
tags:
    - MongoDB
comment: 'valine'
---
## MongoDB 基础命令
<!-- more -->
##### 启动本地服务端
进入mongodb bin目录下打开命令行执行 mongod 启动服务端(存储引擎参数 --storageEngine=mmapv1)
```
.\mongod.exe --storageEngine=mmapv1 --dbpath E:\Desktop\Java\JavaSoftware\mongoDB\data\
```
##### 启动本地客户端
进入mongodb bin目录下打开命令行执行 mongo 启动客户端
```
.\mongo.exe
```

##### 查看数据库
```
show dbs
```
##### 切换数据库（无需新建，直接引用）
```
use demo
```
##### 插入数据(以创建一个雇员信息表为例)
```
db.Employee.save({code:'E01', name:'Jacky'})
```
##### 查看数据 
```
show collections
```
##### 查找数据 
```
db.Employee.find()
```
##### 格式化输出查找数据 
```
db.Employee.find().pretty()
```
##### 添加不同格式数据 
```
db.Employee.save({code:'E02', name:'Jim', email:'test@email.com'})
```

启动mongodb时，提示Unclean shutdown detected mongodb，解决方法:
```
.\mongod.exe --repair --dbpath E:\Desktop\Java\JavaSoftware\mongoDB\data\
```