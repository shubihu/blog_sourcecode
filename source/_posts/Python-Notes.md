---
title: Python Notes
date: 2021-02-02 14:23:30
index_img: /img/article/python-hero.jpg
categories:
    - Python
tags:
    - Python
comment: 'valine'
---
## Python学习随笔
<!-- more -->
### 字典转dataframe
不定义列名时：pd.DataFrame.from_dict(data, orient='index')
定义列名时：pd.DataFrame.from_dict(data, orient='index', columns=['A', 'B', 'C', 'D'])
### pandas筛选
#### 选取某列值等于某些值的行用 == ，不等于用 ！= ，data.loc[data['a'] == 'one']
#### 选取某列值是否是某一类型的数值用 isin ,取反用 ~
```language
data.loc[data['a'].isin(['one', 'two'])]
data.loc[~data['a'].isin(['one', 'two'])]
```
#### 多种条件的选取用 & , data.loc[(data['a'] == 'one') & (data['b'] == 'two')]
```language
np.linspace(start, stop, num) ##参数为起点，终点，点数，num默认为50
```
### 把某列值设为index，df.set_index('columns')  (df.reset_index()重置索引)
df中merge函数按 键 合并，concat函数按 轴 合并

### 按键 (key) 合并可以分「单键合并」和「多键合并」
### 单键合并：
```language
pd.merge(df1, df2, how=s, on=c ) ##c 是 df1 和 df2 共有的一栏，合并方式 (how=s) 有四种：
```
###### 左连接 (left)：合并之后显示 df1 的所有行
###### 右连接 (right)：合并之后显示 df2 的所有行
###### 外连接 (outer)：合并 df1 和 df2 共有的所有行
###### 内连接 (inner)：只保留两个表中公共部分的信息 (默认情况)
###### 多键合并(俩组数据均有该列)
```language
pd.merge( df1, df2, how=s, on=c )  ## c 是多栏（如一个列表
```
### 多键合并(两组数据不同的列名）
```language
pd.merge(df1, df2, left_on = 'key1', right_on = 'key2')
```
### 插入列：除在最右侧插入用标签直接创建外，其他列用.insert()方法进行插入，比如table.insert(0,'date',date)

当 df1 和 df2 有两个相同的列 (Asset 和 Instrument) 时，单单只对一列 (Asset) 做合并产出的 DataFrame 会有另一列 (Instrument) 重复的名称。
这时 merge 函数给重复的名称加个后缀 _x, _y，也可以设定 suffixes 来改后缀
```language
pd.concat([df1,df2], axis=0, ignore_index=True)  # 默认axis=0（行连接）
```

### 列索引 → 行索引，用 stack 函数;行索引 → 列索引，用 unstack 函数
### 数据透视：
用 pivot 函数将「一张长表」变「多张宽表」，
用 melt 函数将「多张宽表」变「一张长表」  # 函数 melt 实际是将「源表」转化成 id-variable 类型的 DataFrame
```
data_pivot = data.pivot(index='Date',columns='Symbol',values='Adj Close') #若不设置value参数，剩下的列都用来透视
melted_data = pd.melt(data, id_vars=['Date','Symbol'])
sorted(set(list), key=list.index)  ## 消除重复元素不改变原始数据顺序
sorted(dict.items(),key=lambda x:x[1],reverse=True)  ## 对字典按值反向排序（x[0]按键排序）
```

### pandas 删除列
```
df = df.drop(['c', 'e'], axis=1)
#或者
df.drop(['c', 'e'], axis=1, inplace=True)
```
### 对行 z-score 标准化
```
df.apply(lambda x: (x - np.mean(x)) / (np.std(x,ddof=1)), axis=1)
```
### 对 Majority protein IDs 列转成多行
```
df = df[~df['Majority protein IDs'].str.contains('CON|REV', regex=True)]
df = df.drop('Majority protein IDs', axis=1).join(df['Majority protein IDs'].str.split(';', expand=True).stack().reset_index(level=1, drop=True).rename('Majority protein IDs'))

def ab(df): 
    return ';'.join(df.values)
newcolumns = df_merge.columns.tolist()
newcolumns.remove('Majority protein IDs')
newdf = df_merge.groupby(newcolumns)['Majority protein IDs'].apply(ab)   ## 多行合并一行
```
### for、while循环中的else扩展用法
else中的程序只在一种条件下执行，即循环正常遍历所有内容或者由于条件不成立而结束循环，没有因break或者return而退出循环。continue对else没影响
```language
for i in range(10):
    if i==5:
        break
    print('i=',i,end=',')
else:
    print('success')#不输出   在for循环中含有break时则直接终止循环，并不会执行else子句。

for i in range(10):
    if i==5:
        continue
    print('i=',i,end=',')
else:
    print('success')#输出
```
### 展平嵌套列表
```language
newlist = [item for items in newlist for item in items]
#或者您可以像这样从chain中使用itertools
from itertools import chain
newlist = list(chain(*newlist))
#或者您可以使用chain.from_iterable，其中无需解压缩列表
from itertools import chain
newlist = list(chain.from_iterable(newlist)) #效率更高
```
### 生成requirements.txt
```language
pipreqs ./ --encoding=utf-8 --force
```
### 单例
```language
class Singleton(object):
    __instance = None

    def __new__(cls, age, name):
        if not cls.__instance:
            cls.__instance = object.__new__(cls)
        return cls.__instance
```
