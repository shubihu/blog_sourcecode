---
title: Python Sort Algorithm
date: 2021-08-16 15:12:24
index_img: /img/article/Sort.jpg
categories:
    - Python
tags:
    - Python
comment: 'valine'
---
## Python常用排序算法
<!-- more -->
##### 快速排序
```
def quick_sort(array):
    if len(array) <= 1:  # 递归跳出条件
        return array
    pivot = array[0]
    left = [i for i in array[1:] if i < pivot]
    right = [i for i in array[1:] if i >= pivot]
    return quick_sort(left) + [pivot] + quick_sort(right)
```
##### 冒泡排序
```
def bubble_sort(array):
    for i in range(len(array) - 1):
        for j in range(len(array) - i -1): # 已排序好的部分不需再遍历
            if array[j] > array[j+1]:
                array[j], array[j+1] = array[j+1], array[j]
    return array
```
##### 桶排
```
def bucker_sort(array):
    result = []
    minVal, maxVal = min(array), max(array)
    bucket = [0] * (maxVal - minVal + 1)  # 所需的桶数
    for i in array:
        bucket[i - minVal] += 1     # 每个数字出现的次数
    for i in range(len(bucket)):
        if bucket[i]:
            result += [i + minVal] * bucket[i]
    return result
```
