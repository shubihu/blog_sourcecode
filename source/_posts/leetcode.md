---
title: LeetCode
date: 2021-12-28 15:38:10
index_img: /img/article/leetcode.png
categories:
    - leetcode
tags:
    - Python
comment: 'valine'
---
## 力扣笔记
<!-- more -->
##### 题目描述：
给定一个字符串 s ，其中包含字母顺序打乱的用英文单词表示的若干数字（0-9）。按 升序 返回原始的数字。例如：输入：s = "owoztneoer"，输出："012"。
原题地址：https://leetcode-cn.com/problems/reconstruct-original-digits-from-english/
```
### 我的方案，使用了递归，，但依然很惨，没通过力扣的检验（超出时间限制）

def originalDigits(s):
    en_num = {'zero':0, 'one':1,'two':2, 'three':3,'four':4,'five':5,'six':6,'seven':7,
    'eight':8,'nine':9}

    ss = []
    for k, v in en_num.items():
        l = len(k)
        c = 0
        for i in k:
            if i in s:
                c += 1

        if c == l:
            ss.append(str(v))
            for j in k:
                s = s.replace(j, '', 1)
    if s:
        ss.append(originalDigits(s))
        ss.sort()
        return ''.join(ss)
    else:
        ss.sort()
        return ''.join(ss)
```
```
## 力扣方案

class Solution:
    def originalDigits(self, s: str) -> str:
        c = Counter(s)

        cnt = [0] * 10
        cnt[0] = c["z"]
        cnt[2] = c["w"]
        cnt[4] = c["u"]
        cnt[6] = c["x"]
        cnt[8] = c["g"]

        cnt[3] = c["h"] - cnt[8]
        cnt[5] = c["f"] - cnt[4]
        cnt[7] = c["s"] - cnt[6]
        
        cnt[1] = c["o"] - cnt[0] - cnt[2] - cnt[4]

        cnt[9] = c["i"] - cnt[5] - cnt[6] - cnt[8]

        return "".join(str(x) * cnt[x] for x in range(10))
```
如果单看这个代码的话我依然看不懂，还是要看解释，解释在上面的网址里都有。反正这个题确实挺考验智商的吧，我这智商还是洗洗睡了。

##### 题目描述：
无重复字符串的排列组合。编写一种方法，计算某字符串的所有排列组合，字符串每个字符均不相同。
例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
原题地址：https://leetcode-cn.com/problems/permutation-i-lcci/
python的itertools包中的permutations函数可以实现。
```
### itertools.permutations函数源码如下

def permutation(iterable, r=None):
    #  r:length permutations of elements in the iterable
    # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
    # permutations(range(3)) --> 012 021 102 120 201 210
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r
    if r > n:
        return
    indices = list(range(n))
    cycles = list(range(n, n-r, -1))
    yield tuple(pool[i] for i in indices[:r])
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i+1:] + indices[i:i+1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:r])
                break
        else:
            return
```
源码还是挺复杂的，，，，，
```
### 力扣方法，也是递归

def permutation(S):
        if len(S)==1:
            return [S]
        ans=[]
        for i in range(len(S)):
            s=S[:i]+S[i+1:]              
            for string in permutation(s):
                ans.append(S[i]+string)
        return ans
```

##### 题目描述：
给定两个单词 word1 和 word2，请计算出将 word1 转换成 word2 所使用的最少操作数。


##### 题目描述：
快速排序
```
arr = [1, 4, 5, 12, 32, 198, 2, 3, 15, 112, 132]

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    tmp = arr[0]
    left = [i for i in arr[1:] if i < tmp]
    right = [i for i in arr[1:] if i > tmp]
    return quick_sort(left) + [tmp] + quick_sort(right)

print(quick_sort(arr))
```

##### 题目描述：
归并排序应用：将两个有序数组合并成一个有序数组
```
a = [1,3,5,7]
b = [2,6,8,9]

def merge(left, right):
    '''合并操作，将两个有序数组left[]和right[]合并成一个大的有序数组'''
    #left与right的下标指针
    l, r = 0, 0
    result = []
    while l<len(left) and r<len(right):
        if left[l] < right[r]:
            result.append(left[l])
            l += 1
        else:
            result.append(right[r])
            r += 1
    result += left[l:]
    result += right[r:]
    return result

print(merge(a, b))
```
