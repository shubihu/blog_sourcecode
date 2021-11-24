---
title: Linux One Line Command
date: 2021-08-13 14:08:56
index_img: /img/article/linux.png
categories:
    - Linux
tags:
    - Linux
comment: 'valine'
---
## Linux 常用命令
<!-- more -->
```
sh -n ##判断是否有语法错误
sh -x ##执行详细过程
## 修改目录颜色
dircolors -p > ~/.dircolors   ## 编辑 ~/.dircolors 修改
## 忽略大小写
echo 'set completion-ignore-case on' > ~/.inputrc
```
##### Linux 两个文件求交集、并集、差集
```
sort a.txt b.txt | uniq -d   ### 交集
sort a.txt b.txt | uniq   ###并集 
sort a.txt b.txt b.txt | uniq -u  ## 差集 a-b
sort b.txt a.txt a.txt | uniq -u  ## 差集 b-a
```
使用sort可以将文件进行排序，可以使用sort后面的玲玲，例如 -n 按照数字格式排序，例如 -i 忽略大小写，例如使用-r 为逆序输出等
uniq为删除文件中重复的行，得到文件中唯一的行，后面的命令 -d 表示的是输出出现次数大于1的内容 -u表示的是输出出现次数为1的内容，那么对于上述的求交集并集差集的命令做如下的解释：
```
sort a.txt b.txt | uniq -d #将a.txt b.txt文件进行排序，uniq使得两个文件中的内容为唯一的，使用-d输出两个文件中次数大于1的内容，即是得到交集
sort a.txt b.txt | uniq  #将a.txt b.txt文件进行排序，uniq使得两个文件中的内容为唯一的，即可得到两个文件的并集
sort a.txt b.txt b.txt | uniq -u #将两个文件排序，最后输出a.txt b.txt b.txt文件中只出现过一次的内容，因为有两个b.txt所以只会输出只在a.txt出现过一次的内容，即是a.txt-b.txt差集
#对于b.txt-a.txt为同理
```
##### grep 命令是常用的搜索文本内容的，要找交集，如下即可：
```
grep -F -f a.txt b.txt | sort | uniq
```
##### 差集:
```
grep -F -v -f a.txt b.txt | sort | uniq
grep -F -v -f b.txt a.txt | sort | uniq
#第一行结果为b-a；第二行为a-b。注意顺序很重要
```
##### 根据id提取fastq
```
grep -f id -A 3 BC01.fq > test.fq   ### -f 参数为ID文件
```
##### 批量重命名文件
```
#只更改户后缀
rename 's/.txt/.log/' *.txt   #### 把txt后缀改为log
#小写变大写
for i in `ls`;do mv -f $i `echo $i | tr a-z A-Z`;done
for i in `ls`;do mv -f $i `echo $i | sed 's/..../..../'`;done  ##使用sed替换q
rename 's/small/large/' image_*.png
```
##### 删除空行
```
sed -i '/^$/d' file
grep -v '^$' file   或  sed '/^$/d' file 或 sed -n '/./p' file
awk '/./{print}' file 或  tr -s 'n'
#删除最后一列
sed -r -e 's/\t[^\t]*$//g' file   
```
##### 统计文件大小
```
du -sh * 或者 du -h --max-depth=1  或 du -sh * | grep [GM] | sort 提取G 和 M的文件并排序
```
##### 计算reads数
```
expr $(wc -l < *.fastq) / 4
expr $(zcat test/1.R1.fq.gz | wc -l) / 4
```
##### fastq 转 fasta
```
awk '{if(NR%4 == 1){print ">" substr($0, 2)}}{if(NR%4 == 2){print}}' xx.fastq >xx.fasta
awk '{if(NR%4 == 1){print ">" "'$j'""_"NR}}{if(NR%4 == 2){print}}'    #   "'$j'" awk中引用外部变量
```
```
sort -k1,1V -k2,2n file   ## V 参数忽略第一列中的文本按数字排序
awk '$1 ~ /chr1|chr3/' file ## 第一列匹配chr1或chr3
awk 'NR > 3' file ## 取出第四行以后
sed -n '20,50p' file # 取出20到50行
```
```
paste file1 file2  # 横向拼接文件，拼接前可用dos2unix转换文件类型
```
```
'%' 从后向前删除, '#' 从前向后删除
sed 替换每行最后一个匹配的字符
sed 's/\(.*\)src_str\(.*\)/\1dst_str\2/'  yourfile   ##  src_str：要匹配的字符  dst_str: 要替换的字符
```
```
biom=${i##*/}    #返回 / 后的字符
biom=${i%/*}     #返回最后 / 前的字符
```
```
ls -ld 列出文件全路径
```
##### 使用 wget 完成批量下载
如果想下载一个网站上目录中的所有文件, 我需要执行一长串wget命令, 但这样做会更好:
```
wget -nd -r -l1 --no-parent http://www.foo.com/mp3/
```
这条命令可以执行的很好, 但有时会下载像 index.@xx 这样一些我不想要的文件. 如果你知道想要文件的格式, 可以用下面的命令来避免下载那些多余的文件:
```
wget -nd -r -l1 --no-parent -A.mp3 -A.wma http://www.foo.com/mp3/
```
我来简单的介绍一下命令中指定选项的作用.
-nd 不创建目录, wget默认会创建一个目录
-r 递归下载
-l1 (L one) 递归一层,只下载指定文件夹中的内容, 不下载下一级目录中的.
–no-parent 不下载父目录中的文件

##### rsync可视化复制文件时的进度
```
rsync -avPh 源文件 目标文件
```
