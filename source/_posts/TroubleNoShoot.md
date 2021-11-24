---
title: Trouble No Shoot
date: 2021-08-20 10:49:51
index_img: /img/article/troublenoshoot.jpg
categories:
    - TroubleShoot
tags:
    - TroubleShoot
comment: 'valine'
---
## 升级hexo遇到的问题
<!-- more -->
hexo的一个插件需要5.0以上的版本，看了下自己安装的是4.3的版本，所以想着升级一下。查了半天也没找到有效的方法。之后又查看了node的版本看着也很低，想升级的心又来了。折腾了半天愣是没升级成功，还把系统搞坏了，apt、dpkg这些也都没法用了。网上的资料有时候也不能盲目跟着做，还是用root账户删的，真是细思极恐，这要是公司的生产环境，这估计是要被祭天的。估计我也是仗着这是自己电脑里的子系统才敢这么胡作非为。系统坏了，本来想挽救一下的，发现越挽救问题越大。顺放弃。。。于是重新卸载Linux子系统，再重新安装，前后没花10分钟。果然还是微软baba的子系统安装卸载方便啊。
系统重新安装了，很多东西就要重新配置，比如github的免密提交等，这里也简单记录下。
##### 首先配置github及生成ssh秘钥，执行
```
git config --global user.email "you@example.com"    ## 我的 git config --global user.email "jrwjb@sina.com"   
git config --global user.name "Your Name"  ## 我的 git config --global user.name "shubihu"
ssh-keygen        ## 一路回车即可
```
执行完后会在家目录的.ssh下生成下面几个文件
```
id_rsa   ## 私钥
id_rsa.pub  ## 共钥
```
然后把公钥的内容添加到github上即可。
![](/img/article/gitssh.jpg)

##### 回到最开始的问题，升级hexo、node。
因为是新系统，所以相对简单些，直接安装新版的node，可以从官网下载最新的稳定版进行安装，不过我嫌麻烦懒得去下载，所以参考了这篇文章进行安装。
```
## 使用nvm进行安装
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.3/install.sh | bash
nvm install node
```
<!-- <iframe src="https://www.myfreax.com/how-to-install-node-js-on-ubuntu-18-04/" width="100%" height="500" name="topFrame" scrolling="yes"  noresize="noresize" frameborder="0" id="topFrame"></iframe> -->
##### 最后是升级hexo
```
# 使用淘宝源的 cnpm 替换 npm
npm install -g cnpm --registry=https://registry.npm.taobao.org

cnpm install -g cnpm                 # 升级 npm
cnpm cache clean -f                 # 清除 npm 缓存

===更新 hexo: 进入 blog 目录，执行如下命令=== 
# 更新 package.json 中的 hexo 及个插件版本
cnpm install -g npm-check           # 检查之前安装的插件，都有哪些是可以升级的 
cnpm install -g npm-upgrade         # 升级系统中的插件
npm-check
npm-upgrade

# 更新 hexo 及所有插件
cnpm update

# 确认 hexo 已经更新
hexo -v
```
<!-- <iframe src="https://xmuli.tech/posts/cb1e6c4f/" width="100%" height="500" name="topFrame" scrolling="yes"  noresize="noresize" frameborder="0" id="topFrame"></iframe> -->
参考
* https://xmuli.tech/posts/cb1e6c4f
* https://www.myfreax.com/how-to-install-node-js-on-ubuntu-18-04