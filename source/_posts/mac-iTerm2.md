---
title: mac-iTerm2
date: 2021-11-15 14:09:28
index_img: /img/article/iterm.jpg
categories:
    - Mac
tags:
    - 玩
comment: 'valine'
---
## mac air m1 终端配置记录
<!-- more -->
### 安装Homebrew
```
/bin/bash -c "$(curl -fsSL https://cdn.jsdelivr.net/gh/ineo6/homebrew-install/install.sh)"
```
将以上命令粘贴至终端。脚本内置 中科大镜像，所以能让Homebrew安装的更快。

### 安装 oh-my-zsh
```
sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
```
### 配置 oh-my-zsh
参考 https://www.dazhuanlan.com/lyuuawa0508/topics/1599354
注：其中有些命令可能因为版本问题不一致，主要是cask相关，按照提示修改即可
比如 安装 iTerm2 的命令是现在这样
```
brew tap homebrew/cask
brew install iterm2
```
### brew 安装 nvm
```
brew install nvm
mkdir ~/.nvm
vi ~/.zshrc
#################### 将下面内容添加到 ~/.zshrc 中 #############################
export NVM_DIR="$HOME/.nvm"
[ -s "/opt/homebrew/opt/nvm/nvm.sh" ] && . "/opt/homebrew/opt/nvm/nvm.sh" # This loads nvm
[ -s "/opt/homebrew/opt/nvm/etc/bash_completion.d/nvm" ] && . "/opt/homebrew/opt/nvm/etc/bash_completion.d/nvm" # This loads nvm bash_completion
###############################################################################
source ~/.zshrc
```

### github 加速
参考 https://brew.idayer.com/guide/github