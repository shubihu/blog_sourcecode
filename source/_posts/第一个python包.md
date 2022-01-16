---
title: 第一个python包
date: 2021-12-29 14:07:01
index_img: /img/article/pypi.jpg
categories:
    - python
tags:
    - python
comment: 'valine'
---
## Senior Data Structure Tools--SDStools
<!-- more -->
python标准库中没有链表、树、图等高级数据结构，所以整理了一些网上的代码到这个库中。

##### 链表：
* https://zhuanlan.zhihu.com/p/60057180
* https://jackkuo666.github.io/Data_Structure_with_Python_book/chapter3/section1.html

##### 如何发布包到pypi
我的项目目录结构如下：
![](/img/article/py.jpg)

打包主要就是setup的编写
```
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),                ## 必须，如果需要打包test文件夹或其他可参考下面格式进行添加
    # packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['SDStools'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    # package_data={
    #     # include json and pkl files
    #     '': ['*.json', 'models/*.pkl', 'models/*.json'],
    # },
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)
```
编写完可以进行本地打包安装测试：
```
python setup.py build     # 执行构建, 会将包的内容构建到 build 文件夹下。
```
```
python setup.py install  # 会将包直接安装到当前解释器的 site-packages 下，安装完成后即可以使用 pip list 命令查看到。
```
如果没什么问题的话就可以提交到pypi了。
```
python setup.py sdist  ## 打包
twine upload dist/*    ## 发布
# python setup.py upload   ## 如果setup.py里有upload命令也可一键执行打包发布
```
参考：
* https://www.jiqizhixin.com/articles/19060901
* https://zhuanlan.zhihu.com/p/66603015
* https://zhuanlan.zhihu.com/p/66603015

##### 源码地址
* https://github.com/shubihu/SDSTools
