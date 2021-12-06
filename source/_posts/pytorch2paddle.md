---
title: pytorch模型转paddle模型踩坑记录
date: 2021-11-29 17:52:31
index_img: /img/article/torch2paddle.png
categories:
    - TroubleShoot
tags:
    - TroubleShoot
comment: 'valine'
---
## 如题
<!-- more -->
##### 踩坑1
网上有很多使用x2paddle把pytorch转paddle的文章，不过大不部分也都是采用的迂回路线，就是先转ONNX，再转paddle，试了下水，果然没有那么简单的事情，一直报错，最后好像报了个 model not support，，，，遂放弃。
##### 踩坑2
使用工具不行只有一步一步慢慢转，这也是最开始使用的方法，起初报错没解决才找到x2paddle的，没想到又回归到最原始的方法了。
转换的过程一直卡在网络这块，所以就先把网络这块拿出来记录下。
###### 网络
```
######################### torch 版  ############################
import torch
import torch.nn as nn
import torch.nn.functional as F

class SeqNet(nn.Module):
    def __init__(self):
        super(SeqNet, self).__init__()
        # input 
        self.conv1 = nn.Conv1d(12, 10, 50)
        self.conv2 = nn.Conv1d(12, 10, 200)
        self.conv3 = nn.Conv1d(12, 10, 500)
        self.conv4 = nn.Conv1d(12, 10, 1000)
        self.pooling = nn.MaxPool2d((1, 200))
        self.fc1 = nn.Linear(900, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        batch_size = x.size(0)
        
        out1 = self.pooling(F.relu(self.conv1(x)))
        out2 = self.pooling(F.relu(self.conv2(x)))
        out3 = self.pooling(F.relu(self.conv3(x)))
        out4 = self.pooling(F.relu(self.conv4(x)))

        out = torch.cat([out1, out2, out3, out4], 2)
        out = out.view(batch_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        # out = F.dropout(out, p=0.2)
        out = self.fc2(out)
        return out
```
```
######################### paddle 版  ############################
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class SeqNet(nn.Layer):
    def __init__(self):
        super(SeqNet, self).__init__()
        # input 
        self.conv1 = nn.Conv1D(12, 10, 50)
        self.conv2 = nn.Conv1D(12, 10, 200)
        self.conv3 = nn.Conv1D(12, 10, 500)
        self.conv4 = nn.Conv1D(12, 10, 1000)
        # self.pooling = nn.MaxPool2D((1, 200))   
        ### torch版的 nn.MaxPool2D 输入数剧格式为（NCHW 或 CHW）,paddle版的 nn.MaxPool2D 输入数据格式只有 NCHW
        ### N代表batch_size， C代表channel，H代表高度，W代表宽度
        ### 所以这里用 paddle 的 nn.MaxPool1D 替换了 torch 的 nn.MaxPool2D
        self.pooling = nn.MaxPool1D(200)
        self.fc1 = nn.Linear(900, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        ### torch.tensor.size 对应 paddle.tensor.shape
        batch_size = x.shape[0]   
        
        out1 = self.pooling(F.relu(self.conv1(x)))
        out2 = self.pooling(F.relu(self.conv2(x)))
        out3 = self.pooling(F.relu(self.conv3(x)))
        out4 = self.pooling(F.relu(self.conv4(x)))
        
        ### torch.cat 对应 paddle.concat
        # out = torch.cat([out1, out2, out3, out4], 2)  
        out = paddle.concat([out1, out2, out3, out4], 2)
        ### torch.tensor.view 对应 paddle.tensor.reshape
        # out = out.view(batch_size, -1)
        out = paddle.reshape(out, [batch_size, -1])
        out = self.fc1(out)
        out = F.relu(out)
        # out = F.dropout(out, p=0.2)
        out = self.fc2(out)

        return out
```
###### 对于自定义数据集 paddle和pytorch实现的方法类似
```
from paddle.io import Dataset
class MyDataset(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self, mode='train'):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
        """
        super(MyDataset, self).__init__()

        if mode == 'train':
            self.data = [
                ['traindata1', 'label1'],
                ['traindata2', 'label2'],
                ['traindata3', 'label3'],
                ['traindata4', 'label4'],
            ]
        else:
            self.data = [
                ['testdata1', 'label1'],
                ['testdata2', 'label2'],
                ['testdata3', 'label3'],
                ['testdata4', 'label4'],
            ]
    
    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        data = self.data[index][0]
        label = self.data[index][1]

        return data, label

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.data)
```
###### 还有就是训练这块
```
######################### torch 版  ############################
import torch
import torch.nn as nn
model = SeqNet()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4 ,weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
criterion = nn.BCEWithLogitsLoss()

for i, (inputs, labels) in (enumerate(trainloader)):
    inputs = inputs.to(device)
    labels = labels.float().to(device)

    out_linear = model(inputs).to(device)
    loss = criterion(out_linear, labels.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
```
######################### paddle 版  ############################
import paddle
import paddle.nn as nn
import paddle.optimizer as optim

model = SeqNet()
model.to(device)
optimizer = optim.AdamW(learning_rate=1e-4, parameters=model.parameters(),weight_decay=5e-4)
### optimizer = optim.Adam(parameters=model.parameters(), learning_rate=1e-4)
### paddle 版CosineAnnealingDecay接収的是 learning_rate参数
scheduler = optim.lr.CosineAnnealingDecay(1e-4, T_max=max_epoch)
criterion = nn.BCEWithLogitsLoss()

for i, (inputs, labels) in (enumerate(trainloader)):
    # inputs = inputs.to(device)
    inputs = inputs.cuda()
    # labels = labels.float().to(device)
    labels = labels.cuda()
    # labels = paddle.reshape(labels, (30, 1))
    labels = paddle.cast(labels, dtype='float32')  ## 转换数据类型

    out_linear = model(inputs)
    out_linear = paddle.reshape(out_linear, (batch_size,))
    loss = criterion(out_linear, labels)
    # loss = criterion(out_linear, labels.unsqueeze(1))

    # optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()
```

其余剩下就是一些小问题了，直接运行debug改就好了。
pytorch 完整版地址：https://github.com/shubihu/coggle_learn/blob/main/baseline/pytorch.ipynb
paddle 完整版地址：https://github.com/shubihu/coggle_learn/blob/main/baseline/paddle.ipynb
aistudio上项目的地址为：https://aistudio.baidu.com/aistudio/projectdetail/2724787?contributionType=1
