面试question

1.  分类为啥用交叉熵而不是MSE？

2. python手动实现二维卷积

   ```python
   import numpy as np 
   def conv2d(img, in_channels, out_channels ,kernels, bias, stride=1, padding=0):
       N, C, H, W = img.shape 
       kh, kw = kernels.shape
       p = padding
       assert C == in_channels, "kernels' input channels do not match with img"
   
       if p:
           img = np.pad(img, ((0,0),(0,0),(p,p),(p,p)), 'constant') # padding along with all axis
   
       out_h = (H + 2*padding - kh) // stride + 1
       out_w = (W + 2*padding - kw) // stride + 1
   
       outputs = np.zeros([N, out_channels, out_h, out_w])
       # print(img)
       for n in range(N):
           for out in range(out_channels):
               for i in range(in_channels):
                   for h in range(out_h):
                       for w in range(out_w):
                           for x in range(kh):
                               for y in range(kw):
                                   outputs[n][out][h][w] += img[n][i][h * stride + x][w * stride + y] * kernels[x][y]
                   if i == in_channels - 1:
                       outputs[n][out][:][:] += bias[n][out]
       return outputs
   ```

   3. python 注意力机制和多头注意力机制的实现

      自注意力机制

      ```python
      from math import sqrt
      
      import torch
      import torch.nn as nn
      
      class SelfAttention(nn.Module):
          # dim_in: int
          # dim_k: int
          # dim_v: int
      
          def __init__(self, dim_in, dim_k, dim_v):
              super(SelfAttention, self).__init__()
              self.dim_in = dim_in
              self.dim_k = dim_k
              self.dim_v = dim_v
              self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
              self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
              self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
              self._norm_fact = 1 / sqrt(dim_k)
      
          def forward(self, x):
              # x: batch, n, dim_in
              batch, n, dim_in = x.shape
              assert dim_in == self.dim_in
      
              q = self.linear_q(x)  # batch, n, dim_k
              k = self.linear_k(x)  # batch, n, dim_k
              v = self.linear_v(x)  # batch, n, dim_v
      
              dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
              dist = torch.softmax(dist, dim=-1)  # batch, n, n
      
              att = torch.bmm(dist, v)
              return att
      ```

      多头自注意力机制

      ```python
      from math import sqrt
      
      import torch
      import torch.nn as nn
      
      
      class MultiHeadSelfAttention(nn.Module):
          # dim_in: int  # input dimension
          # dim_k: int   # key and query dimension
          # dim_v: int   # value dimension
          # num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads
      
          def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
              super(MultiHeadSelfAttention, self).__init__()
              assert dim_k % num_heads == 0and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
              self.dim_in = dim_in
              self.dim_k = dim_k
              self.dim_v = dim_v
              self.num_heads = num_heads
              self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
              self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
              self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
              self._norm_fact = 1 / sqrt(dim_k // num_heads)
      
          def forward(self, x):
              # x: tensor of shape (batch, n, dim_in)
              batch, n, dim_in = x.shape
              assert dim_in == self.dim_in
      
              nh = self.num_heads
              dk = self.dim_k // nh  # dim_k of each head
              dv = self.dim_v // nh  # dim_v of each head
      
              q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
              k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
              v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)
      
              dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
              dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n
      
              att = torch.matmul(dist, v)  # batch, nh, n, dv
              att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
              return att
      ```

4. 手动实现batch normalization

   ```python
   # 参考并更正自知乎（机器学习入坑者《Batch Normalization原理与python实现》）
   class MyBN:
       def __init__(self, momentum=0.01, eps=1e-5, feat_dim=2):
           """
           初始化参数值
           :param momentum: 动量，用于计算每个batch均值和方差的滑动均值
           :param eps: 防止分母为0
           :param feat_dim: 特征维度
           """
           # 均值和方差的滑动均值
           self._running_mean = np.zeros(shape=(feat_dim, ))
           self._running_var = np.ones((shape=(feat_dim, ))
           # 更新self._running_xxx时的动量
           self._momentum = momentum
           # 防止分母计算为0
           self._eps = eps
           # 对应Batch Norm中需要更新的beta和gamma，采用pytorch文档中的初始化值
           self._beta = np.zeros(shape=(feat_dim, ))
           self._gamma = np.ones(shape=(feat_dim, ))
   
       def batch_norm(self, x):
           """
           BN向传播
           :param x: 数据
           :return: BN输出
           """
           if self.training:
               x_mean = x.mean(axis=0)
               x_var = x.var(axis=0)
               # 对应running_mean的更新公式
               self._running_mean = (1-self._momentum)*x_mean + self._momentum*self._running_mean
               self._running_var = (1-self._momentum)*x_var + self._momentum*self._running_var
               # 对应论文中计算BN的公式
               x_hat = (x-x_mean)/np.sqrt(x_var+self._eps)
           else:
               x_hat = (x-self._running_mean)/np.sqrt(self._running_var+self._eps)
           return self._gamma*x_hat + self._beta                      
   ```

5. 手写k-means

   ```python
   import numpy as np
   
   
   def kmeans(data, k, thresh=1, max_iterations=100):
     # 随机初始化k个中心点
     centers = data[np.random.choice(data.shape[0], k, replace=False)]
   
     for _ in range(max_iterations):
       # 计算每个样本到各个中心点的距离
       distances = np.linalg.norm(data[:, None] - centers, axis=2)
   
       # 根据距离最近的中心点将样本分配到对应的簇
       labels = np.argmin(distances, axis=1)
   
       # 更新中心点为每个簇的平均值
       new_centers = np.array([data[labels == i].mean(axis=0) for i in range(k)])
   
       # 判断中心点是否收敛，多种收敛条件可选
       # 条件1：中心点不再改变
       if np.all(centers == new_centers):
         break
       # 条件2：中心点的阈值小于某个阈值
       # center_change = np.linalg.norm(new_centers - centers)
       # if center_change < thresh:
       #     break
       centers = new_centers
   
     return labels, centers
   
   
   # 生成一些随机数据作为示例输入
   data = np.random.rand(100, 2)  # 100个样本，每个样本有两个特征
   
   # 手动实现K均值算法
   k = 3# 聚类数为3
   labels, centers = kmeans(data, k)
   
   # 打印簇标签和聚类中心点
   print("簇标签:", labels)
   print("聚类中心点:", centers)
   ```

   