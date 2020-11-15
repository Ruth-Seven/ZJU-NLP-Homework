# Node Classification Task

NLP作业3要求在两个不同数据集上分别使用不同的图卷积层来建立图神经网络，完成图上的节点分类任务。



[toc]

# 实验报告

## 0 安装

该部分安装比较繁琐，舍去了CUDA安装部分。

### 0.1 安装`pytorch-geometric`

使用命令下载安装相关安装包：

```shell
pip install torch-scatter
pip install torch-sparse
pip install torch-cluster
pip install torch-spline-conv
pip install torch-geometric
pip install torch-geometric
```

但是前四个包发现无法直接使用`pip`下载相关包，只能去[官网](https://pytorch-geometric.com/whl/torch-1.6.0.html
)下载相关适配当前环境的`whl`文件，然后使用`pip install "*.whl"` 安装环境。

![image-20201108142717592](D:\个人文件\重要文件\闲书与笔记\MD暂存文件\image-20201108142717592.png)

### 0.2 安装`pytorch`



```shell
conda install pytorch==1.6.0 torchvision==0.7.0 cpuonly -c pytorch
```

### 0.3 测试

## 1. 数据准备

根据Project要求，我们选择了`Cora`数据集和`CiteSeer`数据集。

![image-20201114230413394](http://static.come2rss.xyz/image-20201114230413394.png)

## 2. 图卷积

类比于计算机视觉中使用的卷积层概念，我们需在在图神经网络上也能够进行图卷积，将一个节点的上的信息能够和周围一度甚至$K$度邻居的信息相结合。我们希望$f(G)$能够将相似的节点映射到低维的`embedding space`中同样相似的两个节点中。

![image-20201115073546138](http://static.come2rss.xyz/image-20201115073546138.png)

但是由于图的拓扑不确定性和复杂性，不同于图像的空间规律性，所以难以直接使用卷积。所以图上的卷积需要另一种方式：`Aggregate Neighbors`。图上的邻居和节点信息给出了卷积的方式，定义了卷积的计算图。我们可以用神经网络来聚合一个节点`A`以及其周围邻居的信息，之后再聚合一次节点`A`的邻居`B`以及周围已经聚合过的节点信息来获取二度邻居的信息，以此递推，可以节点就可以聚集到`K`度邻居的信息。

![image-20201115074405795](D:\个人文件\重要文件\闲书与笔记\MD暂存文件\image-20201115074405795.png)

如下图所示：

![image-20201115074729584](D:\个人文件\重要文件\闲书与笔记\MD暂存文件\image-20201115074729584.png)





### 2.1 基础方法




$$
\begin{split}
h^0_v&=X_v\\
h_v^k&= \sigma(W_k\sum_{u\in N(v)} \frac{h_u^{k-1}}{|N(v)|} + B_k h_v^{k-1}), \forall k \in {1,\cdots, K} \\
z_v &=  h_v^K

\end{split}
$$
其中$\sigma$是非线性函数，提供非线性能力。聚合策略是平均值，当然也有最大值。其中$W_k$和$B_k$是需要学习的参数。

等价的向量化形式是：
$$
H^{l+1}= \sigma(H^{(l)}W_0^{(l)}+\widetilde AH^{(l)}W_1^{(l)})
$$
其中$\widetilde A= D^{-\frac{1}{2}}AD^{-\frac{1}{2}} $，做了一个标准化。



结果显示，这种方法具有一定的`Inductive`能力，在制定数据上训练的模型可以在另一个数据集上的图中应用，也能够对临时生成的节点进行`embedding`。



![image-20201115075858492](http://static.come2rss.xyz/image-20201115075858492.png)





### 2.2 `GraphSAGE`

相比于第一种思想，它将neighbor embedding和self embedding连接起来，
$$
h_v^k= \sigma([W_k \cdot AGG({h_u^{k-1}, \forall u \in N(v)}) , B_k h_v^{k-1}]), \forall k \in {1,\cdots, K} \\
$$
其中$AGG$可以使用平均值函数，池化函数，甚至LSTM函数。`GraphSAGE`相比于之前的方法添加了自身的信息。



![image-20201115080954355](D:\个人文件\重要文件\闲书与笔记\MD暂存文件\image-20201115080954355.png)

下面是模型的简单实现：

```python
class GraphSage(pyg_nn.MessagePassing):
    """Non-minibatch version of GraphSage."""
    def __init__(self, in_channels, out_channels, reducer='mean',
                 normalize_embedding=True):
        super(GraphSage, self).__init__(aggr='mean')
        self.lin = torch.nn.Linear(in_channels*2, out_channels, bias=True)
        self.agg_lin = F.normalize

        if normalize_embedding:
            self.normalize_emb = True

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x):
        aggr_out = torch.cat([aggr_out, x], dim=1)
        aggr_out = F.relu(self.lin(aggr_out))
        if self.normalize_emb:
            aggr_out = self.agg_lin(aggr_out, p=2, dim=1)
        return aggr_out
```





### 2.3 Graph Attention Network(GAT)

在多个计算机视觉和自然语言处理中大方异彩的Attention机制在引入到图神经网络也能有一席之地。之前给出的一般思想对所有节点同等对待，Attention机制改变了GCN平等对待周围所有节点的策略，对每个不同的令居分配不同的注意力得分，从而识别出更重要的邻居。

`GAT`在传播传播过程引入自注意力（self-attention）机制，每个节点的隐藏状态通过注意其邻居节点来计算。

GAT网络由堆叠简单的图注意力层（graph attention layer）来实现，每一个注意力层对节点对 $(i,j)$ ，注意力系数计算方式为：
$$
a_{ij}= \frac{LeakyReLU(a^T[Wh_i || Wh_j])}
{\sum_{k\in N_i}LeakyReLU(a^T[Wh_i || Wh_k])}
$$
其中， ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_%7Bij%7D) 是节点 $j$ 到 ![[公式]](https://www.zhihu.com/equation?tex=i) 的注意力系数， ![[公式]](https://www.zhihu.com/equation?tex=N_i) 表示节点 ![[公式]](https://www.zhihu.com/equation?tex=i) 的邻居节点。节点输入特征为 ![[公式]](https://www.zhihu.com/equation?tex=h%3D%5C%7Bh_1%2Ch_2%2C...%2Ch_N%5C%7D%2Ch_i%5Cin+R%5EF) ，其中 ![[公式]](https://www.zhihu.com/equation?tex=N%2CF) 分别表示节点个数和特征维数。节点特征的输出为 ![[公式]](https://www.zhihu.com/equation?tex=h%27%3D%5C%7Bh%27_1%2Ch%27_2%2C...%2Ch%27_N%5C%7D%2Ch%27_i%5Cin+R%5E%7BF%27%7D) 。 ![[公式]](https://www.zhihu.com/equation?tex=W%5Cin+R%5E%7BF%27%5Ctimes+F%7D) 是在每一个节点上应用的线性变换权重矩阵， ![[公式]](https://www.zhihu.com/equation?tex=a%5Cin+R%5E%7B2F%27%7D) 是权重向量，可以将输入映射到 ![[公式]](https://www.zhihu.com/equation?tex=R) 。最终使用`softmax`进行归一化并加入`LeakyReLU`以提供非线性性（其中负输入的斜率为0.2）。

最终节点的特征输出由以下式子得到：
$$
h_i^` = \parallel ^{K}_{k=1} \sigma(\sum_{j \in N_i} \alpha_{ij}^k W^k h_j)
$$

下面是模型实现：


```python
class GAT(pyg_nn.MessagePassing):

    def __init__(self, in_channels, out_channels, num_heads=8, concat=True,
                 dropout=0, bias=True, **kwargs):
        super(GAT, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels // num_heads
        self.heads = num_heads
        self.concat = concat
        self.dropout = dropout

        ############################################################################

        self.lin = torch.nn.Linear(in_channels, self.out_channels*self.heads, bias=True)
        self.att = torch.nn.Parameter(torch.Tensor(1, self.heads, self.out_channels * 2))
        ############################################################################

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(self.heads * self.out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)


    def forward(self, x, edge_index, size=None):
        ############################################################################
        x = self.lin(x)
        ############################################################################

        # Start propagating messages.
        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        #  Constructs messages to node i for each edge (j, i).

        ############################################################################
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        x = torch.cat([x_i, x_j], dim=-1)
        alpha = (x * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = pyg_utils.softmax(src=alpha, index=edge_index_i, num_nodes=size_i)
        ############################################################################

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = x_j * alpha.view(-1, self.heads, 1)
        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        return out

    def update(self, aggr_out):
        # Updates node embedings.
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out
```

## 3. 实验结果

|            | `GCN` | `GraphSAGE` | `GAT` |
| ---------- | ----- | ----------- | ----- |
| `Cora`     | 82.1% | 80.7%       | 83.1% |
| `Citeseer` | 64.3% | 68.3%       | 69%   |

总体来看`GAT`效果最好，`GCN`和`GraphSAGE`效果略逊。



## 参考资料

Inductive Representation Learning on Large Graphs

GRAPH ATTENTION NETWORKS

[GCN理解](https://zhuanlan.zhihu.com/p/140819743)

[GNN模型之入门GCN](https://zhuanlan.zhihu.com/p/120311352)

