# **基于VGG的目标检测系统**

<div style="text-align: center;">专业：智能科学与技术 学号：2312654 姓名：李玉泽</div>

![image-20250427095650741](C:\Users\cassi\AppData\Roaming\Typora\typora-user-images\image-20250427095650741.png)

## 一、实验目的

本实验的目的是实现对于给定图片的目标检测，**目标检测（Object Detection）** 是计算机视觉领域的核心任务之一，旨在识别图像或视频中的特定物体，并确定它们的位置和类别。与简单的图像分类（只判断图像类别）不同，目标检测需要同时完成以下两个任务：

**定位（Localization）**：用边界框（Bounding Box）标出物体的位置。

**分类（Classification）**：判断边界框内物体的类别（如人、车、狗等）。

## 二、实验原理

本实验围绕目标检测任务展开，采用改进后的SSD（Single Shot MultiBox Detector）结构，同时以VGG为骨干网络，设计轻量化检测模型。以下将从算法原理、VGG特征提取结构、整体模型设计、训练策略及数据处理五个方面进行详细说明。

#### 2.1 算法原理：SSD框架

SSD是一种单阶段目标检测算法，能够在一次前向传播中同时完成目标的类别分类与位置回归。它摒弃了候选区域生成模块，直接在不同尺度的特征图上进行检测，使其推理速度远高于Faster R-CNN等双阶段检测器。SSD的核心思想包括多尺度检测、先验框机制和端到端预测。

多尺度检测指的是利用网络中不同层级的特征图，来检测不同尺寸的目标。浅层特征图具有更高的空间分辨率，适合检测小目标；深层特征图则更具语义信息，适合识别较大目标。先验框（也称锚框）是在特征图的每个位置上预设若干长宽比和尺寸不同的框，作为检测的起点，网络学习的是预测这些先验框的偏移以及它们所属的类别。SSD通过端到端的方式直接输出所有框的分类与回归结果，无需额外的区域提议步骤，具有较强的实时性。

#### 2.2 VGG基础网络

作为SSD的主干网络，本实验采用了VGG16的裁剪版本。VGG结构以其层次分明、卷积核统一为3×3而著称，广泛应用于图像分类与目标检测任务。其优点包括结构简单、预训练模型易获取、浅层信息丰富，特别适合进行多尺度特征提取。

![](C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计part2\实验报告\文档插图\插图1.png)

本实验中构建的VGGBase网络保留了VGG16的卷积层结构，并增加了对`conv3_3`特征图的输出，提升了对超小目标的识别能力。同时，在`conv4_3`层之后加入了L2归一化操作，增强特征响应的稳定性。为了减少计算开销，网络去除了VGG的全连接层，仅保留至`pool5`为止的卷积部分。所有卷积层均可加载ImageNet预训练参数，以加快模型收敛速度并提高初期性能。

![](C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计part2\实验报告\文档插图\插图2.png)

#### 2.3 模型结构设计

本实验搭建的模型结构主要由三个部分构成：VGGBase特征提取网络、多尺度检测头PredictionConvolutions，以及先验框生成模块。

首先，VGGBase网络输出四个不同尺度的特征图：分别为`conv3_3`（尺寸为56×56）、`conv4_3`（28×28）、`conv5_3`（14×14）和`pool5`（7×7）。这些特征图从浅到深涵盖了从超小目标到大目标的多种感受野信息。对于小目标检测中至关重要的浅层特征，如`conv4_3`，通过L2归一化操作保证其特征幅度在训练过程中保持稳定。

其次，在每个特征图上独立添加两个卷积模块，分别用于位置回归和类别分类。这些卷积层被称为“预测卷积”，它们会在每个像素点处预测多个边界框的位置偏移与类别置信度。每层特征图使用不同数量的先验框，例如`conv3_3`和`conv4_3`使用4种，`conv5_3`和`pool5`使用6种，确保在各个尺度范围内覆盖多种目标形态。

最后，模型中的先验框生成模块会基于每个特征图的空间位置及配置的长宽比和尺寸，生成大量标准化的锚框。这些先验框经过坐标归一化处理，确保输出框的范围始终在[0,1]之间，方便后续进行回归解码与后处理。

#### 2.4 训练策略

本实验采用SSD原始论文中提出的MultiBox Loss作为损失函数。该损失包含位置回归损失与分类损失两个部分。其中，位置回归部分采用Smooth L1损失函数，用于度量预测边界框与真实框之间的坐标差异；分类部分采用交叉熵损失函数，并引入难例挖掘机制，仅对损失较大的负样本进行反向传播，以保持正负样本比不超过1:3，提升训练稳定性与收敛速度。

本实验采用SSD原始论文中提出的 MultiBox Loss 作为损失函数。该损失函数由两部分组成：位置回归损失和分类损失。其中，位置回归部分用于度量预测边界框与真实边界框之间的几何差异，采用平滑的L1损失（Smooth L1），其表达式为：
$$
d(x, y) = \sum_{i=1}^{n} \mathrm{smooth}_{L1}(x_i - y_i)
$$
其中 x~i~ 为预测边界框的坐标，y~i~ 为真实边界框的坐标，n表示坐标维度。

分类部分使用交叉熵损失来度量每个先验框预测类别与真实类别之间的差异，同时引入难例挖掘（Hard Negative Mining）策略以平衡正负样本比例。交叉熵损失定义如下：
$$

\mathrm{Loss}_{cls} = -\sum_{i=1}^{C} y_i \log(p_i)
$$
其中 y~i~是真实类别的独热编码，p~i~是预测的类别概率。最终的总损失函数是上述两部分损失的加权和，形式为：
$$

\mathrm{Loss}_{total} = \alpha \cdot \mathrm{Loss}_{cls} + \beta \cdot \mathrm{Loss}_{bbox}
$$
其中α和β为调节权重的超参数。

优化器方面，实验使用带动量的随机梯度下降法（SGD），动量因子设为 0.9，同时加入权重衰减项以抑制过拟合。学习率初始设为 0.001，并在训练到第 150 和第 190 个 epoch 时衰减为原来的0.1，保证模型在后期更精细地收敛。

此外，训练脚本支持断点恢复机制与迁移学习：在已有训练模型的基础上继续训练或进行结构扩展后重新加载骨干网络参数，能够有效节省训练时间并提升性能。

#### 2.5 数据处理与增强

实验采用PASCAL VOC 2007与VOC 2012数据集作为训练与测试基础。每张图像配有XML标注文件，记录图像中每个目标的边界框坐标与所属类别。为了提升模型的泛化能力，图像在输入前会经过一系列数据增强操作，包括图像缩放、随机裁剪、亮度与饱和度扰动、随机水平翻转等。所有边界框坐标也会同步调整，并归一化到[0,1]区间。

此外，为突出检测效果，本实验在数据加载阶段保留了目标类别筛选机制，仅对部分关注类别（如bicycle、person）进行训练。这一处理有助于在复杂背景中增强特定目标的检测精度，同时减少无关类别干扰，提高模型学习效率。

## 三、实验步骤

#### 3.1 **数据准备**

首先，需要将 PASCAL VOC2007 与 VOC2012 数据集格式转换为适合模型训练的 JSON 格式。该步骤通过执行 `create_data_lists.py` 文件完成，调用了 `utils.py` 中定义的 `create_data_lists()` 函数。该函数会遍历所有图像及其对应的 XML 标注文件，提取目标的类别标签与边界框坐标信息，并将其以 `.json` 格式保存至指定路径。

![](C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计part2\实验报告\文档插图\9f1da2028227e46eef0658b95e5bc8fb.png)

#### 3.2**模型设计**

模型结构定义在 `model.py` 文件中，主要实现了改进版的 `tiny_detector_enhanced` 模型。该模型包含三个模块：

1. `VGGBase`：精简的 VGG 特征提取网络，输出四个尺度的特征图；
2. `PredictionConvolutionsEnhanced`：对多尺度特征图分别进行边界框与类别预测；
3. `create_prior_boxes()`：根据特征图尺寸和预设的尺度比例生成先验框。

模型支持从预训练的标准 SSD 模型加载兼容权重，并保留“L2Norm”归一化层增强小目标稳定性。`model.py` 中还包含了损失函数 `MultiBoxLoss` 的实现。

#### 3.3**数据加载**

在 `datasets.py` 中，定义了 `PascalVOCDataset` 类来加载和预处理数据。该类继承自 PyTorch 的 `Dataset` 抽象类，实现了图像、边界框、标签、难度标记的加载与增强。处理过程包括图像缩放、随机裁剪、颜色扰动、坐标归一化等。

此外，该类中还实现了“关注类别筛选”功能。在每张图像读取后，程序会判断图像中是否包含如 bicycle、person 等指定类别，若无，则跳过这张图像。这种做法可减少不相关样本，提高训练效率。

数据加载使用 PyTorch 的 `DataLoader` 搭配 `collate_fn()` 函数，确保每个 batch 内的图像与其变长标签能够正确打包并送入模型。

#### 3.4**训练流程**

训练流程在 `train.py` 中实现。首先构建模型并初始化优化器、损失函数，并支持如下机制：

- **迁移学习**：自动加载之前训练得到的模型参数，继续训练；
- **断点恢复**：支持从中断处恢复训练，保存当前 epoch 和 batch 索引；
- **微调策略**：冻结除 conv3_3 之外的 VGG 卷积层，仅训练检测头部分；
- **学习率调度**：训练到设定轮数时自动降低学习率以细化参数。

训练过程采用 `MultiBoxLoss` 损失函数，联合优化位置回归与分类预测。每隔若干轮保存模型权重，并打印损失、准确率等中间结果供监控使用。

模型训练时的数据增强与筛选均由 `PascalVOCDataset` 自动完成；损失计算则自动处理正负样本分布，包含 hard negative mining 策略。

#### **3.5 检测与可视化**

模型训练完成后，可通过 `detect.py` 文件对测试图像进行预测和可视化。该脚本加载训练完成的权重文件，调用模型进行前向推理，并输出每个预测框的类别、位置与置信度。

为提升实际效果，程序中增加了 `max_size_ratio` 参数，用于过滤过大或过小的检测框。这一策略可根据全景图或实际场景对目标大小做出限制，提高预测结果的可控性与精度。

预测结果将以彩色边界框绘制在原图上，并保存输出图片至指定目录，便于人工检查。

## 四、程序代码

代码总共包含六个部分，分别为create_data_lists.py，datasets.py，model.py，train.py，utils.py,detect.py以及utils.py，其中create_data_lists.py，datasets.py程序对数据进行预处理，将VOC数据集转化为 JSON 格式，model.py给出了模型的结构，train.py实现了对神经网络的训练，detect.py将用训练产生的权重对输入图像进行预测,utils.py包含了一些必要的函数，如LoU的计算等。

create_data_lists.py

```python
"""
加载数据
"""
from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(voc07_path='./dataset/VOCdevkit/VOC2007',
                      voc12_path='./dataset/VOCdevkit/VOC2012',
                      output_folder='./dataset/VOCdevkit')
```

datasets.py

```python
import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform


class PascalVOCDataset(Dataset):
    """
    一个PyTorch数据集类，用于在PyTorch DataLoader中创建批次。
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: 存储数据文件的文件夹
        :param split: 划分，'TRAIN'或'TEST'之一
        :param keep_difficult: 是否保留被认为难以检测的对象？
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # 读取数据文件
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)


    def __getitem__(self, i):
        # 读取图像
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # 读取此图像中的对象（边界框、标签、难度）
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_个对象, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_个对象)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_个对象)

        # 如果需要，丢弃难以检测的对象
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # 应用转换
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        # 在datasets.py中的PascalVOCDataset.__getitem__方法中
        # 添加过滤代码，跳过不包含我们关注类别的图像
        keep_classes = {2, 7, 14, 15}  # bicycle, car, motorbike, person的索引

        # 检查图像中是否有我们关注的类别
        has_target_class = any(label.item() in keep_classes for label in labels)
        if not has_target_class:
            # 可以跳过这张图或返回下一张图
            return None

        return image, boxes, labels, difficulties


    def __len__(self):
        return len(self.images)


    def collate_fn(self, batch):
        """
        由于每张图像可能包含不同数量的对象，我们需要一个整合函数（传递给DataLoader）。

        这描述了如何组合这些不同大小的张量。我们使用列表。

        注意：这不必在此类中定义，可以独立存在。

        :param batch: 来自__getitem__()的N组可迭代对象
        :return: 一个图像张量，包含边界框、标签和难度的不同大小张量的列表
        """
        # 过滤None值
        batch = [b for b in batch if b is not None]
        
        # 空批次处理
        if len(batch) == 0:
            return torch.tensor([]), [], [], []
        
        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)
        return images, boxes, labels, difficulties


```

model.py

```python
from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGGBase(nn.Module):
    """
    VGG基础卷积层，用于生成多个特征图。
    修改后返回多个尺度的特征图，以便检测不同大小的目标。
    """

    def __init__(self):
        super(VGGBase, self).__init__()

        # VGG16中的标准卷积层
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 步长 = 1，默认设置
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)    # 224->112

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)    # 112->56

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)    # 56->28

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)    # 28->14

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)    # 14->7
        
        # 添加特征层归一化操作，特别是对浅层特征图
        self.l2_norm_conv4_3 = L2Norm(512, scale=20)
        
        # 加载在ImageNet上预训练的权重
        self.load_pretrained_layers()

    def forward(self, image):
        """
        前向传播。

        :param image: 图像，维度为(N, 3, 224, 224)的张量
        :return: 多个特征图，用于检测不同尺度的目标
        """
        out = F.relu(self.conv1_1(image))  # (N, 64, 224, 224)
        out = F.relu(self.conv1_2(out))  # (N, 64, 224, 224)
        out = self.pool1(out)  # (N, 64, 112, 112)

        out = F.relu(self.conv2_1(out))  # (N, 128, 112, 112)
        out = F.relu(self.conv2_2(out))  # (N, 128, 112, 112)
        out = self.pool2(out)  # (N, 128, 56, 56)

        out = F.relu(self.conv3_1(out))  # (N, 256, 56, 56)
        out = F.relu(self.conv3_2(out))  # (N, 256, 56, 56)
        conv3_3_feats = F.relu(self.conv3_3(out))  # (N, 256, 56, 56)
        out = self.pool3(conv3_3_feats)  # (N, 256, 28, 28)

        out = F.relu(self.conv4_1(out))  # (N, 512, 28, 28)
        out = F.relu(self.conv4_2(out))  # (N, 512, 28, 28)
        conv4_3_feats = F.relu(self.conv4_3(out))  # (N, 512, 28, 28)
        # 对conv4_3特征进行L2归一化
        norm_conv4_3_feats = self.l2_norm_conv4_3(conv4_3_feats)
        out = self.pool4(conv4_3_feats)  # (N, 512, 14, 14)

        out = F.relu(self.conv5_1(out))  # (N, 512, 14, 14)
        out = F.relu(self.conv5_2(out))  # (N, 512, 14, 14)
        conv5_3_feats = F.relu(self.conv5_3(out))  # (N, 512, 14, 14)
        out = self.pool5(conv5_3_feats)  # (N, 512, 7, 7)

        # 返回四个尺度的特征图，包括新增的conv3_3
        return conv3_3_feats, norm_conv4_3_feats, conv5_3_feats, out

    def load_pretrained_layers(self):
        """
        我们使用在ImageNet任务上预训练的VGG-16作为基础网络。
        """
        # 基础网络的当前状态
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # 预训练的VGG基础网络
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        # 将预训练模型的卷积参数传输到当前模型
        for i, param in enumerate(param_names):
            if param.startswith('l2_norm'):
                continue  # 跳过L2归一化层的参数
            if i < len(pretrained_param_names):
                state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        self.load_state_dict(state_dict)
        print("\n已加载基础模型。\n")


class L2Norm(nn.Module):
    """
    L2归一化层，用于归一化conv4_3特征图
    """
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.scale)

    def forward(self, x):
        # x - [N, C, H, W]
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = x / norm
        # 对每个通道应用可学习的缩放因子
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) * x
        return out


class PredictionConvolutions(nn.Module):
    """
    用于多尺度特征图的预测卷积
    """

    def __init__(self, n_classes):
        """
        :param n_classes: 不同类型对象的数量
        """
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        # 针对不同特征图使用不同数量的先验框
        # conv4_3 (28x28) - 4种先验框
        # conv5_3 (14x14) - 6种先验框  
        # pool5 (7x7) - 6种先验框
        n_boxes = {'conv4_3': 4, 'conv5_3': 6, 'pool5': 6}
        
        # 增加针对conv3_3特征图的预测卷积
        n_boxes['conv3_3'] = 4  # 为超小目标设置4个先验框

        # Conv3_3特征图预测层 (超小目标) - 56x56
        self.loc_conv3_3 = nn.Conv2d(256, n_boxes['conv3_3'] * 4, kernel_size=3, padding=1)
        self.cl_conv3_3 = nn.Conv2d(256, n_boxes['conv3_3'] * n_classes, kernel_size=3, padding=1)

        # 为不同特征图创建定位和分类卷积
        # Conv4_3特征图 (小目标) - 28x28
        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        
        # Conv5_3特征图 (中等目标) - 14x14
        self.loc_conv5_3 = nn.Conv2d(512, n_boxes['conv5_3'] * 4, kernel_size=3, padding=1)
        self.cl_conv5_3 = nn.Conv2d(512, n_boxes['conv5_3'] * n_classes, kernel_size=3, padding=1)
        
        # Pool5特征图 (大目标) - 7x7
        self.loc_pool5 = nn.Conv2d(512, n_boxes['pool5'] * 4, kernel_size=3, padding=1)
        self.cl_pool5 = nn.Conv2d(512, n_boxes['pool5'] * n_classes, kernel_size=3, padding=1)

        # 初始化卷积参数
        self.init_conv2d()

    def init_conv2d(self):
        """
        初始化卷积参数。
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv3_3_feats, conv4_3_feats, conv5_3_feats, pool5_feats):
        """
        前向传播。

        :param conv3_3_feats: conv3_3特征图，用于检测超小目标 (N, 256, 56, 56)
        :param conv4_3_feats: conv4_3特征图，用于检测小目标 (N, 512, 28, 28)
        :param conv5_3_feats: conv5_3特征图，用于检测中等目标 (N, 512, 14, 14)
        :param pool5_feats: pool5特征图，用于检测大目标 (N, 512, 7, 7)
        :return: 所有特征图的位置和类别预测
        """
        batch_size = conv4_3_feats.size(0)
        
        # Conv3_3 预测 (56x56特征图，适合超小目标)
        l_conv3_3 = self.loc_conv3_3(conv3_3_feats)  # (N, 4*4, 56, 56)
        l_conv3_3 = l_conv3_3.permute(0, 2, 3, 1).contiguous()
        l_conv3_3 = l_conv3_3.view(batch_size, -1, 4)  # (N, 12544, 4)，56*56*4=12544个框
        
        c_conv3_3 = self.cl_conv3_3(conv3_3_feats)  # (N, 4*n_classes, 56, 56)
        c_conv3_3 = c_conv3_3.permute(0, 2, 3, 1).contiguous()
        c_conv3_3 = c_conv3_3.view(batch_size, -1, self.n_classes)  # (N, 12544, n_classes)
        
        # Conv4_3 预测 (28x28特征图，适合小目标)
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  # (N, 4*4, 28, 28)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()  # (N, 28, 28, 4*4)
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)  # (N, 3136, 4)，28*28*4=3136个框
        
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)  # (N, 4*n_classes, 28, 28)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.n_classes)  # (N, 3136, n_classes)
        
        # Conv5_3 预测 (14x14特征图，适合中等目标)
        l_conv5_3 = self.loc_conv5_3(conv5_3_feats)  # (N, 6*4, 14, 14)
        l_conv5_3 = l_conv5_3.permute(0, 2, 3, 1).contiguous()
        l_conv5_3 = l_conv5_3.view(batch_size, -1, 4)  # (N, 1176, 4)，14*14*6=1176个框
        
        c_conv5_3 = self.cl_conv5_3(conv5_3_feats)  # (N, 6*n_classes, 14, 14)
        c_conv5_3 = c_conv5_3.permute(0, 2, 3, 1).contiguous()
        c_conv5_3 = c_conv5_3.view(batch_size, -1, self.n_classes)  # (N, 1176, n_classes)
        
        # Pool5 预测 (7x7特征图，适合大目标)
        l_pool5 = self.loc_pool5(pool5_feats)  # (N, 6*4, 7, 7)
        l_pool5 = l_pool5.permute(0, 2, 3, 1).contiguous()
        l_pool5 = l_pool5.view(batch_size, -1, 4)  # (N, 294, 4)，7*7*6=294个框
        
        c_pool5 = self.cl_pool5(pool5_feats)  # (N, 6*n_classes, 7, 7)
        c_pool5 = c_pool5.permute(0, 2, 3, 1).contiguous()
        c_pool5 = c_pool5.view(batch_size, -1, self.n_classes)  # (N, 294, n_classes)
        
        # 合并所有特征图的预测，增加conv3_3部分
        locs = torch.cat([l_conv3_3, l_conv4_3, l_conv5_3, l_pool5], dim=1)  # (N, 17150, 4)
        classes_scores = torch.cat([c_conv3_3, c_conv4_3, c_conv5_3, c_pool5], dim=1)  # (N, 17150, n_classes)
        
        return locs, classes_scores


class tiny_detector(nn.Module):
    """
    修改后的tiny_detector网络
    使用多尺度特征图和更小的先验框来增强小目标检测能力
    """

    def __init__(self, n_classes):
        super(tiny_detector, self).__init__()

        self.n_classes = n_classes

        self.base = VGGBase()
        self.pred_convs = PredictionConvolutions(n_classes)

        # 先验框
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        """
        前向传播。

        :param image: 图像，维度为(N, 3, 224, 224)的张量
        :return: 位置和类别预测
        """
        # 获取四个特征图
        conv3_3_feats, conv4_3_feats, conv5_3_feats, pool5_feats = self.base(image)

        # 对各个特征图进行预测
        locs, classes_scores = self.pred_convs(conv3_3_feats, conv4_3_feats, conv5_3_feats, pool5_feats)
        return locs, classes_scores

    def create_prior_boxes(self):
        """
        为多尺度特征图创建先验框，包括新增的conv3_3特征图
        """
        prior_boxes = []
        
        # 特征图尺寸
        fmap_dims = {'conv3_3': 56, 'conv4_3': 28, 'conv5_3': 14, 'pool5': 7}
        
        # 先验框尺度配置
        obj_scales = {
            'conv3_3': [0.01, 0.025, 0.05, 0.08],  # 超小目标尺度
            'conv4_3': [0.1, 0.15, 0.2, 0.25],     # 小目标尺度
            'conv5_3': [0.3, 0.37, 0.44, 0.51, 0.58, 0.65],  # 修改为6个尺度
            'pool5': [0.7, 0.76, 0.82, 0.88, 0.94, 1.0]      # 修改为6个尺度
        }
        
        # 长宽比配置，确保与先验框数量匹配
        aspect_ratios = {
            'conv3_3': [1., 2., 0.5, 1.],  # 4个
            'conv4_3': [1., 2., 0.5, 1.],  # 4个
            'conv5_3': [1., 2., 0.5, 3., 1./3., 1.],  # 6个
            'pool5': [1., 2., 0.5, 3., 1./3., 1.]     # 6个
        }
        
        # 为conv3_3特征图(56x56)创建先验框
        for i in range(fmap_dims['conv3_3']):
            for j in range(fmap_dims['conv3_3']):
                cx = (j + 0.5) / fmap_dims['conv3_3']
                cy = (i + 0.5) / fmap_dims['conv3_3']
                
                for s, ar in zip(obj_scales['conv3_3'], aspect_ratios['conv3_3']):
                    prior_boxes.append([cx, cy, s * sqrt(ar), s / sqrt(ar)])

        # 为conv4_3特征图(28x28)创建先验框
        for i in range(fmap_dims['conv4_3']):
            for j in range(fmap_dims['conv4_3']):
                cx = (j + 0.5) / fmap_dims['conv4_3']
                cy = (i + 0.5) / fmap_dims['conv4_3']
                
                for s, ar in zip(obj_scales['conv4_3'], aspect_ratios['conv4_3']):
                    prior_boxes.append([cx, cy, s * sqrt(ar), s / sqrt(ar)])

        # 为conv5_3特征图(14x14)创建先验框
        for i in range(fmap_dims['conv5_3']):
            for j in range(fmap_dims['conv5_3']):
                cx = (j + 0.5) / fmap_dims['conv5_3']
                cy = (i + 0.5) / fmap_dims['conv5_3']
                
                for s, ar in zip(obj_scales['conv5_3'], aspect_ratios['conv5_3']):
                    prior_boxes.append([cx, cy, s * sqrt(ar), s / sqrt(ar)])
        
        # 为pool5特征图(7x7)创建先验框
        for i in range(fmap_dims['pool5']):
            for j in range(fmap_dims['pool5']):
                cx = (j + 0.5) / fmap_dims['pool5']
                cy = (i + 0.5) / fmap_dims['pool5']
                
                for s, ar in zip(obj_scales['pool5'], aspect_ratios['pool5']):
                    prior_boxes.append([cx, cy, s * sqrt(ar), s / sqrt(ar)])
        
        # 转换为PyTorch张量
        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (17150, 4)
        prior_boxes.clamp_(0, 1)  # 确保先验框坐标在[0,1]范围内
        
        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score=0.2, max_overlap=0.5, top_k=200):
        """
        后处理步骤 - 将预测转换为边界框、标签和得分
        
        :param predicted_locs: 预测的边界框偏移，形状为 (N, n_priors, 4)
        :param predicted_scores: 预测的类别分数，形状为 (N, n_priors, n_classes)
        :param min_score: 最低置信度阈值
        :param max_overlap: 非极大值抑制的最大重叠阈值
        :param top_k: 每张图像保留的最多检测数量
        :return: 检测到的边界框坐标、标签和得分（每个图像的列表）
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, n_priors, n_classes)

        # 创建存储最终检测结果的列表
        all_images_boxes = []
        all_images_labels = []
        all_images_scores = []

        # 处理每张图像
        for i in range(batch_size):
            # 解码边界框坐标
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (n_priors, 4)

            # 对每个类别（跳过背景）进行处理
            image_boxes = []
            image_labels = []
            image_scores = []

            for c in range(1, self.n_classes):
                # 保留超过阈值的置信度
                class_scores = predicted_scores[i][:, c]  # (n_priors)
                score_above_min_score = class_scores > min_score  # (n_priors)
                n_above_min_score = score_above_min_score.sum().item()

                if n_above_min_score == 0:
                    continue

                # 提取超过阈值的框和置信度
                class_scores = class_scores[score_above_min_score]  # (n_qualified)
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # 按置信度降序排序
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)
                class_decoded_locs = class_decoded_locs[sort_ind]

                # 应用非极大值抑制 (NMS)
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_qualified)
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                # 按分数高低逐个处理框
                for box in range(class_decoded_locs.size(0)):
                    # 如果这个框已被抑制，跳过
                    if suppress[box] == 1:
                        continue

                    # 抑制与当前框重叠高的其他框
                    suppress = torch.max(suppress, (overlap[box] > max_overlap).to(torch.uint8))
                    # 不要抑制当前框
                    suppress[box] = 0

                # 保留未被抑制的框
                image_boxes.append(class_decoded_locs[suppress == 0])
                image_labels.append(torch.LongTensor((suppress == 0).sum().item() * [c]).to(device))
                image_scores.append(class_scores[suppress == 0])

            # 如果没有物体被检测到
            if len(image_boxes) == 0:
                all_images_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                all_images_labels.append(torch.LongTensor([0]).to(device))
                all_images_scores.append(torch.FloatTensor([0.]).to(device))
                continue

            # 将所有类别的检测结果合并
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # 保留得分最高的top_k个
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]
                image_boxes = image_boxes[sort_ind][:top_k]
                image_labels = image_labels[sort_ind][:top_k]

            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores


class MultiBoxLoss(nn.Module):
    """
    目标检测的损失函数。
    对于Loss的计算，完全遵循SSD的定义，即 MultiBox Loss

    这是以下组合：
    (1) 针对预测框位置的定位损失。
    (2) 针对预测类别分数的置信度损失。
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        前向传播。

        :param predicted_locs: 相对于441个先验框的预测位置/框，维度为(N, 441, 4)的张量
        :param predicted_scores: 每个编码位置/框的类别分数，维度为(N, 441, n_classes)的张量
        :param boxes: 真实目标边界框，N个张量的列表
        :param labels: 真实目标标签，N个张量的列表
        :return: 多框损失，一个标量
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 441, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 441)

        # 对每张图像
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)  # (n_objects, 441)

            # 对每个先验框，找到与之最大重叠的对象
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (441)

            # 我们不希望出现这样的情况：一个对象在我们的正例（非背景）先验框中未被表示 -
            # 1. 一个对象可能不是所有先验框的最佳对象，因此不在object_for_each_prior中。
            # 2. 基于阈值(0.5)，所有与该对象对应的先验框可能被指定为背景。

            # 为了解决这个问题 -
            # 首先，找到与每个对象具有最大重叠的先验框。
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # 然后，将每个对象分配给相应的最大重叠先验框。（这解决了问题1。）
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # 为了确保这些先验框满足条件，人为地给它们一个大于0.5的重叠。（这解决了问题2。）
            overlap_for_each_prior[prior_for_each_object] = 1.

            # 每个先验框的标签
            label_for_each_prior = labels[i][object_for_each_prior]  # (441)
            # 将与对象重叠小于阈值的先验框设为背景（无对象）
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (441)

            # 存储
            true_classes[i] = label_for_each_prior

            # 将中心-大小对象坐标编码为我们回归预测框的形式
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (441, 4)

        # 确定哪些先验框是正例（对象/非背景）
        positive_priors = true_classes != 0  # (N, 441)

        # 定位损失

        # 定位损失仅针对正例（非背景）先验框计算
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), 标量

        # 注意：使用torch.uint8（字节）张量进行索引时，当索引跨越多个维度（N和441）时，会将张量展平
        # 所以，如果predicted_locs的形状为(N, 441, 4)，则predicted_locs[positive_priors]将为(总正例数, 4)

        # 置信度损失

        # 置信度损失针对正例先验框和每个图像中最困难（最硬）的负例先验框计算
        # 也就是说，对于每张图像，
        # 我们将选取最困难的（neg_pos_ratio * n_positives）个负例先验框，即损失最大的框
        # 这称为困难负例挖掘 - 它集中在每个图像中最困难的负例上，并且最小化正/负不平衡

        # 每张图像的正例和困难负例先验框数量
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # 首先，计算所有先验框的损失
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 441)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 441)

        # 我们已经知道哪些先验框是正例
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # 接下来，找出哪些先验框是困难负例
        # 为此，仅按照损失递减顺序对每个图像中的负例先验框进行排序，并选取前n_hard_negatives个
        conf_loss_neg = conf_loss_all.clone()  # (N, 441)
        conf_loss_neg[positive_priors] = 0.  # (N, 441), 忽略正例先验框（永远不在top n_hard_negatives中）
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 441), 按困难程度递减排序
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 441)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 441)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # 如论文所述，仅对正例先验框取平均，尽管计算同时考虑了正例和困难负例先验框
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), 标量

        # 返回总损失
        return conf_loss + self.alpha * loc_loss


class tiny_detector_enhanced(nn.Module):
    """
    增强版tiny_detector，针对超小目标进行了优化
    添加了conv3_3特征图(56x56)以更好地检测超小目标
    """

    def __init__(self, n_classes):
        super(tiny_detector_enhanced, self).__init__()

        self.n_classes = n_classes

        self.base = VGGBase()
        self.l2_norm = L2Norm(512, 20)  # conv4_3的规范化层
        self.pred_convs = PredictionConvolutionsEnhanced(n_classes)

        # 先验框
        self.priors_cxcy = self.create_prior_boxes()
        # 转换到边界坐标格式，用于MultiBoxLoss中的jaccard计算
        self.priors_xy = cxcy_to_xy(self.priors_cxcy)

    def forward(self, image):
        """
        前向传播。

        :param image: 图像，维度为(N, 3, 224, 224)的张量
        :return: 位置和类别预测
        """
        # 获取多尺度特征图
        conv3_3_feats, conv4_3_feats, conv5_3_feats, pool5_feats = self.base(image)
        
        # 对conv4_3特征图进行L2归一化
        norm_conv4_3_feats = self.l2_norm(conv4_3_feats)
        
        # 进行预测
        locs, classes_scores = self.pred_convs(conv3_3_feats, norm_conv4_3_feats, conv5_3_feats, pool5_feats)

        return locs, classes_scores

    def create_prior_boxes(self):
        """
        为多尺度特征图创建先验框，包括新增的conv3_3特征图
        """
        prior_boxes = []
        
        # 特征图尺寸
        fmap_dims = {'conv3_3': 56, 'conv4_3': 28, 'conv5_3': 14, 'pool5': 7}
        
        # 先验框尺度配置
        obj_scales = {
            'conv3_3': [0.01, 0.025, 0.05, 0.08],  # 超小目标尺度
            'conv4_3': [0.1, 0.15, 0.2, 0.25],     # 小目标尺度
            'conv5_3': [0.3, 0.37, 0.44, 0.51, 0.58, 0.65],  # 修改为6个尺度
            'pool5': [0.7, 0.76, 0.82, 0.88, 0.94, 1.0]      # 修改为6个尺度
        }
        
        # 长宽比配置，确保与先验框数量匹配
        aspect_ratios = {
            'conv3_3': [1., 2., 0.5, 1.],  # 4个
            'conv4_3': [1., 2., 0.5, 1.],  # 4个
            'conv5_3': [1., 2., 0.5, 3., 1./3., 1.],  # 6个
            'pool5': [1., 2., 0.5, 3., 1./3., 1.]     # 6个
        }
        
        # 为conv3_3特征图(56x56)创建先验框
        for i in range(fmap_dims['conv3_3']):
            for j in range(fmap_dims['conv3_3']):
                cx = (j + 0.5) / fmap_dims['conv3_3']
                cy = (i + 0.5) / fmap_dims['conv3_3']
                
                for s, ar in zip(obj_scales['conv3_3'], aspect_ratios['conv3_3']):
                    prior_boxes.append([cx, cy, s * sqrt(ar), s / sqrt(ar)])
        
        # 为conv4_3特征图(28x28)创建先验框
        for i in range(fmap_dims['conv4_3']):
            for j in range(fmap_dims['conv4_3']):
                cx = (j + 0.5) / fmap_dims['conv4_3']
                cy = (i + 0.5) / fmap_dims['conv4_3']
                
                for s, ar in zip(obj_scales['conv4_3'], aspect_ratios['conv4_3']):
                    prior_boxes.append([cx, cy, s * sqrt(ar), s / sqrt(ar)])
        
        # 为conv5_3特征图(14x14)创建先验框
        for i in range(fmap_dims['conv5_3']):
            for j in range(fmap_dims['conv5_3']):
                cx = (j + 0.5) / fmap_dims['conv5_3']
                cy = (i + 0.5) / fmap_dims['conv5_3']
                
                for s, ar in zip(obj_scales['conv5_3'], aspect_ratios['conv5_3']):
                    prior_boxes.append([cx, cy, s * sqrt(ar), s / sqrt(ar)])
        
        # 为pool5特征图(7x7)创建先验框
        for i in range(fmap_dims['pool5']):
            for j in range(fmap_dims['pool5']):
                cx = (j + 0.5) / fmap_dims['pool5']
                cy = (i + 0.5) / fmap_dims['pool5']
                
                for s, ar in zip(obj_scales['pool5'], aspect_ratios['pool5']):
                    prior_boxes.append([cx, cy, s * sqrt(ar), s / sqrt(ar)])
        
        # 转换为PyTorch张量
        prior_boxes = torch.FloatTensor(prior_boxes).to(device)
        prior_boxes.clamp_(0, 1)  # 确保先验框坐标在[0,1]范围内
        
        return prior_boxes
    
    def detect_objects(self, predicted_locs, predicted_scores, min_score=0.2, max_overlap=0.5, top_k=200):
        """
        后处理步骤 - 将预测转换为边界框、标签和得分
        
        :param predicted_locs: 预测的边界框偏移，形状为 (N, n_priors, 4)
        :param predicted_scores: 预测的类别分数，形状为 (N, n_priors, n_classes)
        :param min_score: 最低置信度阈值
        :param max_overlap: 非极大值抑制的最大重叠阈值
        :param top_k: 每张图像保留的最多检测数量
        :return: 检测到的边界框坐标、标签和得分（每个图像的列表）
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, n_priors, n_classes)

        # 创建存储最终检测结果的列表
        all_images_boxes = []
        all_images_labels = []
        all_images_scores = []

        # 处理每张图像
        for i in range(batch_size):
            # 解码边界框坐标
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (n_priors, 4)

            # 对每个类别（跳过背景）进行处理
            image_boxes = []
            image_labels = []
            image_scores = []

            for c in range(1, self.n_classes):
                # 保留超过阈值的置信度
                class_scores = predicted_scores[i][:, c]  # (n_priors)
                score_above_min_score = class_scores > min_score  # (n_priors)
                n_above_min_score = score_above_min_score.sum().item()

                if n_above_min_score == 0:
                    continue

                # 提取超过阈值的框和置信度
                class_scores = class_scores[score_above_min_score]  # (n_qualified)
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # 按置信度降序排序
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)
                class_decoded_locs = class_decoded_locs[sort_ind]

                # 应用非极大值抑制 (NMS)
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_qualified)
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                # 按分数高低逐个处理框
                for box in range(class_decoded_locs.size(0)):
                    # 如果这个框已被抑制，跳过
                    if suppress[box] == 1:
                        continue

                    # 抑制与当前框重叠高的其他框
                    suppress = torch.max(suppress, (overlap[box] > max_overlap).to(torch.uint8))
                    # 不要抑制当前框
                    suppress[box] = 0

                # 保留未被抑制的框
                image_boxes.append(class_decoded_locs[suppress == 0])
                image_labels.append(torch.LongTensor((suppress == 0).sum().item() * [c]).to(device))
                image_scores.append(class_scores[suppress == 0])

            # 如果没有物体被检测到
            if len(image_boxes) == 0:
                all_images_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                all_images_labels.append(torch.LongTensor([0]).to(device))
                all_images_scores.append(torch.FloatTensor([0.]).to(device))
                continue

            # 将所有类别的检测结果合并
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # 保留得分最高的top_k个
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]
                image_boxes = image_boxes[sort_ind][:top_k]
                image_labels = image_labels[sort_ind][:top_k]

            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores


class PredictionConvolutionsEnhanced(nn.Module):
    """
    增强版预测卷积，使用多尺度特征图进行预测
    增加了conv3_3特征图的预测，以更好地检测超小目标
    """

    def __init__(self, n_classes):
        """
        :param n_classes: 不同类型对象的数量
        """
        super(PredictionConvolutionsEnhanced, self).__init__()

        self.n_classes = n_classes

        # 修改这里，使先验框数量与原始模型保持一致
        # 为不同特征图设置不同数量的先验框
        n_boxes = {'conv3_3': 4, 'conv4_3': 4, 'conv5_3': 6, 'pool5': 6}  # 从5改为6

        # Conv3_3特征图预测层 (超小目标) - 56x56
        self.loc_conv3_3 = nn.Conv2d(256, n_boxes['conv3_3'] * 4, kernel_size=3, padding=1)
        self.cl_conv3_3 = nn.Conv2d(256, n_boxes['conv3_3'] * n_classes, kernel_size=3, padding=1)
        
        # Conv4_3特征图预测层 (小目标) - 28x28
        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        
        # Conv5_3特征图预测层 (中等目标) - 14x14
        self.loc_conv5_3 = nn.Conv2d(512, n_boxes['conv5_3'] * 4, kernel_size=3, padding=1)
        self.cl_conv5_3 = nn.Conv2d(512, n_boxes['conv5_3'] * n_classes, kernel_size=3, padding=1)
        
        # Pool5特征图预测层 (大目标) - 7x7
        self.loc_pool5 = nn.Conv2d(512, n_boxes['pool5'] * 4, kernel_size=3, padding=1)
        self.cl_pool5 = nn.Conv2d(512, n_boxes['pool5'] * n_classes, kernel_size=3, padding=1)
        
        # 初始化卷积参数
        self.init_conv2d()

    def init_conv2d(self):
        """
        初始化卷积参数。
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv3_3_feats, conv4_3_feats, conv5_3_feats, pool5_feats):
        """
        前向传播。

        :param conv3_3_feats: conv3_3特征图，用于检测超小目标 (N, 256, 56, 56)
        :param conv4_3_feats: conv4_3特征图，用于检测小目标 (N, 512, 28, 28)
        :param conv5_3_feats: conv5_3特征图，用于检测中等目标 (N, 512, 14, 14)
        :param pool5_feats: pool5特征图，用于检测大目标 (N, 512, 7, 7)
        :return: 所有特征图的位置和类别预测
        """
        batch_size = conv4_3_feats.size(0)
        
        # Conv3_3 预测 (56x56特征图，适合超小目标)
        l_conv3_3 = self.loc_conv3_3(conv3_3_feats)  # (N, 4*4, 56, 56)
        l_conv3_3 = l_conv3_3.permute(0, 2, 3, 1).contiguous()
        l_conv3_3 = l_conv3_3.view(batch_size, -1, 4)  # (N, 12544, 4)，56*56*4=12544个框
        
        c_conv3_3 = self.cl_conv3_3(conv3_3_feats)  # (N, 4*n_classes, 56, 56)
        c_conv3_3 = c_conv3_3.permute(0, 2, 3, 1).contiguous()
        c_conv3_3 = c_conv3_3.view(batch_size, -1, self.n_classes)  # (N, 12544, n_classes)
        
        # Conv4_3 预测 (28x28特征图，适合小目标)
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  # (N, 4*4, 28, 28)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()  # (N, 28, 28, 4*4)
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)  # (N, 3136, 4)，28*28*4=3136个框
        
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)  # (N, 4*n_classes, 28, 28)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.n_classes)  # (N, 3136, n_classes)
        
        # Conv5_3 预测 (14x14特征图，适合中等目标)
        l_conv5_3 = self.loc_conv5_3(conv5_3_feats)  # (N, 6*4, 14, 14)
        l_conv5_3 = l_conv5_3.permute(0, 2, 3, 1).contiguous()
        l_conv5_3 = l_conv5_3.view(batch_size, -1, 4)  # (N, 1176, 4)，14*14*6=1176个框
        
        c_conv5_3 = self.cl_conv5_3(conv5_3_feats)  # (N, 6*n_classes, 14, 14)
        c_conv5_3 = c_conv5_3.permute(0, 2, 3, 1).contiguous()
        c_conv5_3 = c_conv5_3.view(batch_size, -1, self.n_classes)  # (N, 1176, n_classes)
        
        # Pool5 预测 (7x7特征图，适合大目标)
        l_pool5 = self.loc_pool5(pool5_feats)  # (N, 6*4, 7, 7)
        l_pool5 = l_pool5.permute(0, 2, 3, 1).contiguous()
        l_pool5 = l_pool5.view(batch_size, -1, 4)  # (N, 294, 4)，7*7*6=294个框
        
        c_pool5 = self.cl_pool5(pool5_feats)  # (N, 6*n_classes, 7, 7)
        c_pool5 = c_pool5.permute(0, 2, 3, 1).contiguous()
        c_pool5 = c_pool5.view(batch_size, -1, self.n_classes)  # (N, 294, n_classes)
        
        # 合并所有特征图的预测
        locs = torch.cat([l_conv3_3, l_conv4_3, l_conv5_3, l_pool5], dim=1)
        classes_scores = torch.cat([c_conv3_3, c_conv4_3, c_conv5_3, c_pool5], dim=1)
        
        return locs, classes_scores


```

train.py

```python
import time
import os
import signal
import sys
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import tiny_detector, MultiBoxLoss, tiny_detector_enhanced
from datasets import PascalVOCDataset
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# 数据参数
data_folder = './dataset/VOCdevkit'  # 数据文件根路径
keep_difficult = True  # 使用被认为难以检测的对象
n_classes = len(label_map)  # 不同类型对象的数量

# 学习参数
total_epochs = 230  # 训练的轮次数
batch_size = 32  # 批量大小
workers = 4  # 数据加载器中的工作线程数
print_freq = 100  # 每隔多少批次打印训练状态
save_freq = 500  # 每多少批次保存一次临时检查点
lr = 1e-3  # 学习率
decay_lr_at = [150, 190]  # 在这些轮次之后衰减学习率
decay_lr_to = 0.1  # 将学习率衰减到现有学习率的这一比例
momentum = 0.9  # 动量
weight_decay = 5e-4  # 权重衰减
checkpoint_path = './checkpoints'  # 检查点保存路径

# 确保检查点目录存在
os.makedirs(checkpoint_path, exist_ok=True)


def save_checkpoint(epoch, model, optimizer, batch_idx=None, loss=None, is_best=False):
    """
    保存训练检查点
    
    :param epoch: 当前训练的epoch
    :param model: 模型
    :param optimizer: 优化器
    :param batch_idx: 当前批次索引(可选)
    :param loss: 当前损失(可选)
    :param is_best: 是否为最佳模型(基于损失)
    """
    # 最新检查点的文件名
    latest_filename = f'{checkpoint_path}/checkpoint_latest.pth.tar'
    
    state = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }
    
    # 保存最新检查点
    torch.save(state, latest_filename)
    
    # 如果是最佳模型，额外保存一个best版本
    if is_best:
        best_filename = f'{checkpoint_path}/checkpoint_best.pth.tar'
        torch.save(state, best_filename)
        print(f'\n【最佳模型】已保存 (loss: {loss:.4f}): {best_filename}\n')
    else:
        print(f'\n检查点已保存: {latest_filename}\n')


def load_checkpoint(model, optimizer):
    """
    加载最新的检查点
    """
    start_epoch = 0
    start_batch = 0
    
    latest_checkpoint = f'{checkpoint_path}/checkpoint_latest.pth.tar'
    
    if os.path.isfile(latest_checkpoint):
        print(f"Loading checkpoint '{latest_checkpoint}'")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        
        # 安全地获取epoch和batch_idx
        start_epoch = checkpoint.get('epoch', 0)
        
        # 安全处理batch_idx
        batch_idx = checkpoint.get('batch_idx')
        start_batch = 0 if batch_idx is None else batch_idx + 1
        
        # 如果是epoch结束的检查点，从下一个epoch开始
        if batch_idx is None:
            start_epoch += 1
            start_batch = 0
            
        # 加载模型和优化器状态
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        print(f"=> 成功加载检查点 (epoch {start_epoch}, batch {start_batch})")
        
        return start_epoch, start_batch
    else:
        print("=> 未找到检查点，从头开始训练")
        return 0, 0


def main():
    """
    训练。
    """
    # 注册信号处理器，用于捕获中断
    global current_epoch, current_batch, current_model, current_optimizer
    current_epoch = 0
    current_batch = 0
    
    def signal_handler(sig, frame):
        print('\n捕获到中断信号，保存检查点...')
        save_checkpoint(current_epoch, current_model, current_optimizer, current_batch)
        print('检查点已保存，退出程序')
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # ===== 修改这部分代码 =====
    # 1. 加载原有模型检查点
    latest_checkpoint = f'{checkpoint_path}/checkpoint_latest.pth.tar'
    if os.path.isfile(latest_checkpoint):
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        old_model_dict = checkpoint['model']
        old_optimizer_dict = checkpoint['optimizer']
        start_epoch = checkpoint.get('epoch', 0)
        batch_idx = checkpoint.get('batch_idx')
        start_batch = 0 if batch_idx is None else batch_idx + 1
        
        # 如果是epoch结束的检查点，从下一个epoch开始
        if batch_idx is None:
            start_epoch += 1
            start_batch = 0
    else:
        # 无检查点，从头开始
        old_model_dict = None
        old_optimizer_dict = None
        start_epoch = 0
        start_batch = 0
    
    # 2. 创建新的增强模型
    model = tiny_detector_enhanced(n_classes=n_classes)
    model = model.to(device)  # 先将模型移到GPU

    # 然后使用已在GPU上的先验框创建损失函数
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy)
    criterion = criterion.to(device)  # 确保损失函数所有部分都在GPU上

    optimizer = torch.optim.SGD(params=model.parameters(),
                               lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    # 在main函数中
    # 3. 如果有旧模型权重，进行迁移学习
    if old_model_dict is not None:
        # 筛选和复制可复用的权重
        new_model_dict = model.state_dict()
        incompatible_layers = []
        
        for name, param in old_model_dict.items():
            # 如果参数存在于新模型中
            if name in new_model_dict:
                # 检查形状是否匹配
                if param.shape == new_model_dict[name].shape:
                    new_model_dict[name] = param
                else:
                    incompatible_layers.append(name)
            
        # 加载筛选后的权重
        model.load_state_dict(new_model_dict, strict=False)
        print("成功加载原有模型的兼容权重进行迁移学习")
        if incompatible_layers:
            print(f"以下层由于形状不兼容未能加载: {incompatible_layers}")
        
        # 可选：加载优化器状态
        if old_optimizer_dict is not None:
            try:
                optimizer.load_state_dict(old_optimizer_dict)
                print("成功加载优化器状态")
            except:
                print("优化器状态无法加载，使用新的优化器")
    
    # 4. 冻结骨干网络权重进行微调训练
    for name, param in model.named_parameters():
        if 'base' in name and 'conv3_3' not in name:
            param.requires_grad = False
    # ===== 修改结束 =====
    
    # 移至默认设备
    model = model.to(device)
    criterion = criterion.to(device)
    
    # 设置全局变量，用于信号处理器
    current_model = model
    current_optimizer = optimizer
    current_epoch = start_epoch
    
    # 删除旧的检查点加载逻辑
    # start_epoch, start_batch = load_checkpoint(model, optimizer)
    
    # 自定义数据加载器
    train_dataset = PascalVOCDataset(data_folder,
                                    split='train',
                                    keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                              collate_fn=train_dataset.collate_fn, num_workers=workers,
                                              pin_memory=True)
    
    # 删除重复的迁移学习代码
    
    # 添加跟踪最佳损失的变量
    best_loss = float('inf')
    
    # 训练周期
    try:
        for epoch in range(start_epoch, total_epochs):
            current_epoch = epoch
            
            # 在特定epoch衰减学习率
            if epoch in decay_lr_at:
                adjust_learning_rate(optimizer, decay_lr_to)
            
            # 一个epoch的训练，获取返回的平均损失
            epoch_loss = train(train_loader=train_loader,
                 model=model,
                 criterion=criterion,
                 optimizer=optimizer,
                 epoch=epoch,
                 start_batch=start_batch if epoch == start_epoch else 0)
            
            # 重置起始批次（仅第一个epoch使用）
            start_batch = 0
            
            # 检查是否为最佳模型
            is_best = epoch_loss < best_loss
            if is_best:
                best_loss = epoch_loss
                print(f"发现新的最佳模型! Loss: {best_loss:.4f}")
            
            # 保存epoch检查点和最佳模型(如果需要)
            save_checkpoint(epoch, model, optimizer, None, epoch_loss, is_best)
    
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        print("保存紧急检查点...")
        save_checkpoint(current_epoch, current_model, current_optimizer, current_batch)


def train(train_loader, model, criterion, optimizer, epoch, start_batch=0):
    """
    一个轮次的训练，带有检查点功能。

    :param train_loader: 用于训练数据的DataLoader
    :param model: 模型
    :param criterion: MultiBox损失函数
    :param optimizer: 优化器
    :param epoch: 轮次编号
    :param start_batch: 从哪个批次开始 (用于恢复训练)
    """
    global current_batch
    model.train()  # 训练模式启用dropout

    batch_time = AverageMeter()  # 前向传播 + 反向传播时间
    data_time = AverageMeter()  # 数据加载时间
    losses = AverageMeter()  # 损失

    start = time.time()
    
    # 跳过开始批次之前的批次
    train_iter = enumerate(train_loader)
    if start_batch > 0:
        print(f"Skipping to batch {start_batch}...")
        # 跳过批次
        for i in range(start_batch):
            try:
                next(train_iter)
            except StopIteration:
                print(f"Warning: 批次索引 {start_batch} 超出数据集范围，从头开始")
                train_iter = enumerate(train_loader)
                break

    # 批次
    for i, (images, boxes, labels, _) in train_iter:
        current_batch = i
        try:
            data_time.update(time.time() - start)

            # 移至默认设备
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # 初始化变量为None，防止异常时引用错误
            predicted_locs, predicted_scores = None, None
            
            # 前向传播
            predicted_locs, predicted_scores = model(images)

            # 损失
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)

            start = time.time()

            # 打印状态
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                      batch_time=batch_time,
                                                                      data_time=data_time, loss=losses))
            
            # 只在必要时保存检查点，不再按批次频率保存中间检查点
            # 这里可以完全移除，或者保留但降低频率
            if (i + 1) % save_freq == 0 and i > 0:
                # 先保存latest检查点
                save_checkpoint(epoch, model, optimizer, i, losses.avg)
                
                # 检查是否为最佳模型（将best_loss作为全局变量）
                global best_loss
                if losses.avg < best_loss:
                    best_loss = losses.avg
                    # 单独保存最佳模型
                    save_checkpoint(epoch, model, optimizer, i, losses.avg, is_best=True)
                    print(f"【批次级别】发现新的最佳模型! Loss: {best_loss:.4f}")
        
        except Exception as e:
            print(f"批次 {i} 处理时出错: {e}")
            # 跳过问题批次，继续下一个
            continue

    del predicted_locs, predicted_scores, images, boxes, labels  # 释放一些内存，因为它们的历史可能被存储
    
    # 在函数末尾返回平均损失
    return losses.avg


if __name__ == '__main__':
    main()

```

utils.py

```python
import json
import os
import torch
import random
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 标签映射
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # 反向映射
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}


def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()
    for object in root.iter('object'):

        difficult = int(object.find('difficult').text == '1')

        label = object.find('name').text.lower().strip()
        if label not in label_map:
            continue

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}


def create_data_lists(voc07_path, voc12_path, output_folder):
    """
    创建图像列表、图像中物体的边界框和标签，并将这些保存到文件中。

    :param voc07_path: 'VOC2007'文件夹的路径
    :param voc12_path: 'VOC2012'文件夹的路径
    :param output_folder: 必须保存JSON文件的文件夹
    """
    voc07_path = os.path.abspath(voc07_path)
    voc12_path = os.path.abspath(voc12_path)

    train_images = list()
    train_objects = list()
    n_objects = 0

    # 训练数据
    for path in [voc07_path,voc12_path]:

        # 查找训练数据中图像的ID
        with open(os.path.join(path, 'ImageSets/Main/trainval.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            # 解析标注的XML文件
            objects = parse_annotation(os.path.join(path, 'Annotations', id + '.xml'))
            if len(objects['boxes']) == 0:
                continue
            n_objects += len(objects['boxes'])
            train_objects.append(objects)
            train_images.append(os.path.join(path, 'JPEGImages', id + '.jpg'))

    assert len(train_objects) == len(train_images)

    # 保存到文件
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  # 也保存标签映射

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_objects, os.path.abspath(output_folder)))

    # 测试数据
    test_images = list()
    test_objects = list()
    n_objects = 0

    # 查找测试数据中图像的ID
    with open(os.path.join(voc07_path, 'ImageSets/Main/test.txt')) as f:
        ids = f.read().splitlines()

    for id in ids:
        # 解析标注的XML文件
        objects = parse_annotation(os.path.join(voc07_path, 'Annotations', id + '.xml'))
        if len(objects['boxes']) == 0:
            continue
        test_objects.append(objects)
        n_objects += len(objects['boxes'])
        test_images.append(os.path.join(voc07_path, 'JPEGImages', id + '.jpg'))

    assert len(test_objects) == len(test_images)

    # 保存到文件
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print('\nThere are %d test images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), n_objects, os.path.abspath(output_folder)))


def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):
    """
    计算检测到的物体的平均精度(mAP)。
    这里的map指标遵循的VOC2007的标准，具体地：
    统一用IOU>0.5作为目标框是否准召的标准
    AP的计算标准采用召回分别为 0:0.05:1 时的准确率平均得到

    :param det_boxes: 张量列表，每个图像一个张量，包含检测到的物体的边界框
    :param det_labels: 张量列表，每个图像一个张量，包含检测到的物体的标签
    :param det_scores: 张量列表，每个图像一个张量，包含检测到的物体标签的分数
    :param true_boxes: 张量列表，每个图像一个张量，包含实际物体的边界框
    :param true_labels: 张量列表，每个图像一个张量，包含实际物体的标签
    :param true_difficulties: 张量列表，每个图像一个张量，包含实际物体的难度(0或1)
    :return: 所有类别的平均精度列表，平均精度均值(mAP)
    """
    # 确保所有张量在同一设备上
    device = true_boxes[0].device if true_boxes else torch.device("cpu")
    
    # 创建张量时指定设备
    true_images = torch.LongTensor([i for i in range(len(true_labels)) 
                                   for _ in range(true_labels[i].size(0))]).to(device)
    
    # 确保所有张量列表长度相同，即图像数量
    assert len(det_boxes) == len(det_labels) == len(det_scores) == \
           len(true_boxes) == len(true_labels) == len(true_difficulties)
    n_classes = len(label_map)

    # 将所有(真实)物体存储在单个连续张量中，同时跟踪它来自哪个图像
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(device)  # (n_objects), n_objects: 所有图像中物体的总数
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # 将所有检测结果存储在单个连续张量中，同时跟踪它来自哪个图像
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # 为每个类别(除背景外)计算AP
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # 只提取具有此类别的物体
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # 忽略难检测的物体

        # 跟踪该类别中哪些真实物体已经被'检测'到
        # 到目前为止，没有
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8)
        true_class_boxes_detected = true_class_boxes_detected.to(device)  # (n_class_objects)

        # 只提取具有此类别的检测结果
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # 按置信度/分数降序排序检测结果
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # 按照分数降序的顺序，检查是真阳性还是假阳性
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), 标量

            # 找出同一图像中具有此类别的物体、它们的难度以及它们是否已被检测到
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img, 4)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # 如果此图像中没有此类物体，则检测为假阳性
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # 找出此检测结果与此类别中此图像物体的最大重叠
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - 标量

            # 'ind'是这些图像级别张量'object_boxes'，'object_difficulties'中物体的索引
            # 在原始类级别张量'true_class_boxes'等中，'ind'对应的物体索引为...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
            # 我们需要'original_ind'来更新'true_class_boxes_detected'

            # 如果最大重叠大于0.5阈值，则匹配成功
            if max_overlap.item() > 0.5:
                # 如果匹配的物体被标记为'difficult'，则忽略它
                if object_difficulties[ind] == 0:
                    # 如果此物体尚未被检测到，则为真阳性
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # 此物体现已被检测到/计算
                    # 否则，它是假阳性（因为此物体已被计算）
                    else:
                        false_positives[d] = 1
            # 否则，检测结果出现在不同于实际物体的位置，是假阳性
            else:
                false_positives[d] = 1

        # 计算按分数降序排列的每个检测点的累积精度和召回率
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # 找出对应于高于阈值't'的召回率的精度的最大值的平均值
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c在[1, n_classes - 1]范围内

    # 计算平均精度均值(mAP)
    mean_average_precision = average_precisions.mean().item()

    # 在字典中保存各类别的平均精度
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision


def xy_to_cxcy(xy):
    """
    将边界框从边界坐标(x_min, y_min, x_max, y_max)转换为中心-尺寸坐标(c_x, c_y, w, h)。

    :param xy: 边界坐标中的边界框，尺寸为(n_boxes, 4)的张量
    :return: 中心-尺寸坐标中的边界框，尺寸为(n_boxes, 4)的张量
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    """
    将边界框从中心-尺寸坐标(c_x, c_y, w, h)转换为边界坐标(x_min, y_min, x_max, y_max)。

    :param cxcy: 中心-尺寸坐标中的边界框，尺寸为(n_boxes, 4)的张量
    :return: 边界坐标中的边界框，尺寸为(n_boxes, 4)的张量
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    将边界框(中心-尺寸形式)相对于相应的先验框(也是中心-尺寸形式)进行编码。

    对于中心坐标，找到相对于先验框的偏移量，并按先验框的大小进行缩放。
    对于尺寸坐标，按先验框的大小进行缩放，并转换到对数空间。

    在模型中，我们以这种编码形式预测边界框坐标。

    :param cxcy: 中心-尺寸坐标中的边界框，尺寸为(n_priors, 4)的张量
    :param priors_cxcy: 必须相对其进行编码的先验框，尺寸为(n_priors, 4)的张量
    :return: 编码后的边界框，尺寸为(n_priors, 4)的张量
    """

    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    解码模型预测的边界框坐标，因为它们以上面提到的形式编码。

    它们被解码为中心-尺寸坐标。

    这是上面函数的逆运算。

    :param gcxgcy: 编码后的边界框，即模型的输出，尺寸为(n_priors, 4)的张量
    :param priors_cxcy: 定义编码的先验框，尺寸为(n_priors, 4)的张量
    :return: 中心-尺寸形式的解码后边界框，尺寸为(n_priors, 4)的张量
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


def find_intersection(set_1, set_2):
    """
    查找两组边界坐标框之间每个框组合的交集。

    :param set_1: 集合1，维度为(n1, 4)的张量
    :param set_2: 集合2，维度为(n2, 4)的张量
    :return: 集合1中每个框相对于集合2中每个框的交集，维度为(n1, n2)的张量
    """

    # PyTorch自动广播单例维度
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    查找两组边界坐标框之间每个框组合的Jaccard重叠(IoU)。

    :param set_1: 集合1，维度为(n1, 4)的张量
    :param set_2: 集合2，维度为(n2, 4)的张量
    :return: 集合1中每个框相对于集合2中每个框的Jaccard重叠，维度为(n1, n2)的张量
    """

    # 查找交集
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # 查找两组中每个框的面积
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # 查找并集
    # PyTorch自动广播单例维度
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)



def expand(image, boxes, filler):
    """
    通过将图像放置在更大的填充材料画布上执行放大操作。

    有助于学习检测更小的物体。

    :param image: 图像，维度为(3, original_h, original_w)的张量
    :param boxes: 边界坐标中的边界框，维度为(n_objects, 4)的张量
    :param filler: 填充材料的RGB值，类似[R, G, B]的列表
    :return: 扩展后的图像，更新后的边界框坐标
    """
    # 计算建议扩展(放大)图像的尺寸
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # 用填充材料创建这样的图像
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
    # 注意 - 不要像 new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w) 这样使用expand()
    # 因为所有扩展的值都将共享相同的内存，所以更改一个像素将更改所有像素

    # 将原始图像放置在这个新图像中的随机坐标处(原点在图像的左上角)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # 相应地调整边界框坐标
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(
        0)  # (n_objects, 4), n_objects是此图像中的物体数量

    return new_image, new_boxes


def random_crop(image, boxes, labels, difficulties):
    """
    按照论文中所述的方式执行随机裁剪。有助于学习检测更大和部分物体。

    请注意，某些物体可能会被完全裁剪掉。

    :param image: 图像，维度为(3, original_h, original_w)的张量
    :param boxes: 边界坐标中的边界框，维度为(n_objects, 4)的张量
    :param labels: 物体的标签，维度为(n_objects)的张量
    :param difficulties: 这些物体检测的难度，维度为(n_objects)的张量
    :return: 裁剪后的图像，更新后的边界框坐标，更新后的标签，更新后的难度
    """
    original_h = image.size(1)
    original_w = image.size(2)
    # 继续选择最小重叠直到成功裁剪
    while True:
        # 随机抽取最小重叠值
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None'表示不裁剪

        # 如果不裁剪
        if min_overlap is None:
            return image, boxes, labels, difficulties

        # 尝试此最小重叠选择最多50次
        # 当然，这在论文中没有提到，但是在论文作者的原始Caffe库中选择了50
        max_trials = 50
        for _ in range(max_trials):
            # 裁剪尺寸必须在原始尺寸的[0.3, 1]范围内
            # 注意 - 论文中是[0.1, 1]，但在作者的库中实际上是[0.3, 1]
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # 纵横比必须在[0.5, 2]范围内
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # 裁剪坐标(原点在图像的左上角)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            # 计算裁剪和边界框之间的Jaccard重叠
            overlap = find_jaccard_overlap(crop.unsqueeze(0),
                                           boxes)  # (1, n_objects), n_objects是此图像中的物体数量
            overlap = overlap.squeeze(0)  # (n_objects)

            # 如果没有单个边界框具有大于最小值的Jaccard重叠，则重试
            if overlap.max().item() < min_overlap:
                continue

            # 裁剪图像
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # 找出原始边界框的中心
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

            # 找出中心在裁剪区域内的边界框
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)  # (n_objects)，一个Torch uInt8/Byte张量，可以用作布尔索引

            # 如果没有单个边界框的中心在裁剪区域内，则重试
            if not centers_in_crop.any():
                continue

            # 丢弃不符合此标准的边界框
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            # 计算边界框在裁剪中的新坐标
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2]是[left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:]是[right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels, new_difficulties


def flip(image, boxes):
    """
    水平翻转图像。

    :param image: 图像，PIL图像
    :param boxes: 边界坐标中的边界框，维度为(n_objects, 4)的张量
    :return: 翻转后的图像，更新后的边界框坐标
    """
    # 翻转图像
    new_image = FT.hflip(image)

    # 翻转框
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes


def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
    """
    调整图像大小。
    对于SSD300，调整为(300, 300)。
    对于我们的演示，调整为(224, 224)。

    由于在此过程中计算了边界框的百分比/分数坐标(相对于图像尺寸)，
    你可以选择保留这些坐标。

    :param image: 图像，PIL图像
    :param boxes: 边界坐标中的边界框，维度为(n_objects, 4)的张量
    :return: 调整大小后的图像，更新后的边界框坐标(或分数坐标，在这种情况下它们保持不变)
    """
    # 调整图像大小
    new_image = FT.resize(image, dims)

    # 调整边界框大小
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # 百分比坐标

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


def photometric_distort(image):
    """
    以随机顺序扭曲亮度、对比度、饱和度和色调，每个都有50%的几率。

    :param image: 图像，PIL图像
    :return: 扭曲后的图像
    """
    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                # Caffe库使用'hue_delta'为18 - 我们除以255是因为PyTorch需要归一化值
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe库对亮度、对比度和饱和度使用'lower'和'upper'值为0.5和1.5
                adjust_factor = random.uniform(0.5, 1.5)

            # 应用这个扭曲
            new_image = d(new_image, adjust_factor)

    return new_image


def transform(image, boxes, labels, difficulties, split):
    """
    应用上述变换。

    :param image: 图像，PIL图像
    :param boxes: 边界坐标中的边界框，维度为(n_objects, 4)的张量
    :param labels: 物体的标签，维度为(n_objects)的张量
    :param difficulties: 这些物体检测的难度，维度为(n_objects)的张量
    :param split: 'TRAIN'或'TEST'之一，因为应用了不同的变换集
    :return: 变换后的图像，变换后的边界框坐标，变换后的标签，变换后的难度
    """
    assert split in {'TRAIN', 'TEST'}

    # 我们的基础VGG从torchvision训练的ImageNet数据的均值和标准差
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    # 对评估/测试跳过以下操作
    if split == 'TRAIN':
        # 一系列随机顺序的光度扭曲，每种都有50%的发生几率，如Caffe库中一样
        new_image = photometric_distort(new_image)

        # 将PIL图像转换为Torch张量
        new_image = FT.to_tensor(new_image)

        # 扩展图像(放大)有50%的几率 - 有助于训练小物体的检测
        # 用我们的基础VGG训练的ImageNet数据的均值填充周围空间
        if random.random() < 0.5:
            new_image, new_boxes = expand(new_image, boxes, filler=mean)

        # 随机裁剪图像(放大)
        new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels,
                                                                         new_difficulties)

        # 将Torch张量转换为PIL图像
        new_image = FT.to_pil_image(new_image)

        # 有50%的几率翻转图像
        if random.random() < 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)

    # 将图像调整为(224, 224) - 这也将边界框的绝对坐标转换为它们的分数形式
    new_image, new_boxes = resize(new_image, new_boxes, dims=(224, 224))

    # 将PIL图像转换为Torch张量
    new_image = FT.to_tensor(new_image)

    # 用我们的基础VGG训练的ImageNet数据的均值和标准差归一化
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels, new_difficulties


def adjust_learning_rate(optimizer, scale):
    """
    按指定因子缩放学习率。

    :param optimizer: 需要缩小学习率的优化器。
    :param scale: 与学习率相乘的因子。
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))


def accuracy(scores, targets, k):
    """
    计算top-k准确率，根据预测和真实标签。

    :param scores: 模型的分数
    :param targets: 真实标签
    :param k: top-k准确率中的k
    :return: top-k准确率
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D张量
    return correct_total.item() * (100.0 / batch_size)


def save_checkpoint(epoch, model, optimizer):
    """
    保存模型检查点。

    :param epoch: 轮次数
    :param model: 模型
    :param optimizer: 优化器
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint.pth.tar'
    torch.save(state, filename)


class AverageMeter(object):
    """
    跟踪度量指标的最近值、平均值、总和和计数。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


```

detect.py

```python
import json
import os.path as osp
from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
# 导入增强型模型
from model import tiny_detector_enhanced

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载检查点
best_checkpoint_path = 'checkpoints/checkpoint_best.pth.tar'
if os.path.isfile(best_checkpoint_path):
    checkpoint_path = best_checkpoint_path
else:
    checkpoint_path = 'checkpoints/checkpoint_latest.pth.tar'
    
checkpoint = torch.load(checkpoint_path, map_location=device)
print(f"使用{os.path.basename(checkpoint_path)}模型，损失值：{checkpoint.get('loss', 'unknown')}")

start_epoch = checkpoint['epoch'] + 1

# 创建增强型模型实例并加载权重
model = tiny_detector_enhanced(n_classes=len(label_map))
model.load_state_dict(checkpoint['model'])
model = model.to(device)
model.eval()

print(f'\nLoaded checkpoint from epoch {start_epoch}.\n')

# 设置检测变换（与训练保持一致非常重要）
resize = transforms.Resize((224, 224))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def detect(original_image, min_score, max_overlap, top_k, max_size_ratio=0.3, image_name=None):
    """
    使用训练好的小目标检测器在图像中检测对象，并可视化结果。
    添加大目标过滤功能，只检测小目标
    添加输出检测结果到JSON文件功能

    :param original_image: 图像，一个PIL Image对象
    :param min_score: 检测框被视为某一类别匹配的最小阈值
    :param max_overlap: 两个框可以具有的最大重叠，以便通过非极大值抑制(NMS)不抑制得分较低的框
    :param top_k: 如果所有类别的检测结果很多，只保留前'k'个
    :param max_size_ratio: 目标最大允许尺寸占图像的比例，超过此比例的目标将被过滤掉
    :param image_name: 图像文件名，用于JSON输出
    :return: 标注后的图像，一个PIL Image对象
    """
    # 确保图像是 RGB 模式
    if original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')
        
    # 分步骤进行转换以避免错误
    img_resized = resize(original_image)
    img_tensor = to_tensor(img_resized)
    img_normalized = normalize(img_tensor)
    image = img_normalized.to(device)

    # 预测
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # 预测的边界框和分数
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # 将检测结果转换为 CPU 张量
    det_boxes = det_boxes[0].to('cpu')
    det_labels = det_labels[0].to('cpu').tolist()
    det_scores = det_scores[0].to('cpu').tolist()

    # 过滤掉低置信度的检测结果
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # 过滤目标 - 只保留小目标
    filtered_boxes = []
    filtered_labels = []
    filtered_scores = []
    
    img_width, img_height = original_image.width, original_image.height
    max_width = img_width * max_size_ratio
    max_height = img_height * max_size_ratio
    
    for i, box in enumerate(det_boxes):
        width = box[2] - box[0]
        height = box[3] - box[1]
        
        # 只保留小目标（宽度和高度都小于设定阈值）
        if width <= max_width and height <= max_height:
            filtered_boxes.append(box)
            filtered_labels.append(det_labels[i])
            filtered_scores.append(det_scores[i])
    
    # 如果没有符合条件的小目标
    if not filtered_boxes:
        print("未检测到符合尺寸要求的小目标")
        return original_image
    
    # 将过滤后的结果转换回张量格式
    det_boxes = torch.stack(filtered_boxes)
    det_labels = filtered_labels
    det_scores = filtered_scores

    det_labels_text = [rev_label_map[l] if isinstance(l, int) else l for l in det_labels]

    annotated_image = original_image.copy()  # 创建副本以避免修改原图
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()
    
    # 准备JSON输出数据
    if image_name is None:
        image_name = "unknown.jpg"
    
    json_data = {
        "image_name": image_name,
        "objects": []
    }
    
    for i in range(det_boxes.size(0)):
        box_location = det_boxes[i].tolist()
        
        # 添加到JSON数据
        json_data["objects"].append({
            "label": det_labels_text[i],
            "bbox": [
                int(box_location[0]),  # x_min
                int(box_location[1]),  # y_min
                int(box_location[2]),  # x_max
                int(box_location[3])   # y_max
            ]
        })

        # 绘制边界框
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels_text[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[det_labels_text[i]])

        # 绘制标签文本和置信度
        score = det_scores[i]
        text = f"{det_labels_text[i].upper()} ({score:.2f})"
        bbox = font.getbbox(text)
        text_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1],
                            box_location[0] + text_size[0] + 4., box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels_text[i]])
        draw.text(xy=text_location, text=text, fill='white', font=font)
    del draw
    
    # 保存JSON结果
    with open('李玉泽_predicted_result.json', 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"检测结果已保存到 李玉泽_predicted_result.json")

    return annotated_image


if __name__ == '__main__':
    img_path = './测试图片/Maodie_Dora.jpg'
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    
    # 获取图像文件名
    image_name = osp.basename(img_path)
    
    # 检测图像 - 使用max_size_ratio=过滤掉不同目标，并传入图像名
    result_image = detect(original_image, min_score=0.12, max_overlap=0.1, top_k=200, 
                         max_size_ratio=0.9, image_name=image_name)
    
    # 显示图像
    result_image.show()
    # 保存图像
    result_image.save('./detection_small_objects_only.jpg')


```

## 五、实验结果显示

对于给定的两张图片，其原始效果如下：

![alt](C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计part2\S_TOTAL.jpg)

![alt](C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计part222\images\D_TOTAL.jpg)

在训练过程中，我们训练的模型保存在checkpoints目录下，包含两个，一个是训练到最后一次的模型，一次是Loss最小的best模型，在detect.py中我们加载最优模型，并使用最优模型对上面的两张图片进行目标识别。只需要在detect文件中，更改预测图像的目录，运行代码，即可得出以下图片，附有检测框以及类别，同时得到 JSON 格式的预测结果，上面的两个图片的 JSON 格式预测结果分别保存在0473_2312654_李玉泽_课程设计Part2_D和0473_2312654_李玉泽_课程设计Part2_S目录下。

最终的预测结果如下所示：

![](C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计part222\results\S_TOTAL.jpg)

![](C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计part222\results\D_TOTAL.jpg)

可以看出，上述识别结果较好，但是，对于极小的物体，虽然我更改了特征图的大小以期望使其能够正确识别较小的目标，其仍然有部分较小的目标没有被识别到，因此程序仍有改进空间。

除此之外，我还提供了一些图片，存于测试图片目录下，可以直接通过detect.py文件运行获取其目标识别结果与位置信息等，批阅老师可以自行验证，下图是其中一个样例，较好地识别了一只猫和一个人同框地结果：

<div style="display: flex; justify-content: space-between;">
    <img src="C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计part2\测试图片\Maodie_Dora.jpg" width="45%" />
    <img src="C:\Users\cassi\OneDrive\桌面\当前资料\机器视觉\0473_2312654_李玉泽_课程设计part2\detection_small_objects_only.jpg" width="45%" />
</div>


## 六、实验分析总结

#### 6.1 模型优点

模型针对小目标检测做了显著优化，加入了 **conv3_3 层检测超小目标**，检测过程中添加 **目标大小过滤策略**，能有效剔除误检的大框（如运动场背景）。

#### 6.2 模型缺陷

检测精度仍受训练数据多样性限制，需扩展训练样本，图中左上角等极小目标在复杂背景中偶尔仍会漏检，未来可以引入 FPN 或 transformer backbone 进一步增强，提高鲁棒性和准确性。

