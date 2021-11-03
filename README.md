# 本教程来自 Tina 姐公众号【医学图像人工智能实战营】欢迎关注

我们使用 MNIST 手写数字，测试通过自动编码器和对抗性自动编码器学习重建恢复效果。

- 原始图像：
![](https://files.mdnice.com/user/15745/320e030a-0a5d-41f7-b6cb-105829197ff2.png)
- 自动编码器重建效果
![](https://files.mdnice.com/user/15745/5655a14f-5b14-4eaf-83d5-75538965b82f.png)
- 对抗性自动编码器重建效果
![](https://files.mdnice.com/user/15745/c07a5d93-df13-43f7-b9c8-e2ec0343fceb.png)
- 有监督对抗性自动编码器重建效果
![](https://files.mdnice.com/user/15745/1ada4ae8-8035-4f5a-9d55-054c1345276f.png)

虽然这里看到，自动编码器和对抗性自动编码器重建出来的能力差不多，有监督对抗性自动编码器基本上重建出来的图像和输入基本对的上。他们的差别有何不同呢，通过之后几章的学习，大家会有体会。

**我们学习自动编码器有什么用？**
重建图像本身自然是没有任何意义的，但是能把图像重建出来，说明模型学到了输入图像集的分布和特征。
- 提取图像特征，特征我们可以拿来做影像组学。
- 异常检测，图像的分布可以拿来做异常检测。
- 图像去噪，其中可以使用有噪声的图像生成清晰的无噪声图像。
![](https://files.mdnice.com/user/15745/790601c9-d30d-430e-af5d-97aabe5b801b.png)
- 语义散列可以使用降维来加快信息检索速度。
- 最近，以对抗方式训练的自动编码器可以用作生成模型（我们稍后会深入探讨）。

具体地， 我们将从以下几部分来介绍：
1. 自动编码器重建 MNIST 手写数字
2. 对抗性自动编码器重建 MNIST 手写数字
3. 监督自动编码器重建 MNIST 手写数字
4. 使用自动编码器对 MNIST 进行分类

本代码主要参考：https://github.com/Naresh1318/Adversarial_Autoencoder

原作者提供的是 tensorflow 版本，这里我提供了  tensorflow （copy Naresh1318）和 pytorch 两种版本

## 如何使用代码
![img.png](img.png)
红框是我在 Naresh1318 的基础上新增的 torch 版本， 黄框是原作者的 tf 版本。

- torch_version
![img_1.png](img_1.png)
  
这里面是 AE, AAE, SAAE 模型训练的代码

- TorchResults
![img_2.png](img_2.png)
所有的实验结果均保存在这个文件夹。我已经上传了三个模型训练好的模型，可以直接拿来测试。 
  Tensorboard的结果我没有上传，数据太大，如果需要，可以联系我。
  
## Train model
假设要训练 AE model。
直接运行 autoencoder_torch.py 文件。但在运行之前，确保以下事项
1. 数据集是否下载
```python
train_data = MNIST(root='./data', train=True, download=False, transform=transform)
    # 如果没有下载，使用download=True, 下载一次后，后面再运行代码无需下载
```
第一次训练模型，请设置 download=True, 下载手写数字。下载后之后训练就可以设置成False, 避免重复下载耽搁时间

2 确认文件夹地址
每个模型里都有函数 `form_results`， 请确认里面设置的地址正确

都确认好了就可以直接 run 了。

## 查看训练过程

训练会生成 Tensorboard 文件，可以打开实时查看 训练进度。