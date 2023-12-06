我们为三个数据集提供了三套Python文件：

- `celeba-eval-noldp.py`和`celeba-exp-noldp.py`：CelebA数据集
- `cifar-eval-noldp.py`和`cifar-exp-noldp.py`：CIFAR-10数据集
- `imgnet-eval-noldp.py`和`img-exp-noldp.py`：ImageNet数据集

`*-exp*.py`文件用于训练恶意参数，而`*-eval*.py`文件用于评估参数并计算攻击成功率。下面我们以在CIFAR-10数据集上运行攻击为例，其他数据集遵循相同的语法。

要运行攻击，请执行以下命令：

```bash
$ python cifar-exp-noldp.py -r NUMNEURONS -o OUTPUT_PATH
```
- `NUMNEURONS`是第一层中的神经元数量（参数$r$）
- `OUTPUT_PATH`是输出目录的路径

示例：
```bash
$ python cifar-exp-noldp.py -r 2000 -o res/
```

要获取攻击成功率，请执行以下命令：
```bash
$ python cifar-eval-noldp.py -r NUMNEURONS -o OUTPUT_PATH
```
- `OUTPUT_PATH`是攻击输出文件的路径

示例：
```bash
$ python cifar-eval-noldp.py -r 2000 -o res/
```
这将输出TPR、TNR和Adv。

要获取完整的用法信息，请运行`python cifar-exp-noldp.py -h`和`python cifar-eval-noldp.py -h`。