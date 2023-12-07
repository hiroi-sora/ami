import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np


class Img2Vec:
    # ResNet模型的输出大小
    RESNET_OUTPUT_SIZES = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "resnet152": 2048,
    }

    # EfficientNet模型的输出大小
    EFFICIENTNET_OUTPUT_SIZES = {
        "efficientnet_b0": 1280,
        "efficientnet_b1": 1280,
        "efficientnet_b2": 1408,
        "efficientnet_b3": 1536,
        "efficientnet_b4": 1792,
        "efficientnet_b5": 2048,
        "efficientnet_b6": 2304,
        "efficientnet_b7": 2560,
    }

    def __init__(
        self, cuda=False, model="resnet-18", layer="default", layer_output_size=512
    ):
        """Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: 请求的模型的字符串名称
        :param layer: 根据模型不同可以是字符串或整数.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: 请求层的输出大小 Int depicting the output size of the requested layer
        """
        self.device = torch.device("cuda" if cuda else "cpu")
        self.layer_output_size = layer_output_size
        self.model_name = model
        # 获取模型和想要的层
        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)

        self.model = self.model.to(self.device)
        # 设置模型为评估模式，不跟踪梯度和更新参数
        self.model.eval()
        # 定义图像预处理的步骤
        self.scaler = transforms.Resize((224, 224))  # 创建一个大小为(224, 224)的调整器，用于调整图像大小。
        self.normalize = transforms.Normalize(
            # 创建一个归一化器，用于正则化图像。 mean 均值 ， std 标准差
            # 将样本集中图像总体调整为：均值为0、标准偏差为1
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.to_tensor = transforms.ToTensor()  # 用于将图像转为张量

    # 将输入的图像 img 转换为一个特征向量 embedding 。
    def get_vec(self, img, tensor=False):
        """Get vector embedding from PIL image
        :param img: PIL图像或PIL图像列表
        :param tensor: 如果 True, 将返回FloatTensor而不是Numpy数组
        :returns: Numpy ndarray
        返回模型的某一层的输出，这通常被用作图像的特征表示
        """
        if type(img) == list:
            # 将所有图片： 1.缩放为(224, 224)， 2.转换为张量， 3.正则化， 4.放入一个列表
            a = [self.normalize(self.to_tensor(self.scaler(im))) for im in img]
            # 将a中所有张量堆叠起来形成一个新的张量
            images = torch.stack(a).to(self.device)
            # 根据模型名称初始化embedding。第一维为图片数量，第二维为输出层大小
            if self.model_name in ["alexnet", "vgg"]:
                my_embedding = torch.zeros(len(img), self.layer_output_size)
            elif self.model_name == "densenet" or "efficientnet" in self.model_name:
                my_embedding = torch.zeros(len(img), self.layer_output_size, 7, 7)
            else:
                my_embedding = torch.zeros(len(img), self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                # m 被钩子注册的层的模块，i 输入到该层的数据，o 从该层输出的数据
                # 将o的数据复制到my_embedding
                my_embedding.copy_(o.data)

            # 注册一个前向钩子，将copy_data函数注册到self.extraction_layer这一层上。
            # 每次前向传播时，当输入数据通过self.extraction_layer层时，copy_data函数将被自动调用
            h = self.extraction_layer.register_forward_hook(copy_data)
            with torch.no_grad():
                h_x = self.model(images)  # 进行前向传播
            h.remove()  # 移除钩子

            if tensor:
                return my_embedding
            else:  # 转换为 numpy
                if self.model_name in ["alexnet", "vgg"]:
                    return my_embedding.numpy()[:, :]
                elif self.model_name == "densenet" or "efficientnet" in self.model_name:
                    return torch.mean(my_embedding, (2, 3), True).numpy()[:, :, 0, 0]
                else:
                    return my_embedding.numpy()[:, :, 0, 0]
        else:
            image = (
                self.normalize(self.to_tensor(self.scaler(img)))
                .unsqueeze(0)  # 在维度0上添加了一个维度，将形状 原本(C, H, W) 转换为(1, C, H, W)
                .to(self.device)
            )

            if self.model_name in ["alexnet", "vgg"]:
                my_embedding = torch.zeros(1, self.layer_output_size)
            elif self.model_name == "densenet" or "efficientnet" in self.model_name:
                my_embedding = torch.zeros(1, self.layer_output_size, 7, 7)
            else:
                my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            with torch.no_grad():
                h_x = self.model(image)
            h.remove()

            if tensor:
                return my_embedding
            else:
                if self.model_name in ["alexnet", "vgg"]:
                    return my_embedding.numpy()[0, :]
                elif self.model_name == "densenet":
                    return torch.mean(my_embedding, (2, 3), True).numpy()[0, :, 0, 0]
                else:
                    return my_embedding.numpy()[0, :, 0, 0]
        """
        模型的某一层的输出可用作图像的特征表示。在深度学习模型，尤其是卷积神经网络（CNNs），在处理图像时，会通过其多个层次自动学习和提取图像的特征。这些特征从初级到高级具有不同的抽象层次：
        1. **低层特征提取**：在网络的初级阶段，模型倾向于识别基本的图像特征，如边缘、颜色和纹理。这些特征对于图像的基本理解非常重要，但通常不足以完全描述复杂的场景或对象。
        2. **中层特征提取**：在网络的中间层，模型开始合并低层特征来识别更复杂的图案，如形状或特定的纹理组合。这些特征开始捕捉到图像中的某些部分和对象的特性。
        3. **高层特征提取**：在网络的更高层，模型能够识别更加抽象和复杂的概念，如整个对象、场景的组成部分或甚至是场景的整体语义。这些特征通常与图像的具体内容紧密相关，并且能够支持复杂的视觉任务，如图像识别、分类和检测。

        **特征向量的作用**：
        - **数据压缩**：特征向量提供了一种压缩图像信息的方法，使其仅包含对完成特定任务最重要的信息。这样可以减少处理和存储需求。
        - **去除冗余**：原始图像数据通常包含大量的冗余信息。网络层的输出通过提取最有用的特征来减少这种冗余。
        - **提高性能**：在许多视觉任务中，使用深度特征而不是原始像素数据可以显著提高性能。这是因为深度特征更具鉴别力，并且对图像中的变化（如光照、位置、尺度）更加鲁棒。
        - **迁移学习**：预训练的网络层的输出可以用于新的任务，这是迁移学习的基础。即使是在不同的、但相关的任务上训练的模型，其提取的特征也可能对新任务非常有用。

        因此，特征向量是图像内容的一个高度抽象和信息密集的表示，它可以用于各种下游任务，如图像分类、检索或识别。通过使用预训练的网络层的输出，可以利用这些模型在大型数据集上学到的丰富视觉知识，而不必从头开始训练模型。
        """

    # 获取模型和对应层
    def _get_model_and_layer(self, model_name, layer):
        """Internal method for getting layer from model
        :param model_name: 模型名称，如'resnet-18'
        :param layer: 对于resnet-18是字符串，对于alexnet是整数。可以填 default
        :returns: 返回俩元素： pytorch模型, 选择的层
        """

        if model_name.startswith("resnet") and not model_name.startswith("resnet-"):
            # 从torchvision.models中获取指定名称的模型，并加载预训练权重。
            model = getattr(models, model_name)(pretrained=True)
            # 默认选择模型的"avgpool"层作为目标层。
            if layer == "default":
                layer = model._modules.get("avgpool")
                # layer_output_size 默认是512，这里更改为 RESNET_OUTPUT_SIZES 中的值， 512 or 2048
                self.layer_output_size = self.RESNET_OUTPUT_SIZES[model_name]
            else:
                layer = model._modules.get(layer)
            return model, layer

        elif model_name == "resnet-18":
            model = models.resnet18(pretrained=True)
            if layer == "default":
                layer = model._modules.get("avgpool")
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == "alexnet":
            model = models.alexnet(pretrained=True)
            if layer == "default":
                layer = model.classifier[-2]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        elif model_name == "vgg":
            # VGG-11
            model = models.vgg11_bn(pretrained=True)
            if layer == "default":
                layer = model.classifier[-2]
                self.layer_output_size = model.classifier[
                    -1
                ].in_features  # should be 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        elif model_name == "densenet":
            # Densenet-121
            model = models.densenet121(pretrained=True)
            if layer == "default":
                layer = model.features[-1]
                self.layer_output_size = model.classifier.in_features  # should be 1024
            else:
                raise KeyError("Un support %s for layer parameters" % model_name)

            return model, layer

        elif "efficientnet" in model_name:
            # efficientnet-b0 ~ efficientnet-b7
            if model_name == "efficientnet_b0":
                model = models.efficientnet_b0(pretrained=True)
            elif model_name == "efficientnet_b1":
                model = models.efficientnet_b1(pretrained=True)
            elif model_name == "efficientnet_b2":
                model = models.efficientnet_b2(pretrained=True)
            elif model_name == "efficientnet_b3":
                model = models.efficientnet_b3(pretrained=True)
            elif model_name == "efficientnet_b4":
                model = models.efficientnet_b4(pretrained=True)
            elif model_name == "efficientnet_b5":
                model = models.efficientnet_b5(pretrained=True)
            elif model_name == "efficientnet_b6":
                model = models.efficientnet_b6(pretrained=True)
            elif model_name == "efficientnet_b7":
                model = models.efficientnet_b7(pretrained=True)
            else:
                raise KeyError("Un support %s." % model_name)

            if layer == "default":
                layer = model.features
                self.layer_output_size = self.EFFICIENTNET_OUTPUT_SIZES[model_name]
            else:
                raise KeyError("Un support %s for layer parameters" % model_name)

            return model, layer

        else:
            raise KeyError("Model %s was not found" % model_name)
