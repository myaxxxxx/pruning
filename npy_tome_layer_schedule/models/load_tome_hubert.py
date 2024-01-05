




# import torch
# from torchvision.models import resnet18

# # 创建一个与加载的模型结构不匹配的模型
# class ModifiedResNet(torch.nn.Module):
#     def __init__(self):
#         super(ModifiedResNet, self).__init__()
#         self.features = torch.nn.Sequential(
#             # 修改了原始模型中的一部分分类的名字
#             torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
#             torch.nn.BatchNorm2d(64),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#         )
#         self.fc = torch.nn.Linear(512, 10)  # 修改了原始模型中的另一部分类的名字

# # 创建一个与加载的模型结构一致的空模型
# model = resnet18()

# # 加载模型参数，并处理不匹配的部分
# pretrained_dict = torch.load('path_to_pretrained_model.pth', map_location=torch.device('cpu'))
# model_dict = model.state_dict()

# # 自定义映射函数，处理不匹配的部分
# mapping = {
#     'features.0.weight': 'conv1.weight',  # 将修改的卷积层的参数映射到预训练模型中对应的参数
#     'fc.weight': 'fc.weight',  # 将修改的全连接层的参数映射到预训练模型中对应的参数
#     'fc.bias': 'fc.bias',  # 将修改的全连接层的参数映射到预训练模型中对应的参数
# }

# # 更新模型参数
# for k, v in mapping.items():
#     if k in pretrained_dict and v in model_dict:
#         model_dict[v] = pretrained_dict[k]

# model.load_state_dict(model_dict)