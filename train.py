import torch as t
import torchvision as tv
import os
import time
import numpy as np
from tqdm import tqdm


# 一些参数配置
class DefaultConfigs(object):
    data_dir = "./imageData/"
    data_list = ["train", "test"]
    lr = 0.001
    epochs = 51
    num_classes = 10
    image_size = 32
    batch_size = 40
    channels = 3
    use_gpu = t.cuda.is_available()


config = DefaultConfigs()
config.use_gpu = False

normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform = {

    config.data_list[0]: tv.transforms.Compose(
        [tv.transforms.Resize([config.image_size, config.image_size]),
         tv.transforms.CenterCrop([config.image_size,
                                   config.image_size]),
         tv.transforms.ToTensor(), normalize]),
    # test 测试数据
    config.data_list[1]: tv.transforms.Compose([
        tv.transforms.Resize([config.image_size, config.image_size]),
        tv.transforms.ToTensor(),
        normalize
    ])
}

# 数据集
datasets = {
    x: tv.datasets.ImageFolder(root=os.path.join(config.data_dir, x), transform=transform[x])
    for x in config.data_list
}

# 数据加载器
dataloader = {
    x: t.utils.data.DataLoader(dataset=datasets[x],
                               batch_size=config.batch_size,
                               shuffle=True)
    for x in config.data_list
}


# 构建网络模型 resnet18
def get_model(num_classes):
    model = tv.models.resnet18(pretrained=True)

    # for parma in model.parameters():
    #  parma.requires_grad = False

    model.fc = t.nn.Sequential(t.nn.Dropout(p=0.3), t.nn.Linear(512, num_classes))

    return model


# 训练模型
def train(epochs):
    model = get_model(config.num_classes)

    loss_f = t.nn.CrossEntropyLoss()

    # GPU
    if config.use_gpu:
        model = model.cuda()
        loss_f = loss_f.cuda()

    opt = t.optim.Adam(model.fc.parameters(), lr=config.lr)
    # 时间
    time_start = time.time()

    for epoch in range(epochs):
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []
        model.train(True)  # 将模块设置为训练模式
        print("Epoch {}/{}".format(epoch + 1, epochs))
        for batch, datas in tqdm(enumerate(iter(dataloader["train"]))):
            x, y = datas
            # 开启GPU 加速
            if config.use_gpu:
                x, y = x.cuda(), y.cuda()

            y_ = model(x)

            # print(x.shape, y.shape, y_.shape)
            _, pre_y_ = t.max(y_, 1)
            pre_y = y
            # print(y_.shape)
            loss = loss_f(y_, pre_y)
            # print(y_.shape)
            acc = t.sum(pre_y_ == pre_y)

            loss.backward()
            opt.step()
            opt.zero_grad()
            if config.use_gpu:
                loss = loss.cpu()
                acc = acc.cpu()
            train_loss.append(loss.data)
            train_acc.append(acc)
        time_end = time.time()
        print("正式 批次 {}, Train 损失:{:.4f}, Train 准确率:{:.4f}, 训练时间: {}".format(batch + 1,
                                                                             np.mean(train_loss) / config.batch_size,
                                                                             np.mean(train_acc) / config.batch_size,
                                                                             (time_end - time_start)))

        model.train(False)  # 关闭训练模式
        for batch, datas in tqdm(enumerate(iter(dataloader["test"]))):
            x, y = datas
            if config.use_gpu:
                x, y = x.cuda(), y.cuda()
            y_ = model(x)
            # print(x.shape,y.shape,y_.shape)
            _, pre_y_ = t.max(y_, 1)
            pre_y = y
            # print(y_.shape)
            loss = loss_f(y_, pre_y)
            acc = t.sum(pre_y_ == pre_y)

            if config.use_gpu:
                loss = loss.cpu()
                acc = acc.cpu()

            test_loss.append(loss.data)
            test_acc.append(acc)

            print("测试 批次 {}, 损失:{:.4f}, 准确率:{:.4f}".format(batch + 1, np.mean(test_loss) / config.batch_size,
                                                           np.mean(test_acc) / config.batch_size))

    t.save(model, 'model/' + str(epoch + 1) + "_ttmodel.pkl")  # 保存整个神经网络的结构和模型参数

    t.save(model.state_dict(), 'model/' + str(epoch + 1) + "_ttmodel_params.pkl")  # 只保存神经网络的模型参数
    print('训练结束')

#开始训练
if __name__ == "__main__":
    train(config.epochs)
