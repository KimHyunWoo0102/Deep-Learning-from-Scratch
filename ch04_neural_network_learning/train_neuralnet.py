import sys, os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(parent_dir)

sys.path.append(parent_dir)

import numpy as np
import matplotlib.pylab as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# hyper parameters
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    print(f"{i}번째 실행중...")
    # 1. 배치 설정
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch)

    for key in grad:
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1 epoch 당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc |" + str(train_acc) + ", " + str(test_acc))

print("Done!")

x = np.arange(len(train_loss_list))  # x축 생성 (0부터 반복 횟수만큼)
plt.plot(x, train_loss_list, label="loss")  # y축에 손실 값 할당하여 그래프 그리기
plt.xlabel("iteration")
plt.ylabel("loss")
plt.legend()  # 범례 표시 (선택사항)
plt.show()

markers = {"train": "o", "test": "s"}
x = np.arange(len(train_acc_list))

plt.figure(figsize=(10, 6))  # 그래프 크기 조절 (선택사항)
plt.plot(x, train_acc_list, label="train acc")
plt.plot(x, test_acc_list, label="test acc", linestyle="--")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)  # y축 범위를 0~1 사이로 고정
plt.legend(loc="lower right")
plt.show()
