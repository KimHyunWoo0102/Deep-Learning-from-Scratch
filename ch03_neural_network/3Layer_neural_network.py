import numpy as np
import matplotlib.pylab as plt
import activate_functions.sigmoid_function as sig


def init_network():
    """
    3단계 신경망의 기본데이터들을 설정하여 반환
    """
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b2"] = np.array([0.1, 0.2])
    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])

    return network


def identity_function(x):
    return x


def softmax(a):
    # 숫자가 커지면 오버플로우가 발생하니 지수 크기 조절
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def forward(network, x):
    """
    입력신호를 연산하여 결과 반환
    """
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sig.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sig.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


if __name__ == "__main__":
    network = init_network()
    x = np.array([0.3, 0.5])
    y = forward(network, x)
    print(y)
