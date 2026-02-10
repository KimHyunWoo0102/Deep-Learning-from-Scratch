## 📑 [Chapter 03]: 신경망

### 1. 퍼셉트론의 한계와 신경망의 탄생

- **퍼셉트론의 한계:** 단층 퍼셉트론은 '직선'으로만 영역을 나눌 수 있어 XOR와 같은 비선형 문제를 해결하지 못합니다.
- **신경망의 등장:** 이를 해결하기 위해 퍼셉트론을 여러 층으로 쌓아 복잡한 연산을 가능하게 한 것이 바로 **신경망**입니다.

------

### 2. 활성화 함수: 신호를 전달하는 "문지기"

신경망은 각 층에서 계산된 신호가 다음 층으로 전달될 때, 그 값을 그대로 보내지 않고 **활성화 함수(Activation Function)**를 통과시킵니다.

![img](https://blog.kakaocdn.net/dna/AfSHP/btrPQ4rbpND/AAAAAAAAAAAAAAAAAAAAANOwYzwxBHXUjhyW-v6Cz97JBhyzJk8s7gxg608PaJBj/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=rrHvcou3UfW2pqoPsduDLmZZVPI%3D)

#### ① 계단 함수(Step Function)의 문제점

초기 신경망에서는 계단 함수를 사용했지만, 두 가지 치명적인 단점이 있었습니다.

1. **정보 손실:** 임계값(예: 0.5) 직전인 0.49와 아주 작은 0.01을 똑같이 '0'으로 취급하여 신호의 미세한 변화를 무시합니다.
2. **미분 불가:** 딥러닝의 핵심 학습 알고리즘인 **역전파(Backpropagation)**는 기울기(미분값)를 사용합니다. 하지만 계단 함수는 0을 제외한 모든 구간에서 기울기가 0이라 학습이 불가능합니다.

------

### 3. 시그모이드(Sigmoid): 부드러운 전이의 시작

![img](https://blog.kakaocdn.net/dna/sib2q/btsDJkfaBMy/AAAAAAAAAAAAAAAAAAAAAJwQd20TZhIxLx58GiUJdYxCLH_4xkSt66vEea-TnEkx/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=IJIcAxzqa0Vqo3wLhMizySvy5iQ%3D)

계단 함수의 단점을 보완하기 위해 '부드러운 곡선' 형태인 시그모이드가 도입되었습니다.

- **장점:** 모든 지점에서 미분이 가능하며, 신호의 강약을 연속적으로 전달할 수 있습니다.
- **단점 (한계):**
  1. **지그재그 현상:** 출력이 항상 양수($0 \sim 1$)이기 때문에, 학습 시 가중치가 효율적인 직선 경로가 아닌 '지그재그'로 비효율적으로 업데이트됩니다.
  2. **기울기 소실(Gradient Vanishing):** 입력값이 너무 크거나 작아지면 그래프가 평평해지며 기울기가 0에 수렴합니다. 이로 인해 심층 신경망에서 학습 신호가 끝까지 전달되지 못합니다.

------

### 4. 현대의 표준, ReLU (Rectified Linear Unit)

![img](https://blog.kakaocdn.net/dna/vgJna/btqQzRGmwcO/AAAAAAAAAAAAAAAAAAAAAF2TqoGwCrvDWO4WUKsa9LPeybwgzWbow476RGHGR8n8/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1772290799&allow_ip=&allow_referer=&signature=kJEBoFhHnhS0LiU6LsdAyP6cTKw%3D)

시그모이드의 한계를 극복하기 위해 등장한 것이 현재 가장 널리 쓰이는 **ReLU**입니다.

- **특징:** 입력이 0을 넘으면 그 값을 그대로 전달하고, 0 이하면 차단합니다.
- **장점:** * 0보다 큰 구간에서 기울기가 죽지 않아 학습 속도가 비약적으로 빠릅니다.
  - 연산이 매우 간단하여 하드웨어 리소스를 적게 소모합니다.

------

## 5. 출력층의 설계: 결과의 해석

신경망은 풀고자 하는 문제의 성격에 따라 출력층의 활성화 함수를 다르게 선택합니다.

- **회귀(Regression):** 입력 데이터에서 연속적인 수치를 예측할 때 사용하며, **항등 함수(Identity Function)**를 써서 계산된 값을 그대로 내보냅니다.
- **분류(Classification):** 데이터를 정해진 범주 중 하나로 나눌 때 사용하며, 주로 **소프트맥스 함수(Softmax Function)**를 사용하여 결과를 확률로 변환합니다.

### ① 소프트맥스(Softmax)의 수식과 의미

소프트맥스는 출력층의 모든 입력($a$)을 받아 다음과 같이 계산합니다.

$$y_k = \frac{\exp(a_k)}{\sum_{i=1}^{n} \exp(a_i)}$$

- **확률적 해석:** 모든 출력 값($y$)은 $0$과 $1.0$ 사이의 실수가 되며, **출력의 총합은 항상 $1$**이 됩니다. 덕분에 우리는 "이 사진이 숫자 3일 확률은 90%이다"라고 통계적으로 해석할 수 있습니다.
- **대소 관계 유지:** 지수 함수($y = \exp(x)$)는 단조 증가 함수이기 때문에, 입력값들 사이의 크기 순서가 출력에서도 그대로 유지됩니다. 즉, 가장 큰 점수를 받은 녀석이 가장 높은 확률을 갖게 됩니다.

### ② 오버플로우(Overflow) 대책

컴퓨터가 지수 함수($\exp$)를 계산할 때, 입력값이 조금만 커져도(예: $1000$ 이상) 결과가 무한대(`inf`)로 치솟아 계산이 불가능해지는 오버플로우 문제가 발생합니다.

- **해결책 (Max 차감법):** 소프트맥스의 지수 법칙 성질을 이용하여, 입력값 중 **최댓값($C$)을 모든 입력에서 빼준 뒤** 계산해도 결과는 변하지 않습니다.

$$y_k = \frac{\exp(a_k - C)}{\sum \exp(a_i - C)} \quad (\text{where } C = \max(a))$$


