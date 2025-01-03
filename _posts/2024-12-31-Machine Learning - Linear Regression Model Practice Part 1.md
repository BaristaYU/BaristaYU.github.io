---
classes: wide
title: "[INCOMPLETE] (ENG) Machine Learning - Linear Regression Model Practice Part I"
categories:
  - 머신러닝
tags:
  - AI
  - Andrew Ng
  - Supervised learning
  - Linear Regression model
  - numpy
  - Python
---

Hello, my name is Yongkyun Yu.
I'm Korean, and I'm not good at English, so I'm trying to improve my English skills by sharing some data science knowledge in English without translator.
You'll learn some knowledge, and I'll learn some English skills. Quite cool, isn't it? Thank you for your understanding, I'll try my best.

This post was written after attending Professor Andrew Ng's Supervised Machine Learning: Regression and Classification Course at Coursera.<br>
It'll be nice to understand this post if you know how to use numpy module in python and some mathmatical knowledge, but it's okay if not.

**본 글의 코드는 상기한 강의 내용을 기반으로 변형하였음을 밝힘.**

---

본 글에서는 파이썬의 numpy 모듈을 활용하여 일변량 선형회귀 모델을 만들고, 간단한 matplotlib을 통해 시각화하는 과정을 배워볼 것이다.
**파이썬의 기본 문법을 사용할 줄 안다는 가정하에 작성된 글이므로**, 파이썬에 대해 잘 모른다면 배우고 오는 것이 좋다.

지난 글에서 우리는 다음과 같은 것들을 배웠다.

- 일변량 선형회귀 모델
- 일변량 선형회귀 모델의 비용 함수
- 일변량 선형회귀 모델의 경사하강법 알고리즘
  - $w$,$b$에 대한 접선의 기울기

그리고 위는 아래의 수식으로 표현할 수 있다.

- $f_{w,b}(x^{(i)})=\hat{y}^{(i)}=wx^{(i)}+b$
- $ J(w,b)=\frac{1}{2m}\displaystyle\sum\_{i=1}^m(\hat{y}^{(i)}-y^{(i)})^2 $
- $\text{repeat until convergence}$:{<br>
  $w=w-a{\frac{d}{dw}}J(w,b)$<br>
  $b=b-a{\frac{d}{db}}J(w,b)$<br>
  }<br>
  - $\frac{d}{dw}J(w,b)=\frac{1}{m} \displaystyle\sum^m_{i=1}(wx^{(i)}+b-y^{(i)})x^{(i)}$
  - $\frac{d}{db}J(w,b)=\frac{1}{m} \displaystyle\sum^m_{i=1}(wx^{(i)}+b-y^{(i)})$

파이썬을 비롯한 프로그래밍 언어의 장점은, 수식이 있다면 그것을 구현하는 것이 어렵지 않다는 것이다. 하나씩 차근차근 구현해보기 앞서, numpy 모듈과 matplotlib 모듈에 대해 아주 간단히 알아보고 넘어가자.

- numpy 소개: 배열을 필두로 한 벡터, `행렬 연산`에 쓰이는 모듈이다.
- matplotlib: 데이터를 `시각화`하는 데 쓰이는 모듈이다.

본 글에서는 다음과 같은 순서로 코드를 구현할 것이다:

1. 모듈과 데이터셋 선언
2. 비용 함수 선언
3. 기울기 계산 함수 선언
4. 경사하강법 함수 선언
5. 일변량 선형회귀 모델 선언
6. 일변량 선형회귀 모델 실행
7. 시각화(Plotting)

<br>
이 순서는 우선순위와 같다. <br>
경사하강법에서 특정 반복횟수때의 비용 함수를 출력하기 위해 비용 함수와, 기울기를 사용하기 위해 기울기 계산 함수가 필요하다. 일변량 선형회귀 모델에서 경사하강법으로 최적화된 $w,b$값이 필요하다. <br>
파이썬은 `인터프리터` 언어로, 위에서부터 아래로 코드를 읽기 때문에 필요한 코드를 위에 배치하는 것이 좋다.

---

### 모듈과 데이터셋

```python
import math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
```

사용할 모듈을 `import`하고, 데이터를 선언하는 부분이다. <br>
x-y 데이터쌍은 2개(1.0, 300.0 / 2.0, 500.0)만 예제로 사용하겠다.
`x`는 집의 `평방피트`, `y`는 집의 `달러`이다.<br>
`math` 모듈은, 경사하강법을 진행할 때 특정 횟수마다 출력되도록 하기 위해 사용되었다. `math` 모듈 없이도 구현이 가능하므로, 원한다면 선언하지 않아도 된다.
<br> `plt` 메서드 두 줄은 한글 제목을 사용하기 위한 구문이다.

---

### 비용 함수

```python
def compute_cost(x, y, w, b):

    m = x.shape[0]
    cost = 0

    for i in range(m): # sum 0 to m
        f_wb = w * x[i] + b #f_w,b(x)
        cost = cost + (f_wb - y[i])**2 # J(w,b)
    total_cost = 1 / (2 * m) * cost # 1/2m

    return total_cost
```

- $f_{w,b}(x^{(i)})=\hat{y}^{(i)}=wx^{(i)}+b$
- $ J(w,b)=\frac{1}{2m}\displaystyle \sum^m\_{i=1}(\hat{y}^{(i)}-y^{(i)})^2 $

`x.shape[0]`는 `numpy` 모듈의 메서드로, `행, 열의 길이`를 표현한다. [0]은 첫 번째 행, [1]은 두 번째 행...을 의미한다. 본 글에서는 1차원 배열, 벡터를 사용할 것이므로 [0]은 곧 `벡터의 길이`를 의미한다. 즉, `데이터셋의 개수`와 동일한 의미를 가진다. <br>
$i$가 1부터 m까지 있는 것은 파이썬의 `for`문으로 표현 가능하다. 이 때, `for`문은 대개 `0부터 시작`하므로, 사실은 $\displaystyle\sum_{i=1}^m$가 아닌 $\displaystyle\sum_{i=0}^{m-1}$이 맞다. 표현의 차이일 뿐, 값은 같다. 1부터 m까지를 구현하고 싶다면, `range(m)` 대신 `range(1,m+1)`으로 작성하면 된다. <br>
`MSE` 방식으로 오차를 구하여 모두 더한 뒤, $\frac{1}{2m}$으로 나누어 비용을 계산한다.

---

### 기울기 계산

```python
def compute_gradient(x, y, w, b):

    m = x.shape[0]
    dj_dw = 0 #w에 대한 접선의 기울기
    dj_db = 0 #b에 대한 접선의 기울기

    for i in range(m): #sum 0 to m
        f_wb = w * x[i] + b #f_w,b(x)
        dj_dw_i = (f_wb - y[i]) * x[i] #dj_dw
        dj_db_i = f_wb - y[i] #dj_db
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m #1/m
    dj_db = dj_db / m #1/m

    return dj_dw, dj_db
```

- $\frac{d}{dw}J(w,b)=\frac{1}{m} \displaystyle\sum^m_{i=1}(wx^{(i)}+b-y^{(i)})x^{(i)}$
- $\frac{d}{db}J(w,b)=\frac{1}{m} \displaystyle\sum^m_{i=1}(wx^{(i)}+b-y^{(i)})$

자세한 설명은 비용 함수와 같다.

---

### 경사하강법

```python
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):

    J_history = [cost_function(x, y, w_in, b_in)] #비용 내역을 저장할 list, 초기값 저장
    p_history = [[w_in, b_in]] #w,b 내역을 저장할 list, 초기값 저장
    b = b_in #입력받은 b
    w = w_in #입력받은 w

    for i in range(num_iters): #목표 횟수만큼 반복
        dj_dw, dj_db = gradient_function(x, y, w, b) # 기울기 계산하여 저장

        w = w - alpha * dj_dw # w 경사하강
        b = b - alpha * dj_db # b 경사하강

        J_history.append( cost_function(x, y, w, b)) # 현재 비용 구하여 list에 저장
        p_history.append([w,b]) # 현재 w,b list에 저장

        if i% math.ceil(num_iters/10) == 0:
        # 만약 반복 횟수가 목표 횟수를 10으로 나눈 것을 올림한 값으로 나누었을 때 나머지가 0이면
        # = 목표 횟수가 500000이면, 50000, 100000, ..., 450000번째의 값들 출력
            print(f"Iteration {i:4}: Cost {J_history[-1]:.3e} ",
                  f"dj_dw: {dj_dw:.3e}, dj_db: {dj_db:.3e}  ",
                  f"w: {w:.3e}, b:{b:.3e}")

    return w, b, J_history, p_history
```

- $\text{repeat until convergence}$:{<br>
  $w=w-a{\frac{d}{dw}}J(w,b)$<br>
  $b=b-a{\frac{d}{db}}J(w,b)$<br>
  }<br>

경사하강법 함수는 초기 $w,b$값, 학습률(learning rate) $a$, 목표 횟수, 비용 함수와 기울기 계산 함수를
입력받는다. 여기서 중요한 점은, `기울기를 구한 다음 w,b에 반영`하는 것이다.
$w$에 먼저 반영한 뒤 기울기를 다시 구하면, 값이 동일하게(simultaneously) 반영되지 않는다.<br>

잘못된 예시:

```python
for i in range(num_iters): #목표 횟수만큼 반복
        w = w - alpha * dj_dw # w 경사하강
        dj_dw, dj_db = gradient_function(x, y, w , b) # 보정된 w로 기울기 계산
        b = b - alpha * dj_db # b 경사하강
```

`math` 모듈은 여기서 사용된다. 반복횟수가 499999와 같으면, 49999의 배수일 때 출력하게 되므로 보기 쉬운 값으로 올림 처리하는 것이다.

자릿수를 표현할 때 `e`는 `10의 n승`을 의미한다. e+02는 10의 2승(100), e-02는 10의 -2승(0.01)이다. 위 print문에서의 .3e는 세번째 자리까지 표현 후 e로 표기한다는 의미이다. 만약 print문이 이해가 되지 않는다면, 파이썬의 f-string 문법에 대해 보고 오자.

---

### 일변량 선형회귀

```python
def linear_regression(x):
  global w_final,b_final
  return w_final*x + b_final
```

`전역 변수(global)`로 최종 $w,b$를 받아 예측값($\hat{y}$)를 반환한다.

---

### 변수 설정

```python
#하이퍼파라미터 4개: w 초기값, b 초기값, 반복횟수, 학습률 alpha
w_init = 0
b_init = 0
iterations = 10000
alpha = 0.001 #0.6: 발산 / 0.5: 수렴

x_new = np.array([150, 225, 275])

w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, alpha, iterations, compute_cost, compute_gradient)
print(f"(w,b,J) found by gradient descent: ({w_final:.3e},{b_final:.3e},{J_hist[-1]:.3e})")

print(f"2 sqft house prediction {linear_regression(2):.2f} dollars")
print(f"3 sqft house prediction {linear_regression(3):.2f} dollars")
print(f"4 sqft house prediction {linear_regression(4):.2f} dollars")
```

하이퍼파라미터 4개를 설정한 뒤, 예측값을 찾고싶은 데이터셋 x_new를 선언하여, 경사하강법을 실행 후 최적의 w,b값으로 일변량 선형회귀를 진행한다. 단, $w,b$의 초기값만 하이퍼파라미터일 뿐, 경사하강법 이후의 값은 하이퍼파라미터가 아니다. <br>
0 / 0 / 10000 / 0.001의 결과값:

```python
(w,b,J) found by gradient descent: (1.949e+02,1.082e+02,3.419e+00)
2 sqft house prediction 498.06 Thousand dollars
3 sqft house prediction 692.97 Thousand dollars
4 sqft house prediction 887.88 Thousand dollars
```

초기 데이터셋에서 $x$가 2일때, $y$는 500의 값을 가진다. 그러나 위의 경우 2 정도의 오차가 발생했다. 이에 대한 자세한 설명은 0.6, 0.5의 주석을 포함하여 아래 그래프 해석 파트에서 자세히 다루겠다.

---

### 시각화(Plotting)

```python
w_hist = list(range(len(p_hist))) #p_hist만큼 길이를 가지는 w_hist 선언
#enumerate(): 인덱스와 원소를 순차적으로 반환
for index, weight in enumerate(p_hist):
    w_hist[index] = weight[0] #w_hist에 weight만 저장 // weight[0]: w, weight[1]: b

#constrained_layout: 서브플롯간 간격을 최소의 공백을 가지도록 조절
fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(12,8))

axs[0,0].set_title('Univariate Linear Regression')
axs[0,0].plot(x_train,linear_regression(x_train))
axs[0,0].set_xlabel('x_train');  axs[0,0].set_ylabel('y-hat')

axs[0,1].set_title('Cost vs. weight')
axs[0,1].plot(w_hist,J_hist)
axs[0,1].set_xlabel('weight');  axs[0,1].set_ylabel('Cost')

axs[1,0].plot(J_hist[:1000])
axs[1,0].set_title("Cost vs. iteration(start)")
axs[1,0].set_xlabel('iteration step');  axs[1,0].set_ylabel('Cost')

axs[1,1].plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
axs[1,1].set_title("Cost vs. iteration (end)")
axs[1,1].set_xlabel('iteration step');  axs[1,1].set_ylabel('Cost')
```

`matplotlib` 모듈을 통해 선형회귀 모델의 그래프와, 비용-반복 횟수의 그래프를 1000번 이전과 이후로 나누어 동시에 그래프로 표현하였다. `matplotlib` 모듈의 사용법에 대해서는 다소 지엽적인 부분이 있어 생략하겠다. 필요할 때마다 찾아보는 편이 이롭다.

---

### 그래프 해석

- $w,b$ 초기값은 0으로 고정하고, `반복 횟수`와 `학습률`만 조정

### 10000 / 0.001 (작은 학습률)

#### 그래프

![학습률작음](https://github.com/user-attachments/assets/090b79c9-17a8-45f6-b342-c5590a2e5430)

#### w, b, J, 예측값

```python
(w,b,J) found by gradient descent: (1.949e+02,1.082e+02,3.419e+00)
2 sqft house prediction 498.06 dollars
3 sqft house prediction 692.97 dollars
4 sqft house prediction 887.88 dollars
```

학습률이 작은 경우 10000번의 반복 횟수에도 불구하고 2 정도의 오차가 발생했다. `학습률이 너무 작으면`, 하강하는 정도가 줄어들어 `많은 반복횟수`를 필요로 하여 `메모리에 부담`을 줄 수 있다. 비용-반복 횟수 그래프를 보면, 1000번까지 쭉 내려가다 2000번부터 완만한 경사를 가지고 천천히 내려가는 것을 볼 수 있다.

### 10000 / 0.6 (큰 학습률)

#### 그래프

![발산](https://github.com/user-attachments/assets/f4805bfa-4a56-4982-ac90-7eca90f328f9)

#### w, b, J, 예측값

(w,b,J) found by gradient descent: (-7.326e+239,-4.528e+239,inf)

```python
(w,b,J) found by gradient descent: (-7.326e+239,-4.528e+239,inf)
2 sqft house prediction -1918068600556804811200723259973098665456560009173167338877252848008612238043129452375292671012179411849162821617707229680426965454011496721038020403842579041720952013380160792536622156811754378383734922943215889674338479154282754576828334080.00 dollars
3 sqft house prediction -2650705613215558925091061588994762698453935920686196278160144824435672078927399638551074532852559053344163014178557029223350210520520847409742440217121978422960147912158862415628268349458604126709345673737596640043743404720756083657524903936.00 dollars
4 sqft house prediction -3383342625874313038981399918016426731451311832199225217443036800862731919811669824726856394692938694839163206739406828766273455587030198098446860030401377804199343810937564038719914542105453875034956424531977390413148330287229412738221473792.00 dollars
```

학습률이 너무 큰 경우 뭔가 단단히 잘못된 그래프와 값을 마주할 수 있다. 곡선 형태의 비용 함수에서 발산하는 방향으로 값이 증가하고 있다. 비용함수의 시작점은 weight가 0일 때임을 기억하자.

### 300 / 0.5 (최소 반복 횟수와 최대 학습률)

#### 그래프

![최저](https://github.com/user-attachments/assets/d3067fba-35e2-44cf-b191-4d0c620f23e2)

#### w, b, J, 예측값

```python
(w,b,J) found by gradient descent: (2.000e+02,1.000e+02,4.424e-09)
2 sqft house prediction 500.00 dollars
3 sqft house prediction 700.00 dollars
4 sqft house prediction 900.00 dollars
```

비용이 0으로 수렴하고 있으나, 경사를 따라 내려가는 것이 아닌 건너편으로 뛰어가며 내려가고 있다. 데이터셋이 더 많아지거나 하면 불안정해질 수 있으므로, 여유있게 설정하는 것이 좋다. (마지막 그래프는, 1000번째 이상 반복 횟수의 그래프므로 표시되지 않는다.)

### 6000 / 0.02 (적절한 반복 횟수와 학습률)

#### 그래프

![최적](https://github.com/user-attachments/assets/0765a632-08e2-4c22-b7f1-92dd1c1e2a2f)

#### w, b, J, 예측값

```python
(w,b,J) found by gradient descent: (2.000e+02,1.000e+02,3.618e-07)
2 sqft house prediction 500.00 dollars
3 sqft house prediction 700.00 dollars
4 sqft house prediction 900.00 dollars
```

2 평방피트의 값인 500달러와 0달러의 오차를 보여주며 완벽하게 들어맞는 모습이다. 이로써 이 선형회귀 모델은 $y=200x+100$의 형태를 가진다는 것을 알아내었다.

---

### 정리

그래프에서 보았듯, `비용`은 `반복횟수`가 **초반일 때 빠르게 내려가며**, **이후 천천히 0에 근접**하기 시작한다. 또한, 아래와 같은 특징이 있다.

1. **학습률이 너무 크면**: `비용`이 `발산`한다.
2. **학습률이 너무 작으면**: 많은 `반복횟수`가 필요하다.
3. **반복횟수가 너무 많으면**: `메모리`와 `시간`에 부담이 된다.
4. **반복횟수가 너무 적으면**: 큰 `학습률`이 필요하다.

따라서, 상황과 데이터셋에 맞는 **적절한** `학습률`과 `반복횟수`를 구하는 것이 중요하다. 위에서 그랬듯, 여러 번 조절해가며 값을 도출해내면 된다.

<br> 아래의 코드를 참고하여 직접 실행해보는 것도 좋겠다. 다음 글에서는 일변량이 아닌 `다중 선형회귀 모델`에 대해 배워볼 것이다.

---

### 전체 코드

```python
import math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

def compute_cost(x, y, w, b):

    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost

def compute_gradient(x, y, w, b):

    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):

    J_history = [cost_function(x, y, w_in, b_in)]
    p_history = [[w_in, b_in]]
    b = b_in
    w = w_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        J_history.append( cost_function(x, y, w, b))
        p_history.append([w,b])

        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:.3e} ",
                  f"dj_dw: {dj_dw:.3e}, dj_db: {dj_db:.3e}  ",
                  f"w: {w:.3e}, b:{b:.3e}")

    return w, b, J_history, p_history

def linear_regression(x):
    global w_final,b_final
    return w_final*x + b_final

w_init = 0
b_init = 0

iterations = 6000
alpha = 0.02

x_new = np.array([150, 225, 275])

w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, alpha, iterations, compute_cost, compute_gradient)
print(f"(w,b,J) found by gradient descent: ({w_final:.3e},{b_final:.3e},{J_hist[-1]:.3e})")

print(f"2 sqft house prediction {linear_regression(2):.2f} dollars")
print(f"3 sqft house prediction {linear_regression(3):.2f} dollars")
print(f"4 sqft house prediction {linear_regression(4):.2f} dollars")

w_hist = list(range(len(p_hist)))
for index, weight in enumerate(p_hist):
    w_hist[index] = weight[0]

fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(12,8))

axs[0,0].set_title('Univariate Linear Regression')
axs[0,0].plot(x_train,linear_regression(x_train))
axs[0,0].set_xlabel('x_train');  axs[0,0].set_ylabel('y-hat')

axs[0,1].set_title('Cost vs. weight')
axs[0,1].plot(w_hist,J_hist)
axs[0,1].set_xlabel('weight');  axs[0,1].set_ylabel('Cost')

axs[1,0].plot(J_hist[:1000])
axs[1,0].set_title("Cost vs. iteration(start)")
axs[1,0].set_xlabel('iteration step');  axs[1,0].set_ylabel('Cost')

axs[1,1].plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
axs[1,1].set_title("Cost vs. iteration (end)")
axs[1,1].set_xlabel('iteration step');  axs[1,1].set_ylabel('Cost')

plt.show()
```

---

유용균, 2024-12-29 <br>
Coursera, Andrew Ng 교수님께 감사드립니다.
