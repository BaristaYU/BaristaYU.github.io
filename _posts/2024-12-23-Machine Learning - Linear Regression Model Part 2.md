---
classes: wide
title: "[INCOMPLETE] (ENG) Machine Learning - Linear Regression Model Part II"
categories:
  - 머신러닝
tags:
  - AI
  - Andrew Ng
  - Supervised learning
  - Linear regression model
  - Gradient descent
---

Hello, my name is Yongkyun Yu.
I'm Korean, and I'm not good at English, so I'm trying to improve my English skills by sharing some data science knowledge in English without translator.
You'll learn some knowledge, and I'll learn some English skills. Quite cool, isn't it? Thank you for your understanding, I'll try my best.

This post was written after attending Professor Andrew Ng's Supervised Machine Learning: Regression and Classification Course at Coursera.<br>
It'll be nice to understand this post if you know how to use numpy module in python and some mathmatical knowledge, but it's okay if not.

---

### Gradient Descent - Introducing

In last post, we learned some knowledges about `Univariate Linear Regression Model`.
If you forgot, check the last post quickly.

[Machine Learning - Linear Regression Model Part I](https://baristayu.github.io/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/%ED%9A%8C%EA%B7%80%20%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EC%84%A0%ED%98%95%ED%9A%8C%EA%B7%80-%EB%AA%A8%EB%8D%B8-1/)

Do you remember about `Cost function`? It's a function that indicates difference between $\hat{y}$ and $y$, and its parameters were $w,b$.
Before learning about `Gradient Descent`, let's check about the graph of `Cost function` below.

![image](https://github.com/user-attachments/assets/c783a325-6611-43ba-b7e1-a8ede07c06cf)

It's a graph of $J(w)$ with respect to $w$. Forget about $b$ for a while. I'll talk about later.
Following the graph, its shape looks `Convex`. When $J(w)$ is low, it means predicted value is accurate, so `Vertex` of the graph is the most accurate value.

$a$에서 $w$가 높아지면 예측값이 정확할 것 같고, $b$에서 $w$가 낮아지면 예측값이 정확할 것 같다. <br> 그런데 무턱대고 $w$를 조절했다간, 그래프의 꼭짓점으로 다가가기 어려울 것 같다. 우리가 $w$를 임의로 더하고 빼는 것은 1차원적 접근이지만, 실제 $w$는 2차원 그래프에 있기 때문이다. 그러나 2차원적 접근이 어디 쉬운가? 숫자가 조금만 올라가도 차이는 기하급수적으로 날 것이다.<br> 우리는 `미세한 부분으로 나누어` 조금씩 $w$를 조절해야 한다.
여기서 저번 시간에 얘기했던 `미분`의 필요성이 나온다. 미세한 부분으로 나누어 조금씩 가다보면 $J(w)$의 값이 그래프의 `경사를 따라 내려갈` 것이다. 이것이 `경사하강법`이다.

---

### 경사 하강법(Gradient Descent) - 그래프

![image](https://github.com/user-attachments/assets/0dfb2073-f1fe-45d5-ac3f-d08e3a6c31f8)
출처: [Google Developer](https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent?hl=ko) <br><br>
이제 $b$에 대해서도 생각해보자. 위는 $w,b$에 대한 $J(w,b)\text{(=Loss)}$의 그래프이다. 뭔가 한 눈에 들어오진 않아도, 일단 이것도 볼록해보이긴 한다. 맞다, 이 그래프에서 가장 낮은 곳인 `전역 최솟값(Global Minima)`이 우리에게 필요한 $w,b$를 제공할 것이다. 3D 그래프의 이해를 돕기 위해 간혹 `등고선도`의 형태로 표현되기도 한다:

![image](https://github.com/user-attachments/assets/b5dfa620-75d4-45de-9e14-57fc6bfc072a)

3D 그래프에서 보이는 타원의 모양과 비슷하게 생겼다. 안쪽, 색이 어두워질수록 높이가 낮고(=0에 가깝고), 바깥쪽, 색이 밝아질수록 높이가 높다. 등고선도의 타원 테두리에 있으면 $w,b$의 값이 달라도 $J(w,b)$의 값은 같다. (높이가 같기 때문)

---

### 경사하강법(Gradient Descent) - 공식

$\text{repeat until convergence}$:{<br>
$w=w-a{\frac{d}{dw}}J(w,b)$<br>
$b=b-a{\frac{d}{db}}J(w,b)$<br>
}<br>
이 것이 경사하강법의 공식이다. 단, 이 공식의 `등호`는 수학적 의미가 아닌 `프로그래밍적 의미(대입 연산자)`, 즉 `업데이트`를 의미한다. $a$는 또 뭐고, $\frac{d}{dw}$는 뭔지 도통 모르겠다. 하나씩 알아가보도록 하자.

- $\text{repeat until convergence}$: `수렴`할 때까지 `반복`한다는 것, 즉 `Global minima`에 도달할 때까지 반복하라는 뜻이다.
- $a$: `학습률(Learning rate)`, `경사를 얼마나 내려가는지` 결정하는 `하이퍼파라미터(hyperparameter)`이다. 값이 커질수록 경사를 많이 내려가고, 값이 작아질수록 경사를 적게 내려간다. **항상 양수로 설정한다.**
- `하이퍼파라미터`: 말 그대로 '하이퍼'한 파라미터, 파라미터의 파라미터다. 일반 `파라미터`는 경사하강법 등의 연산으로 결정되지만, `하이퍼파라미터`는 `개발자의 선택`에 의해 결정된다.
- $\frac{d}{dw}J(w,b),\frac{d}{db}J(w,b)$: 수학을 잘 하는 사람이면 알겠지만, $w$, $b$에 대한 `편미분`이다. 즉, **`접선의 기울기`, 경사 하강의 `방향`이다.** 수학 알러지가 있다면, 그냥 방향인 것만 이해해도 괜찮다.

한가지 더 알아야할 사실은, 경사하강법의 두 공식은 `동시에 진행`되는 것이 좋다. 왜 그런지에 대해서는 글 말미를 참고하자.

자, 이제 우리는 $w$, $b$에 $a$만큼, $\frac{d}{dw}J(w,b),\frac{d}{db}J(w,b)$방향으로 이동한다는 것을 알았다. 이제 `편미분`만 진행하면 된다. 일단, 결과는 다음과 같다.

$\text{repeat until convergence}$:{<br>
$w=w-a \frac{1}{m} \displaystyle\sum^m_{i=1}(f_{w,b}(x^{(i)})-y^{(i)})x^{(i)}$ <br>
$b=b-a \frac{1}{m} \displaystyle\sum^m_{i=1}(f_{w,b}(x^{(i)})-y^{(i)})$
<br>}
(편미분 과정에 대해서는 아래에 나와있는데, 모르겠다면 넘어가도 좋다.)

1장에서 배웠듯, $^{(i)}$는 $i$제곱이 아닌 $i$번째 항목이고, $m$은 데이터셋의 개수다.

1. $\frac{d}{dw}J(w,b)$의 값이 음수면, $w$의 값은 증가한다.
2. $\frac{d}{dw}J(w,b)$의 값이 양수면, $w$의 값은 감소한다.
3. $\frac{d}{db}J(w,b)$의 값이 음수면, $b$의 값은 증가한다.
4. $\frac{d}{db}J(w,b)$의 값이 양수면, $b$의 값은 감소한다.

$\frac{d}{dw}J(w,b)$는 상술했듯 **접선의 기울기**이다. 아래 그래프를 잠시 보도록 하자.
![image](https://github.com/user-attachments/assets/d5108da5-533b-419d-8702-8ba72dcb93a8)
<br>(강의 중 일부를 발췌하여 수정하였음.) <br>
접선의 기울기는 $\frac{\Delta{y}}{\Delta{x}}$, 즉 $x$ 변화량 분의 $y$ 변화량이다. <br> 첫번째 그래프에서 $x$가 1 감소했을 때, $y$는 2 감소한다. $\frac{-2}{-1}$ 이므로, `양수`의 값을 가져 $w$는 $a$만큼 `감소한다`. <br> 두번째 그래프에서 $x$가 1 증가했을 때, $y$는 2 감소한다. $\frac{-2}{1}$ 이므로, `음수`의 값을 가져 $w$는 $a$만큼 `증가한다`. <br>
$w$ 뿐만 아니라 $b$의 경우에도 마찬가지이다. <br>
이것을 원하는만큼 반복하여 최적의 $w,b$를 찾는 것이다.
<br> <br>
주의해야할 점은, $a$가 너무 크면, 함수 안에서 `진동`하며 벗어나게 된다. 아래 그래프를 보도록 하자:
![image](https://github.com/user-attachments/assets/6809c28a-4585-4f2d-8ec9-4373594fecb8)
<br>방향은 올바르게 설정되었으나, $a$가 너무 커서 오히려 함수 밖으로 `발산`하고 있다. <br>따라서, 경사하강법은 일반적으로 `작은 학습률`을 `여러 번` 실행하여 최적의 $w,b$를 구하게 된다. <br> 다만, 유의해야할 점이 있다.

1. 경사하강법이 진행될 수록 $\frac{d}{dw}J(w,b),\frac{d}{db}J(w,b)$의 값은 점점 `0에 수렴`하게 되므로,<br> `일정 횟수 이상 반복되면 변화량도 매우 작아지게` 된다.
2. `학습률` $a$값이 `너무 작다면` 마찬가지로 변화량이 매우 작아져 `속도가 느려진다`. <br> 단, 너무 큰 경우처럼 함수 밖으로 발산하지는 않는다.

`적절한` $a$값과 `적절한 시도 횟수`는 다음 글에서 `Python`을 통해 알아보도록 하자.

---

### 경사하강법 - 공식 유도

일단, $w$에 대한 `편미분`은, $w$만 변수로 보고 나머지는 상수로 취급한다는 뜻이다. <br>
미분을 하면, 변수의 차수는 변수의 계수가 되고, 차수는 하나 줄어들게 된다. 또한, 상수항은 미분 시 0이 된다. <br>예를 들면, $x^2+2xy+y+1$을 $x$에 대해 미분하면 $2x+2y$가 된다. <br>$x^2$는 $2x$로, $2xy$는 $x$가 $1$이 되므로 $2y$가 되었고, 상수항 $y,1$은 $0$이 되기 때문이다.<br>

유도에 앞서, $J(w,b)$는 $\frac{1}{2m}\displaystyle\sum_{i=1}^m(\hat{y}^{(i)}-y^{(i)})^2$ 이다. <br>또한, $\hat{y}^{(i)}$는 $f_{w,b}(x^{(i)})$이고, $wx^{(i)}+b$이다. <br> 모르겠다면, 1장을 다시 보고 오자. <br>

#### $w$ 유도과정:

$\frac{d}{dw}J(w,b)=\frac{1}{2m}\frac{d}{dw}\displaystyle\sum_{i=1}^m(\hat{y}^{(i)}-y^{(i)})^2=\frac{1}{2m}\frac{d}{dw}\displaystyle\sum_{i=1}^m(wx^{(i)}+b-y^{(i)})^2$ 이다. <br>
$(wx^{(i)}+b-y^{(i)})^2$를 전부 전개할 필요는 없다. <br> 힘들게 전개해봤자 어차피 $w$가 붙지 않은 것들은 전부 0이 되지 않겠는가? <br> $w$가 붙은 것만 전개해보자. <br> $w^2x^{2(i)}+wx^{(i)}b-wx^{(i)}y^{(i)}+wx^{(i)}b-wx^{(i)}y^{(i)}$
<br>이것을 $w$에 대해 편미분하면: <br>
$2wx^{2(i)}+x^{(i)}b-x^{(i)}y^{(i)}+x^{(i)}b-x^{(i)}y^{(i)}=2x^{2(i)}w+2x^{(i)}b-2x^{(i)}y^{(i)}$이다. <br>
여기서 $2x^{(i)}$를 밖으로 빼면:<br> $(wx^{(i)}+b-y^{(i)})2x^{(i)}$이 된다.<br>
이어서 공식에 대입하면:<br>
$\frac{1}{2m}\displaystyle\sum_{i=1}^m(wx^{(i)}+b-y^{(i)})2x^{(i)}$이 되므로, 2를 제거했을 때 <br>
$\frac{1}{m}\displaystyle\sum_{i=1}^m(wx^{(i)}+b-y^{(i)})x^{(i)}$이 된다. 위에서 말했듯, $wx^{(i)}+b$는 $f_{w,b}(x^{(i)})$이고,<br> $w$의 기울기 공식은 $w=w-a{\frac{d}{dw}}J(w,b)$이므로, 마지막으로 정리하면:<br>
$w=w-a \frac{1}{m} \displaystyle\sum^m_{i=1}(f_{w,b}(x^{(i)})-y^{(i)})x^{(i)}$ <br>
드디어 공식이 완성되었다. 1장에서 시작된 $\frac{1}{2m}$의 비밀이 풀렸다! 깔끔한 공식 유도를 위해 $2m$으로 나눈 것이다.

#### $b$ 유도과정:

$w$와 같은 맥락이므로, 생략하겠다. 미분을 연습할 겸 직접 해보는 것도 나쁘지 않겠다. <br> 최종 공식은:<br>
$b=b-a \frac{1}{m} \displaystyle\sum^m_{i=1}(f_{w,b}(x^{(i)})-y^{(i)})$

---

### 경사하강법 - 동시에 진행?

경사하강법의 두 공식은 `동시에 진행(simultaneously)`되는 것이 좋다. 만일 그렇지 못하면, 공식 속의 $f(x)$값이 영향을 받아 $w$ 업데이트 이후 $b$ 업데이트의 `방향`이 $\Delta{w}$($w$ 변화량)만큼 변하게 된다. 사진을 통해 알아보자.

![경사하강법](https://github.com/user-attachments/assets/1550edb8-0c3e-47bb-9136-160885117730)

아래의 경우가 동시에 진행하지 않았을 경우의 그래프이다. (대략적으로 그렸다.) 3번 화살표가 경사를 이동하는 방향이라 했을 때, 2번 화살표의 영향을 받아 어디로 움직일지 알기 어렵다. 따라서, 우리는 `제어 가능한` 경사의 하강을 위해 동시에 진행해야 한다. <br>
물론, 이것이 옳다, 틀리다의 문제는 아니다. 아래는 python으로 경사하강법을 동시에(동기), 순차적으로(비동기) 진행한 결과이다.

```python
동기 (w,b,J) found by gradient descent: (199.9929,100.0116,0.00000675)
비동기 (w,b,J) found by gradient descent: (199.9940,100.0096,0.00000469)
```

보이다시피, 비동기의 비용 J가 오히려 낮음을 알 수 있다. 비용만 생각한다면, 비동기가 더 좋을 수도 있겠다. 그러나, 이것은 올바른 접근법은 아니다. 우리는 아래와 같은 알고리즘이 필요하다:

1. 제어 가능하며(=안정적이며)
2. 비용이 낮은 $w,b$를 (가능하면 빠르고 쉽게) 찾을 수 있고,
3. $J(w,b)$가 `지역 극솟값(Local minima)`가 아닌 `전역 최솟값(Global minima)`에 수렴해야 한다.

경사에서 무작위로 통통 튀는 테니스공을 굴리는 것보다 묵직한 볼링공을 굴리는 것이, 경로 예측이 더 편하지 않겠는가? 경사하강법과 같이 비용 $J(w,b)$를 0에 가깝게 만드는 $w,b$를 찾는 함수를 `최적화 알고리즘(Optimization Algorithm)`라고 부르는데, 여러 종류가 있다. 1장에서 `데이터와 모델에 맞는 비용함수`를 쓰는게 중요했듯, 마찬가지로 적합한 `최적화 알고리즘`을 고르는 것도 개발자의 몫이자 역량이라 볼 수 있겠다.<br><br> 예를 들면, 3번의 `지역 극솟값(Local minima)`은 아래 그래프와 같이 목표가 아닌 곳을 의미하는데, `경사하강법`은 종종 이곳에 수렴해버리곤 한다. 글 중간에 **경사하강법은 일반적으로 `작은 학습률`을 `여러 번` 실행하여 최적의 $w,b$를 구하게 된다**고 했던 것을 기억하는가? `작은 학습률`로 인해 `지역 극솟값(Local minima)`을 벗어나지 못하는 것이다.<br> 이를 해결하기 위해 다양하게 파생된 `최적화 알고리즘`들이 있다. 이럴 때, `ADAM`과 같은 `최적화 알고리즘`을 통해 파훼할 수 있다. (여기서 설명하지는 않겠다.)

![image](https://github.com/user-attachments/assets/758017b6-d12a-422b-9e6c-fca56f25f207)

---

### 경사하강법 - 마무리

우리는 `경사하강법`이 무엇인지, 그래프는 어떻게 되는지, 그리고 공식의 유도와 전개까지 알아보았다.
`경사하강법`은 비용함수 $J(w,b)$의 값을 최소화하는 $w,b$를 찾는 `최적화 알고리즘`이며, 알고리즘을 동시에 적용함으로써 $w,b$를 $a$만큼 조절해간다. <br><br>다음 글에서, `Python`을 활용한 `일변량 선형회귀 모델`과 `경사하강법`의 구현을 통해 예제를 다뤄보고, <br> 얼만큼의 `학습률`로 얼마나 `반복해야` 낮은 $J(w,b)$와 적절한 $w,b$를 구할 수 있는지 확인해보도록 하겠다.

---

유용균, 2024-12-23 <br>
Special thanks to Coursera and Professor Andrew Ng.
