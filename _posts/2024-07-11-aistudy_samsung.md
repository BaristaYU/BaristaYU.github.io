---
classes: wide
title: "삼성병원 AI강의"
categories:
  - Uncategorized
tags:
  - AI
  - deep_learning
  - study
---

**모르는용어**
선형회귀
BC로스
MLP
Linear Activation
Gradient decent
cross entropy (loss):
  분포 차이를 나타냄
one-hot encoding
다중 분류: 이진분류가 아닌 그 이상의 분류
각 노드가 각 결과를 담당하도록
softmax (합이 1, 양수) : 확률 분포
이미지는 3차원 행렬

FC LAYER에 통과하면 한 픽셀 한 픽셀을 따져가며 찾으므로
인간의 사고과정과 다름. (사람은 그냥 딱 보면 뭐다 하고 알므로)
-> 정형 데이터에 유리한 편

CNN : 비정형 데이터에 유리한 편
객체가 뭔진 몰라도
수만 장을 학습하며 '특정 패턴'을 익혀서
추론하는 것
weight는 ai가 알아서 알아내는 것

1. 일부 픽셀에 weight 주어 확인
2. weight 그대로 유지하여 다른 일부 픽셀에 확인
3. 2번에서 확인한 노드들은 일정 '위치'를 담당하는 노드가 됨
4. 2번에서 만든 웨이트 행렬을 **커널 혹은 필터**라고 부름. (bias 포함)
5. 필터한 결과를 깊이 축(채널 축)으로 쌓은 것을 feature map이라 함

개채 행렬? : 개수 * 채널 * 행 * 열

Feature map 간의 weighted sum:
  커널 사이즈가 1x1일 경우 각 픽셀로 움직이는 것이 곧 그냥 한 피쳐맵에 곱하는것과 같으므로
  -> one by one cnn

cnn 필터의 개체행열에서 정해줘야할 하이퍼파라미터는
개,행,열 : 채널은 정해져있음 (입력의 채널 수)

Padding
1. 사이즈 유지
2. 가생이 정보
- 구석에 있는 픽셀같은건 자주 필터링되지 않으므로 외각에 0과 같은 더미데이터를 넣음

Stride

Pooling layer
-> **maxpool**
.. 그렇게 해서 축약된 정보를
이제 fc layer를 거쳐서 판단함

VGGnet
flatten


