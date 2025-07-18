---
layout: single
title: "Data Augmentation for Deep Learning Regression Tasks by Machine Learning Models 리뷰"
summary: "Data Augmentation for Deep Learning Regression Tasks by Machine Learning Models 리뷰"
author: Kim Changhyeon
categories: papers
published: True
toc: True
toc_sticky: True
comments: True
---
### 1. 소개
- 딥러닝 모델은 기존의 Tabular Regression문제에서 ML을 여전히 잘 넘지 못함.
  - 왜냐면 DL은 더 많은 데이터가 필요하니까.
  - 그러면 Augmentation을 하면 되는 게 아닐까?

### 2.Related Work
- Data Augmentation   
  - 데이터 증강. 이미지를 돌리거나, 텍스트에서 어순을 바꾸는 등, Non-Tabular Data에서는 다양한 기법이 이미 나와있었는데, Tabular Data에는 이런 기법이 부족했음.
  - 기존에 나왔던 Mixup, ADA, VAE, HIST기반 기법등 기존에 나왔던 기법들도 효과가 제한적임.
- AutoDL
  - AutoKeras, H2O, Autogluon등 자동 DL 프레임우어크들이 나와버림.
  - 이걸 활용해서 Augmentation을 하면 되는게 아닐까?
 
### 3. Data Augmentation for Deep learning Regression using machine learning (DADR)
<img width="1643" height="528" alt="image" src="https://github.com/user-attachments/assets/199c4e24-de49-4312-9969-2b7b1e56042f" />
1. Train - Test로 나눈다
2. Train으로 AutoML을 학습시킨다
3. X_train과 noise를 더한 X_synth를 만들어, 2.에서 학습시킨 모델에 넣어 y를 예측하고 이를 y_synth로 잡는다.
4. train과 synth를 더해서 X,y combined를 만든다
5. 4의 combined를 가지고 AutoDL을 학습시킨다
6. 1에서 따로 빼 놓은 test로 결과를 평가한다. 이 때 Baseline은 1의 train만 가지고 학습시킨 AutoDL과 비교한다.

### 4. 실험 및 결과
- AutoML은 TPOT를 사용. 기본 매개변수로 10분간 학습.
- AutoDL은 AutoKeras, H2O, Autogluon 사용. H2O는 DL만 하게 제한, Autogluon은 최대 10분, H2O와 AutoKera는 200에포크로 제한.
- 20개의 데이터셋을 활용해 기존 증강 방법과 비교.

- 20개의 실제 데이터셋 대상으로
##### C-Mix / ADA / 순수 노이즈 추가 / 증강 없이 vs DADR로 비교 결과
<img width="810" height="659" alt="image" src="https://github.com/user-attachments/assets/9b759587-c4a2-46e2-999d-7ce7faffaacd" />



- 전체 AutoML에 대해서 # best가 가장 높은 것을 확인할 수 있으며, 특히 H2O에서는 50%에 해당하는 케이스에서 BEST를 반환하는 것을 볼 수 있음. 

##### Train Size와 증강 Size에 따른 성능 비교 (행이 Train Size, 열이 Augment Size)
<img width="725" height="737" alt="image" src="https://github.com/user-attachments/assets/567cab31-8cfe-427e-a279-15ad2fd7435c" />


- 이처럼 Train Size가 작을 수록, 증강을 더 만힝 할수록 증강의 효과는 더 커지는 것을 확인 가능. 
##### 증강 vs 증류
- 사실 지식 증류의 효과인거 아님?
- 어느 정도는 맞다고 볼 수 있음!
- 논문도 이부분에 대해서 설명이 부족함. 뭐 증강의 효과가 더 큰 걸 검증했다고 하는데, 일단 본인은 어케 검증했는지 못찾음.


##### 느낀 점
1. Tabular Data의 증강을 시도했다는 것 자체가 흥미로움. 실제로 업무를 하다보면 진짜 쥐꼬리만한 데이터로 자꾸 뭘 해야 하는 때가 찾아오고 그래서 찾아본 논문이기도 함.
2. ML을 가지고 DL을 학습시키는 접근이 아니더라도, ML to ML 형태의 증강도 가능하지 않나? 싶어서 Autogluon만 가지고 해봤는데 실제로 어느 정도 성과를 거뒀음. 앞으로도 종종 사용해볼듯.
3. 증류인지 증강인지 애매하게 끝내는 걸 보면 저자도 자신은 없는듯? 아무튼 성능이 올라갔으니 OK다 하고 넘어가기에는 좀 애매한 느낌이 있는 건 사실.
