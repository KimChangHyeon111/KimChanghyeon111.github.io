---
layout: posts
title: "Large Language Model as a Teacher for Zero-shot Tagging at Extreme Scales 리뷰"
summary: "Large Language Model as a Teacher for Zero-shot Tagging at Extreme Scales 리뷰"
author: Kim Changhyeon
categories: papers
published: True
toc: True
toc_sticky: True
---
### 1. 소개
- Extreme Zero-shot Extreme Multi-label Text Classification (EZ-XMC) : 수천개의 라벨 중에 적절한 라벨을 찾아야 하는데, 레이블이 없는 문제. 다만 레이블의 설명은 존재하는 상황에 사용이 가능함.
- 본 논문은 LLM으로 고품질 pseudo-label을 생성하고, 추론시에는 경량 Bi-Encoder만 사용하는 LMTX 기법을 제안.
   - LMTX는 **더 적은 데이터를 필요**로 하는데, 애초에 pseudo-label과 설명 같에 높은 상관성이 있는 데이터를 가져와, 데이터의 퀄리티가 좋기 때문임.
   - LMTX는 **더 가벼운 모델**을 만드는데, bi-encoder만 사용하고 LLM이 연관되지 않기 때문.

### 2.문제 정의
- X_i : 문서. ex. 아마존의 상품 설명. 정답 라벨이 없다는게 EZ-XMC 문제의 특징.
- l_i : 라벨. X와 매핑이 되지는 않지만, 텍스트 설명은 존재함.
- E_θ : 문서 X에 대해 적절한 l을 찾는 것이 목적. 이를 위해 둘 다를 동일한 임베딩 공간(S)에 매핑하는 함수가 바로 E_θ. Bi-Encoder구조로 구현됨.

  **Bi-Encoder 구조**
  - 문서와 라벨을 각각 Transformers 기반 인코더에 넣어 임베딩
  - 두 인코더는 가중치를 공유
  - 이 때 문서와 라벨의 관련성은 두 임베딩간의 코사인 유사도로 계산하며, DistillBERT 기반으로 구성


### 3. Bi-Encoder 훈련
- **3단계의 반복적인 프레임워크**를 채택.
![image](https://github.com/user-attachments/assets/5916d23c-56fe-4fb5-a813-82be39927731)
**Bi-Encoder와 LLM의 피드백 수용, 학습 루틴 시각화**
  1. 모든 문서와 라벨을 인베딩한 후, 라벨 임베딩에 대한 근사 최근접 탐색을 통해 문서별 후보 라벨을 얻음 (Cos)
    - 그냥 LLM에 다 때려넣으면 안됨? -> 계산 복잡도도 높고, 비쌈
    - 그래서 Bi-Encoder를 이용해 임배딩 후, 근사 최근접 탐색을 통해 j개의 후보 라벨을 선택     
  2. LLM을 활용해 이 라벨과 문서에 대한 관련성 검토를 수행하고, 이를기반으로 pseudo-positive label을 식별한다
    - 이를 가지고 다음과 같은 질문으로 LLM에게 프롬프트로 제공
      ```ini
      document = {X_i}, is the tag {l_k} relevant to the document? answer yes or no
      ```
    - 최종적으로 yes를 받은 라벨 중 하나만을 활용해 Bi-Encoder훈련
      <span style="color:red">: 뭐로 고르는지 기준은 딱히 안나오는듯. 아마...유사도 기준으로 고르지 않을까?</span>

  3. 이를 가지고 Bi-Encoder를 훈련한다.
    - Triplet Loss를 손실함수로 사용:
      $$ L = \sum_{i=1}^{N}\; \sum_{k' \neq p} \max\!\Bigl(0,\, \langle E_{\theta}(X_{i}),\, E_{\theta}(l_{k'}) \rangle - \langle E_{\theta}(X_{i}),\, E_{\theta}(l_{p}) \rangle + \gamma \Bigr) $$

    - 문서마다 하나의 양성 라벨을 사용
    - 


