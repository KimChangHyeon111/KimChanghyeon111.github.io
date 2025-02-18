---
layout: post
title: "A PREVIEW OF XIYAN-SQL: A MULTI-GENERATOR ENSEMBLE FRAMEWORK FOR TEXT-TO-SQL 리뷰"
summary: A PREVIEW OF XIYAN-SQL A MULTI-GENERATOR ENSEMBLE FRAMEWORK FOR TEXT-TO-SQL 리뷰
author: Kim Changhyeon
categories: papers
---

# A PREVIEW OF XIYAN-SQL: A MULTI-GENERATOR ENSEMBLE FRAMEWORK FOR TEXT-TO-SQL 리뷰

기존의 LLM기반 NL2SQL 솔류션에는 2가지 접근법 존재
  - 프롬프트 엔지니어링
    - zero / few shot 프롬프트를 GPT / Gemini등에 사용해좋은 결과를 나타냈지만, 추론 오버헤드를 초래하는 경우가 많음.
  - SFT
    - 더 작은 모델을 미세조정해 제어가능한 SQL을 생성하지만, 제한된 매개 변수로 인해 새로운 도메인에 적용하기 힘듦

 이 논문에서는 새로운 프레임워크 **XIYAN-SQL**을 제안함 : 프롬프트 엔지니어링과 SFT를 결합해 후보 SQL을 다양하게 생성하는 Ensenble Startegy. 2단계 교육을 진행할 예정

 1. 먼저 모델의 SQL 생성 기능을 activate하고 다양한 스타일 선호도를 갖는 모델로 전환.
 2. 이후 refiner가 각 결과 / 오류를 기반으로 논리적 / 구문 오류를 수정.
 3. 최종적으로 selection agent가 best option 선택하는 구조

 또한 모델의 스키마에 대한 이해를 높이기 위해, **M-Schema** 라는 새로운 표현방법 제안. 
 - 데이터베이스 / 테이블 및 열 사이의 계층 구조를 semi-structured form으로 제안.

![image](https://github.com/user-attachments/assets/0fe8c296-984e-4091-b5d9-2867becd06f6)
