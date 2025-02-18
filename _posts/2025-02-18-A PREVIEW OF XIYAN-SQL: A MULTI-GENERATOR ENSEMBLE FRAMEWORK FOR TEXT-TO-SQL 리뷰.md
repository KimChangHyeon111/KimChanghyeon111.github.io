---
layout: posts
title: "A PREVIEW OF XIYAN-SQL: A MULTI-GENERATOR ENSEMBLE FRAMEWORK FOR TEXT-TO-SQL 리뷰"
summary: A PREVIEW OF XIYAN-SQL A MULTI-GENERATOR ENSEMBLE FRAMEWORK FOR TEXT-TO-SQL 리뷰
author: Kim Changhyeon
categories: papers
published: True
toc: True
toc_sticky: True
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

## Overall Framework
![image](https://github.com/user-attachments/assets/0fe8c296-984e-4091-b5d9-2867becd06f6)
1. Schema Linking : 관련 열 선택 / 값 검색하여 M-스키마로 구성되어 2)로 전달
2. Candidate Generation :  후보 SQL 생성하여 자체 정재 과정을 통해 정재
3. Candidate Selection : 최종적으로 후보를 비교하여 최종 SQL 결정

## M-Schema
![image](https://github.com/user-attachments/assets/1f17031e-23b2-4a57-b441-1a70b33b5833)
데이터베이스 / 테이블 / 열 사이의 계층적 관계를 반구조화된 형식으로 보여주며, 식별을 위해 특정 토큰을 사용.
- "[DB_ID], [Schema]" : 데이터베이스, 스키마
- "# table" : 테이블
- "[Foreign Keys]" : Foreign Keys.
- 각 테이블의 정보는 list로 저장되며, 각각의 아이템은 열에 대한 상세 정보를 담고 있는 tuple임. 각 정보는 column명, data type, column에 대한 설명, primary key 구분자, 예시로 이뤄짐. forign key는 그 중요성 때문에 별도로 명시되어야 함.

## Schema Linking
- 스키마 링크는 자연어 쿼리의 참조를 테이블 / 열 / 값을 포함해 데이터베이스 스키마에 연결합니다. Retrieval Module과 Column Selector로 구성됩니다.
- Retriver Module : 모델에 few-shot example을 제공해 문제의 키워드와 엔티티를 알 수 있도록 합니다.
    - Column Retriver를 활용해 관련 열을 검색합니다. 의미론적 유사성을 기반으로 TopK 열을 검색합니다.
    - Value Retriever를 활용해 시멘틱 유사성과 Locality Sensitive Hashing을 사용해 유사값을 식별합니다.
    -   
- Column Selector : Retriever의 Column과 Value retriever를 Union을 받으면 Column Selector는 M-스키마로 제공된 이전 단계의 스키마에 대해 few-shot 방법을 통해 실제 쿼리와 연관성을 평가하고 진짜 필요한 열만 가져오게 수정합니다.

## Candidate Generation 
![image](https://github.com/user-attachments/assets/40fcc374-8984-4eef-8478-6c6163021cc0)
- 



