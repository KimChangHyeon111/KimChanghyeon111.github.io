---
layout: single
title: "Graph-Reward-SQL : Execution-Free Reinforcement Learning for Text-to-SQL via graph matching and stepwise reward 리뷰"
summary: "Graph-Reward-SQL : Execution-Free Reinforcement Learning for Text-to-SQL via graph matching and stepwise reward 리뷰"
author: Kim Changhyeon
categories: papers
published: True
toc: True
toc_sticky: True
comments: True
---
### 1. 소개
- 기존 RL 기반 Text to SQL 의 문제점
  - Execution Accuracy (Ex) 보상 모델 : 실제로 SQL을 실행해서 보상 여부 결정 -> 느림
  - Bradley-Terry 모델 (BTRM) : LLM 임베딩을 사용해서 보상 여부 결정 -> GPU를 너무 많이 먹음
  - 이 두 문제를 해결하기 위해 GMNScore와 StepRTM이라는 새로운 보상 모델 도입

### 2.문제 정의
- Xi : 질의.
- q^, q : 생성된 쿼리와 정답 쿼리
- πθ : 정책 모델. 보상을 최대화 하는 방향으로 정책 모델을 학습시키는 게 목표. 다시말해, 두 q가 의미적으로 가장 유사하도록, **보상을 최대화 하도록 하는 파라미터 θ**를 찾는 게 목표
 
### 3. Preliminaries
##### EX
```sql
q = SELECT score FROM posts WHERE user = ‘A’
```
- 일 때, LLM 이 도출한 쿼리의 결과가 정답과 일치하면 보상.   
    🟢 정확한 쿼리를 판단이 가능  
    🔴 실행환경이 추가로 필요. 매우 느리고 오류가 많음.

##### BTRM
- 쿼리의 실행 없이 NL 질문과 SQL 쿼리를 LLM에 넣고, 이를 임베딩하여 벡터를 점수로 변환하는 방식. 선호쌍 기반으로 학습하게 됨.
- 예컨대 평균 학생 수보다 많은 대학의 이름을 찾으라는 질문이 있다면

```sql
--나쁜 쿼리 :
SELECT name FROM college WHERE students < 1000

--좋은 쿼리 :
SELECT name FROM college WHERE students > (SELECT AVG(students) ...)
```
- 이런 데이터로 선호쌍 학습을 하는 것.  
    🟢 실행이 필요 없음.  
    🔴 여전히 LLM을 타야 하므로 무거움


### 4. 논문에서 제공하는 방법론
- GMNScore를 도입하고
- CTE SQL에는 StepRTM을 도입하겠다!
![image](https://github.com/user-attachments/assets/09f62bf9-4d6d-4fe1-8077-aabf44b81136)
  
**보상함수**  
![image](https://github.com/user-attachments/assets/5e737925-7aae-45ae-849f-59488195c38a)
1. 서브쿼리가 완성되었으면 -> StepRTM을 보상으로 적용하겠다
2. 쿼리가 다 끝났으면 -> GMNScore를 보상으로 적용하겠다
3. 항상 KL-divergence 패널티는 유지하겠다

- 그러면 GMNScore랑 StepRTM이 무엇이냐?
- 그 전에 GMN을 알려면 ROT를 알아야 함.

#### ROT
- SQL을 단순히 구조만 보는 것이 아니라 (구조만 보는 것을 AST라 함) 대수적 표현으로 바꾸어 의미 중심의 비교가 가능하게 하는 것.
- SQL의 SQL의 SELECT, WHERE, JOIN 같은 구문을 논리 연산자 노드들로 구성된 트리 형태로 변환
  - 트리의 노드(Node): Project, Filter, Join, Aggregate 등의 연산자
  - 엣지(Edge): 연산 순서와 데이터 흐름
- 예컨대 아래와 같은 쿼리가 있다면
```sql
WITH temp AS (
  SELECT id FROM users WHERE displayname = 'Stephen Turner'
)
SELECT AVG(score)
FROM posts
WHERE owneruserid IN (SELECT id FROM temp);
```
- Apache Clcite를 사용한 ROT는 아래와 같은 구조가 됨
```
Project (AVG(score)
 └── Filter (owneruserid ∈ temp_ids)
      └── Scan(posts)
CTE temp:
 └── Filter (displayname = 'Stephen Turner')
      └── Scan(users
```
- 이를 통해 표현은 달라도 의미가 같은 쿼리를, 구조적으로 유사도를 잡아낼 수 있음.

#### RelPM
- (아마도) 기존 그래프 임베딩 기반 보상 모델로 보임
- 이 각각의 노드들에 대해서 reference q의 모든 노드와 비교하고
- 연산자 타입이 같으면 매칭후보로 간주해서
- 가장 높은 후보 노드를 최종 매칭 노드로 선정

```sql
-- 참조 쿼리
SELECT name FROM users WHERE age >= 30

-- 생성 쿼리
SELECT name FROM users WHERE age > 29
```  

이럴 경우  
    🟢 실행이 필요 없고, Apache 이후에 큰 연산도 없고, 그래프 구조를 반영할 수 있지만.  
    🔴 타입은 같지만 (Filter) 값이 달라서(>, >=) 매칭을 실패하는 문제가 있음.


#### GMNScore
- 드디어.
- 앞서 ROT 그래프로 변환한 쿼리를 그래프 임베딩으로 생성한 뒤
- 두 그래프 벡터간의 유사도를 유클리드 거리로 비교하는 것
  1. Node Embedding  
각 노드를 초기 숫자 벡터로 표현함. 이 때 Global Positional Encoding이라 하여, 해당 노드의 그래프 내 상대적 위치 정보를 나타내는 벡터를 추가로 포함시킴
  2. Message Passing
이웃 노드 정보와 함께 벡터를 업데이트. 각 노드가 엣지로 연결된 이웃 벡터들의 평균 / 합/ attention등의 방법으로 업데이트 되는 것. 이 때 Cross Graph Attention이라는 것을 추가로 사용하는데, GMN의 목표가 그래프간의 의미적 유사도기 때문에, 각 그래프의 노드가 다른 그래프의 노드를 참고하여 값을 업데이트 한다고 함.
  3. Graph pooling
모든 노드 벡터를 종합해 그래프 전체의 임베딩 (hG) 생성. 전체 노드들의 임베딩을 하나로 합치는 것.
- 이 최종적인 pooling이 끝난 hG간의 유사도를 계산한 것이 GMNScore

#### StepRTM
- CTE 구조별로 쿼리를 나눠서, 각 서브쿼리마다 보상을 제공하는 Stepwise reward 모델
- 생성 시점마다 즉시 보상을 제공하여, 중간에 피드백을 제공하는 방식
- 쿼리를 CTE 서브쿼리들의 집합으로 나누고, 각 서브쿼리들을 ROT로 변환해 정답과의 매치를 확인하는 것.




### 5. 실험 및 결과

| Method                           | Spider | BIRD  | Avg.  |
|----------------------------------|--------|-------|-------|
| DeepSeek-Coder-1.3B-Ins          | 39.56  | 11.34 | 25.45 |
| + SFT                            | 57.74  | 13.30 | 35.52 |
| + PPO w/ AstPM                   | 59.96  | 12.52 | 36.24 |
| + PPO w/ RelPM                   | 62.86  | 14.67 | 38.77 |
| + PPO w/ BTRM                    | 61.41  | 14.21 | 37.81 |
| + PPO w/ EX                      | 65.28  | 17.21 | 41.25 |
| + PPO w/ GMNScore (Ours)         | 67.70  | 16.10 | 41.90 |
| DeepSeek-Coder-6.7B-Ins          | 44.97  | 18.38 | 31.68 |
| + SFT                            | 69.05  | 21.71 | 45.38 |
| + PPO w/ AstPM                   | 70.31  | 22.88 | 46.60 |
| + PPO w/ RelPM                   | 70.60  | 26.01 | 48.31 |
| + PPO w/ BTRM                    | 67.99  | 22.23 | 45.11 |
| + PPO w/ EX                      | 71.66  | 26.66 | 49.16 |
| + PPO w/ GMNScore (Ours)         | 72.44  | 26.14 | 49.29 |

1. ROT가 일반 쿼리 형태보다 (AST) 성능이 더 뛰어남 (AstPM vs RelPM)
2. 기존에 실행을 하지 않는 방법론이었던 BTRM은 EX에 비해 성능이 떨어졌지만, GMNScore는 그보다 성능이 더 뛰어남


| Method                                      | Spider        | BIRD           | Avg.           |
|---------------------------------------------|---------------|----------------|----------------|
| DeepSeek-Coder-1.3B-Ins                     | 39.56         | 11.34          | 25.45          |
| + SFT                                       | 57.74         | 13.30          | 35.52          |
| + CTE-SFT                                   | 46.81         | 14.02          | 30.42          |
| + PPO w/ RelPM & SFT∗                       | 62.86         | 14.67          | 38.77          |
| + PPO w/ RelPM & CTE-SFT∗                  | 62.57 (↓0.29) | 19.36 (↑4.69)  | 40.97 (↑2.20)  |
| + PPO w/ RelPM & StepRTM & CTE-SFT         | 64.31 (↑1.74) | 19.43 (↑0.07)  | 41.87 (↑0.90)  |
| + PPO w/ EX & SFT∗                          | 65.28         | 17.21          | 41.25          |
| + PPO w/ EX & CTE-SFT∗                      | 65.09 (↓0.19) | 18.97 (↑1.76)  | 42.03 (↑0.78)  |
| + PPO w/ EX & StepRTM & CTE-SFT∗           | 67.89 (↑2.80) | 19.75 (↑0.78)  | 43.82 (↑1.79)  |
| + PPO w/ GMNScore & SFT∗                    | 67.70         | 16.10          | 41.90          |
| + PPO w/ GMNScore & CTE-SFT∗                | 68.57 (↑0.87) | 20.40 (↑4.30)  | 44.49 (↑2.59)  |
| + PPO w/ GMNScore & StepRTM & CTE-SFT∗     | 68.67 (↑0.10) | 21.97 (↑1.57)  | 45.32 (↑0.83)  |

3. 또한 StepRTM을 추가로 도입해 중간중간 보상함수를 적용해줬더니 성능이 향상된 것을 알 수 있음.
4. DB가 더 어렵고 복잡한 BIRD가 이런 이점을 더 누리는 것을 알 수 있다고 하는데 대충 비슷한거 아닌가 싶긴 함

#### 느낀 점
1. 그래프 기반 모델의 성능이 뛰어남을 증명했고.
2. 실제 실행 없이도 비슷한 성능의 강화학습을 할 수 있다는 게 가장 큰 시의점임.
3. StepwiseRTM을 도입해 CTE 성능을 크게 개선했다는 점도 유의미해보임
