---
layout: single
title: "Personas within Parameters: Fine-Tuning Small Language Models with Low-Rank Adapters to Mimic User Behaviors"
summary: "Personas within Parameters: Fine-Tuning Small Language Models with Low-Rank Adapters to Mimic User Behaviors 논문 리뷰"
author: Kim Changhyeon
categories: papers
published: true
toc: true
toc_sticky: true
comments: true
---
## 1. 논문의 문제의식과 출발점

이 논문은 추천 시스템에서 **사용자 행동 시뮬레이션을 어떻게 확장 가능하면서도 개인화 수준을 유지한 채 구현할 수 있는가**라는 문제에서 출발합니다. 기존 추천 모델은 로그 기반 예측 정확도는 높지만, 실제 서비스 환경에서의 사용자 반응과 괴리가 반복적으로 발생해 왔습니다. 이로 인해 오프라인 실험 결과가 실제 온라인 성능으로 이어지지 않는 문제가 지속적으로 제기되었습니다.

최근에는 LLM 기반 사용자 에이전트가 이러한 간극을 줄이기 위한 대안으로 제안되었으나, 논문은 LLM 중심 접근이 구조적으로 세 가지 한계를 가진다고 지적합니다. 첫째, 대규모 사용자 로그를 지속적으로 처리하기에는 비용이 과도합니다. 둘째, 사전학습 과정에서 형성된 inductive bias로 인해 특정 사용자의 고유한 선호를 정확히 내재화하기 어렵습니다. 셋째, 수백만 명의 사용자를 개별 모델로 다루는 것은 시스템적으로 불가능합니다.

이에 대해 논문은 **LLM은 사용자 이해를 위한 도구로 제한하고, 실제 사용자 행동을 모사하는 주체는 SLM으로 설계한다**는 명확한 역할 분리를 제안합니다. 즉, LLM은 해석과 증류에만 사용하고, 비용 효율적인 소형 언어모델(SLM)에 사용자 행동을 내재화함으로써 확장성과 실용성을 동시에 확보하는 것이 핵심 출발점입니다.

---

## 2. 문제 설정 및 핵심 개념

논문은 전통적인 추천 시스템 설정을 기반으로 사용자 에이전트를 정의합니다. 사용자 집합 U, 아이템 집합 I, 사용자–아이템 상호작용 기록 D가 주어질 때, 목표는 특정 사용자 u가 아이템 i에 대해 보일 반응 y를 예측하는 것입니다.

이때 논문이 도입하는 핵심 개념은 **단기 기억과 장기 기억을 갖는 사용자 에이전트**입니다.

- 단기 기억(Ms): 사용자의 비교적 안정적인 성향과 취향을 요약한 텍스트 프로필  
- 장기 기억(Ml): 과거 특정 아이템에 대해 내린 결정과 그 이유를 설명한 텍스트 기반 기억  

사용자 에이전트는 다음과 같은 형태로 정의됩니다.

- 입력: 사용자 단기 기억, 현재 아이템 설명, 장기 기억 중 일부
- 출력: 해당 아이템에 대한 사용자 반응(예: 평점)

중요한 제약은 SLM의 컨텍스트 길이가 제한되어 있다는 점입니다. 따라서 모든 과거 이력을 입력에 포함하는 것이 아니라, **현재 아이템과 유사한 일부 장기 기억만을 검색하여 사용**합니다. 이 선택적 기억 참조가 이후 RAFT 설계의 핵심 전제가 됩니다.

---

## 3. 제안 방법의 전체 구조

논문이 제안하는 방법은 세 단계로 구성됩니다.

### 3.1 사용자 페르소나 생성 (Stage 1)

첫 단계의 목적은 **표 형식의 사용자–아이템 상호작용 로그를 SLM이 이해 가능한 텍스트 지식으로 변환하는 것**입니다. 이를 위해 연구진은 GPT-4o를 고정된 상태로 사용하여, 사용자별 상호작용 이력 (제품 / 가격 / 카테고리 등의 구매 내역) 시간 순서로 배치 단위(batch ≤ 10)로 입력합니다.

이 과정에서 단순 요약이 아닌 **계층적 자기 성찰(hierarchical self-reflection)**을 적용합니다. 즉, LLM이 “무엇을 선택했는가”가 아니라 “왜 그렇게 선택했는가”를 서술하도록 유도합니다. 그 결과 사용자마다 다음 두 가지 산출물이 생성됩니다.

- Ms: 활동성, 동조성, 다양성 추구 성향, 상세 취향을 포함한 사용자 프로필
  1. 활동성 특성(Activity Trait): 영화 시청 빈도와 추천 시스템 이용 습관 (예: "이따금 시청하는 사용자, 취향에 엄격하게 맞는 영화에만 호기심을 가짐").
  2. 동조성 특성(Conformity Trait): 타인의 평점에 영향을 받는지 여부 (예: "균형 잡힌 평가자, 과거 평점과 개인적 선호를 모두 고려함").
  3. 다양성 특성(Diversity Trait): 새로운 장르를 탐색하려는 의지 (예: "시네마틱 개척자, 독특하고 잘 알려지지 않은 영화를 끊임없이 찾음").
  4. 프로필 설명(Profile Description): 일반적인 선호도와 혐오도에 대한 상세한 기술 (예: "위트와 깊이가 결합된 영화를 선호하며, 심리적 음모가 있는 스릴러에 높은 점수를 줌"). 
- Ml: 특정 아이템에 대해 좋아하거나 싫어한 이유를 설명한 강화된 상호작용 설명. 배치에 들어있는 아이템 개수만큼 반환되는 것으로 추정.

이 단계에서 LLM은 학습 대상이 아니라 **지식 추출기**로만 사용됩니다.

### 3.2 페르소나 기반 LoRA 미세 조정 (Stage 2)

모든 사용자마다 개별 LoRA를 학습하는 것은 비현실적이므로, 연구진은 Ms를 임베딩한 뒤 KMeans++로 사용자 클러스터링을 수행합니다. 실험에서는 엘보우 방법을 통해 페르소나 수를 K=4로 설정합니다.

각 페르소나 그룹에 대해 하나의 LoRA 어댑터를 학습하며, 베이스 SLM(Phi-3-Mini)의 가중치는 고정됩니다. 이 설계는 개인화 수준과 파라미터 효율성 간의 균형을 명확히 의도한 선택입니다.

### 3.3 사용자 에이전트 구성 (Stage 3)

추론 시에는 다음 요소들이 결합됩니다.

- 베이스 SLM + 해당 페르소나 LoRA  
- 사용자 고유의 단기 기억(Ms)을 시스템 프롬프트로 사용  
- 현재 아이템과 유사한 장기 기억(Ml)을 검색하여 입력에 추가  

이 구조를 통해 모델 파라미터 수는 제한하면서도, 사용자별 행동 차이를 프롬프트와 기억 수준에서 유지할 수 있습니다.

---

## 4. RAFT: Retrieval Augmented Fine-Tuning의 핵심 설계

이 논문의 중요한 차별점은 **RAG를 추론 단계뿐 아니라 학습 단계에도 사용한다는 점**입니다. 이를 논문에서는 Retrieval Augmented Fine-Tuning(RAFT)이라 부릅니다.

일반적인 미세 조정은 다음과 같은 구조를 가집니다.

- 입력: 사용자 프로필 + 현재 아이템  
- 출력: 실제 사용자 반응  

반면 RAFT에서는 학습 데이터 자체가 다음과 같이 구성됩니다.

- 입력:  
  - 사용자 단기 기억(Ms)  
  - 현재 아이템 설명  
  - 현재 아이템과 가장 유사한 과거 상호작용의 강화된 설명(Ml, k=1)  
- 출력: 실제 사용자가 부여한 평점  

즉, 모델은 학습 단계에서부터 **“과거에 비슷한 상황에서 왜 그런 결정을 내렸는지”를 함께 보고 현재 판단을 내리는 방식**을 학습합니다. 이는 단순히 기억을 참고하는 것이 아니라, **기억을 활용하는 사고 방식 자체를 LoRA 파라미터에 내재화**하는 과정으로 해석할 수 있습니다.

이 설계는 SLM이 갖는 장기 기억 한계를 보완합니다. 장기 기억 전체를 모델 파라미터에 저장하지 않고, 기억을 해석하는 규칙만을 학습시킴으로써 제한된 모델 용량에서도 개인화된 행동 모사가 가능해집니다.

---

## 5. 실험 설정과 평가 기준

실험은 MovieLens-1M 데이터셋을 기반으로 수행됩니다. 200명의 사용자를 선택하고, 사용자당 100~200개의 상호작용을 사용합니다. 데이터는 시간 기준 60:40으로 분할됩니다.

LLM 대상 비교 대상 및 설계는 아래와 같습니다. (뭔가 이부분 이상한데, test가 왜 더 많지)
- training set 크기 : 5132
- test set 크기 : 7496
- LLaMA-3-8B (프롬프트 기반)  
- Phi-3-Mini (프롬프트 기반)  
- Phi-3-Mini + 단일 LoRA  
- Phi-3-Mini + 페르소나별 LoRA  

평가 지표는 RMSE, MAE, 그리고 과제와 무관한 응답 비율을 측정하는 URR(Unrelated Response Rate)입니다.

페르소나 설정에서 설계는 아래와 같습니다. 
- 학습 데이터셋 크기는 클러스터별로 2000~5000개입니다.
- 상호작용 데이터는 200명 모두를 포함하나, 시간적으로 60:40으로 train - test를 시간적으로 분할했습니다.
- phi-3의 context window 한계를 감안해 2000단어보다 긴 결과는 제외했습니다.
- Rank = 256, Alpha = 32로 설정. 
- A100 80GB 8장으로 학습했다고 합니다.
---

## 6. 실험 결과 해석

실험 결과는 논문의 주장을 일관되게 뒷받침합니다.

첫째, **미세 조정된 SLM은 프롬프트 기반 LLM보다 우수하거나 동등한 성능**을 보입니다. 이는 모델 크기보다 사용자 지식을 어떻게 구조화해 내재화하느냐가 더 중요함을 시사합니다.

둘째, 단기 기억만 사용하는 경우보다 단기 기억과 장기 기억을 함께 사용하는 경우 대부분의 설정에서 성능이 개선됩니다. 이는 강화된 상호작용 설명이 실제 사용자 판단 논리를 효과적으로 보완함을 의미합니다.

셋째, 페르소나 기반 LoRA는 단일 LoRA 대비 데이터 효율성과 성능 간의 균형 측면에서 유리합니다. 다만 일부 페르소나에서는 데이터 부족으로 성능 저하가 관찰되며, 이는 클러스터 크기와 성능 간의 직접적인 연관성을 보여줍니다.

---

## 7. Ablation 및 추가 분석

Ablation 실험은 두 가지 가설을 검증합니다.

- 장기 기억(Ml)이 항상 도움이 되는가  
- 사용자 클러스터링이 실제로 의미 있는가  

결과적으로 장기 기억은 대체로 성능을 개선하지만, 검색 품질이 낮은 경우 오히려 일반화를 저해할 수 있음이 확인됩니다. 이는 RAFT 구조가 **검색 품질에 민감한 설계**임을 보여주며, 논문에서도 향후 연구 과제로 명시됩니다.

또한 페르소나별 성능은 학습 데이터 크기에 크게 의존하며, 이는 페르소나 설계가 단순한 군집 문제가 아니라 데이터 분배 문제와 직결됨을 시사합니다.

---

## 8. 종합 정리 및 개인적 평가

이 논문의 핵심 기여는 다음 한 문장으로 요약할 수 있습니다.

**“대규모 사용자 시뮬레이션은 거대 모델이 아니라, 지식을 잘 정제한 소형 모델로 구현할 수 있다.”**

- LLM을 사용자 이해를 위한 증류 도구로 한정하고, SLM + LoRA + 메모리 구조를 결합한 설계는 실무적으로 매우 현실적인 해법이라고 생각
- LoRA의 수를 소수로 한정하고, RAG와 결합해서 사용하는 접근도 현실적인 현업의 어프로치라고 생각
- 여전히 RAFT는 뭐가 특별한지 잘 모르겠음. 그냥...학습데이터...포맷을 다르게 만들었을 뿐인거 아닌가 싶고.
- 24년의 JP모건 조차도 우리 회사보다 GPU가 많다. 안타까울 따름이다.
- 생각보다 프롬프트로 제공되는 내용의 길이가 굉장히 길다. 2000안에 이게...처리가 되나? 싶을정도.

## 9. 전체 프로세스 총정리
1단계: 페르소나 생성 (Distillation)
- Input: 사용자당 100~200개의 데이터를 10개씩 배치 처리하여 GPT-4o에 입력. 전반적인 맥락을 놓치지 않기 위해 10개씩. 
- Output:
  - Ms : 4가지 사회적 특성이 담긴 요약 프로필 (시스템 프롬프트용).
  - Ml : 모든 과거 선택에 대한 '이유' 설명 (벡터 DB 저장용). 
- 제약: 전체 텍스트 2,000단어 이내.
- Ms와 Ml이 함께 나온 뒤, Ms는 DB에 추가, Ml은 멀티턴에서 과거 메모리를 참조하여 업데이트하는 구조.
- *근데 그냥 Ml을 직접 넣는 방식으로 하는게 토큰 측면에서 이득일 듯?*
  
2단계: 페르소나별 클러스터링 (Clustering)
- 대상: 사용자 프로필(Ms) 텍스트.
- 기술: text-ada-002 임베딩 + KMeans++ 알고리즘.
- 결정: 엘보우 방법을 통해 K=4로 확정.

3단계 : 클러스터별 LoRA학습
- input : 사용자의 단기 기억 (Ms) + 현재 아이템 설명 + 현재 아이템과 가장 유사한 Ml의 설명. 이를 RAFT라 함
- 가장 유사한 Ml설명 역시 Ml을 text-ada-002 임베딩 후 코사인 유사도로 top1추출
- 클러스터별 2000~5000개의 학습 데이터 대상으로, 동일 사용자의 데이터를 시간 기준으로 60:40으로 분할하여 학습. 즉 해당 클러스터에 있는 사람은 다 데이터에 들어가게 학습
- Rank = 256, Alpha = 32

4단계 : 예측
- 개별 사용자가 해당하는 cluster의 LoRA를 가지고 모델을 구성
- 프롬프트에 아래를 제공해 평가하게 구성
  - 해당 사용자의 Ms
  - 해당 item과 가장 유사한 아이템의 Ml
  - 해당 item에 대한 설명
  - Task정보


## 99. 논문에서 제공된 프롬프트
PERSONA: You excel at role-playing. Picture yourself as a user exploring a movie recommendation system. You have the following social traits:. Your activity trait is described as: An Occasional Viewer, seldom attracted by movie recommendations. Only curious about watching movies that strictly align the taste. The movie-watching habits are not very infrequent. And you tend to exit the recommender system if you have a few unsatisfied memories.. Your conformity trait is described as: A Balanced Evaluator who considers both historical ratings and personal preferences when giving ratings to movies. Sometimes give ratings that are different from historical rating.. Your diversity trait is described as: A Cinematic Trailblazer, a relentless seeker of the unique and the obscure in the world of movies. The movie choices are so diverse and avant-garde that they defy categorization.. Beyond that, your profile description expressing your preferences and dislikes is: I have a strong appreciation for movies that blend wit, depth, and compelling narratives. My highest ratings are often reserved for films that offer a mix of dark humor, psychological intrigue, and intense drama. I am particularly drawn to complex characters and intricate plots, as seen in my preference for thrillers and dramas with strong performances by seasoned actors. I enjoy historical and action-packed films, especially those with a rebellious or heroic theme. My taste also includes a fondness for classic comedies and animated features that provide a nostalgic or heartwarming experience . I tend to favor movies with ensemble casts where each actor brings a unique element to the story. While I appreciate a variety of genres, I am less enthusiastic about films that lack depth or fail to engage me emotionally. Overall, my movie-watching habits reflect a desire for thought-provoking and emotionally resonant storytelling, with a particular interest in films that challenge conventional narratives and offer fresh perspectives. However, I have realized that I also appreciate visually stunning and action-packed crime thrillers, as well as heartwarming adventure films, even if they are in the children’s genre. Additionally, I have a newfound appreciation for unique blends of genres, such as comedy and westerns, when they offer a fresh and engaging narrative. I also value films with strong character development and intricate plots, even if they are set in historical or war contexts, as long as they provide a compelling narrative and emotional depth. I have also come to appreciate older cinematic styles and classic films more than I initially thought, as long as they offer a compelling narrative and emotional depth. I also enjoy high-stakes, fast-paced narratives and unique storytelling approaches, as well as films with a strong visual and sci-fi element, provided they offer a compelling narrative..". The activity characteristic pertains to the frequency of your movie-watching habits. The conformity characteristic measures the degree to which your ratings are influenced by historical ratings. The diversity characteristic gauges your likelihood of watching movies that may not align with your usual taste. Given your persona and memory of some of movies you watched in the past, think carefully and rate the movie given in the end. Always use your memory at your own discretion as not everything is helpful. Also, historical average ratings of the movie mean an average rating of how other people have rated it, not you.

YOUR MEMORIES
MOVIE: The movie Falling Down was released in the year 1993 is of Action, Drama genre with historical average rating of 3.45.
MY MEMORY: A man’s descent into madness and violence as he navigates through the frustrations of modern life.. The cast of the movie is as follows: Michael Douglas, Robert Duvall, Barbara Hershey, Rachel Ticotin, Tuesday Weld, Frederic Forrest, Lois Smith, Joey Singer, Ebbe Roe Smith, Michael Paul Chan, Raymond J. Barry, D.W. Moffett, Steve Park, Kimberly Scott, James Keane. I rated movie Falling Down (1993) as 4 because it offers a compelling narrative of a man’s descent into madness and violence, aligning with my appreciation for psychological intrigue and intense drama. The strong performances by Michael Douglas and Robert Duvall, along with the thought-provoking and emotionally resonant storytelling, make it a highly engaging film.
TASK: Rate the movie given below on a likert scale from 1 to 5, where 1 means you hated it, 2 means you disliked it, 3 means you are neutral about it, 4 meaning you liked it and 5 means you absolutely loved it. Always respond with either one of 1, 2, 3, 4 or 5. Do not output anything else.
MOVIE: The movie Big Lebowski, The was released in the year 1998 is of Comedy, Crime, Mystery, Thriller genre with historical average rating of 3.74.
SUMMARY: A laid-back slacker gets caught up in a case of mistaken identity and embarks on a wild adventure involving bowling, kidnapping, and a rug that really tied the room together.. The cast of the movie is as follows: Jeff Bridges, John Goodman, Julianne Moore, Steve Buscemi, David Huddleston, Philip Seymour Hoffman, Tara Reid, Philip Moon, Mark Pellegrino, Peter Stormare, Flea, Torsten Voges, Jimmie Dale Gilmore, Jack Kehler, John Turturro
RATING:

### 999. Ms, Ml 추출하는 프롬프트
- 논문에 없음. NotebookLM이 출처.
- 멀티턴이 아니라 직접 Ml을 넣는 구조로 변환.

역할: 당신은 사용자의 행동 이면에 숨겨진 심리적 동기를 분석하는 전문가입니다.
기존 페르소나 정보: {{이전 단계에서 생성된 Ms}}

입력 데이터: 아래는 사용자 한 명의 영화 시청 이력 10개입니다. (영화명, 장르, 사용자의 평점 포함) 
[여기에 10개의 데이터 입력]

작업 지시:
1. 새로운 데이터 10개 각각에 대한 구체적 이유(Ml)를 추출하세요.
2. 기존 페르소나 정보를 바탕으로, 새로운 데이터를 반영하여 업데이트된 사회적 특성 및 프로필(Ms)을 다시 작성하세요. 전체 길이는 2000단어 이내여야 하며, 아래 4가지 섹션을 반드시 포함해야 합니다. 
    1. Activity Trait: 영화 시청 빈도와 추천 시스템에 대한 반응 습관을 기술하십시오.
    2. Conformity Trait: 자신의 취향과 대중적 평점 사이에서 어떻게 균형을 잡는지 기술하십시오.
    3. Diversity Trait: 익숙한 장르 외에 새로운 장르를 탐색하려는 의지의 정도를 기술하십시오.
    4. Profile Description: 사용자가 선호하는 서사 구조, 감정적 깊이, 좋아하는 배우의 스타일, 특별히 싫어하는 요소 등을 매우 상세하고 서사적으로 서술하십시오.

출력 형식:
- 아이템 1 (Ml): [이유 서술]
- 아이템 2 (Ml): [이유 서술]
  ...
- 아이템 10 (Ml): [이유 서술]
- 현재까지의 통합 성향(Ms) 메모: [4가지 통합 성향]
