# ExCoT 논문 요약 & 원문 인용 정리
*Execution‑Guided Chain‑of‑Thought Direct Preference Optimization*  
Bohan Zhai et al., 2025 (Snowflake Inc.)

---

## 1. 연구 배경 및 문제의식

**설명**  
Text‑to‑SQL에서 단순 CoT 노출이나 CoT 없는 DPO만으로는 성능이 거의 향상되지 않는다. **중간 추론(CoT) + 검증 가능한 실행 피드백**이 함께 있어야 의미 있는 개선이 가능하다.  

> *“…zero‑shot CoT — without any dedicated training — does not provide improvement over baseline methods in text‑to‑SQL tasks … directly applying Direct Preference Optimization (DPO) to text‑to‑SQL without incorporating a CoT does not substantially boost performance…”*  

---

## 2. ExCoT 개요

**설명**  
ExCoT는 **CoT 추론 경로**를 노출하고, **SQL 실행 결과(정답/오답)만을 피드백** 삼아  
1. **Off‑policy DPO**(GPT‑4o 생성 CoT 사용)  
2. **On‑policy Iterative DPO**(모델 자체 생성 CoT 사용)  
를 순차적으로 수행한다. 별도의 보상 모델이나 휴먼 라벨이 전혀 필요 없다.  

> *“…we propose Execution‑Guided Chain‑of‑Thought Direct Preference Optimization (ExCoT)… combining CoT reasoning with off‑policy and on‑policy iterative DPO, **using only execution accuracy as feedback**. Crucially, this approach does not rely on a specialized reward model or human‑annotated preference data…”*  

---

## 3. 방법론 세부 과정

### 3‑1. SFT 데이터 생성
- **설명**: GPT‑4o가 *최대 32개* CoT + SQL 후보를 생성 → SQLite에서 실행하여 **정답만** 보존 → 약 1.2 만 건 SFT 세트 구축.  
- **원문**

  > *“…we generate multiple candidate solutions (up to 32)… only correct solutions — i.e., those whose execution output matches the ground‑truth query’s result — are retained. This yields a high‑quality Supervised Fine‑Tuning (SFT) set…”*

### 3‑2. Off‑policy DPO
- **설명**: GPT‑4o 정답/오답을 *win‑lose pair*로 구성, **편집거리(farthest)**가 가장 큰 페어를 사용해 1차 DPO 수행.  
- **원문**

  > *“…we arrange them into positive (win) and negative (lose) pools… **selecting those with the largest discrepancies** for our final off‑policy DPO training set…*  

### 3‑3. On‑policy Iterative DPO
- **설명**: 개선된 모델이 새 CoT 후보를 생성 → 실행 검증 → **편집거리(nearest)**가 가장 작은 페어로 2‑3차 DPO 수행.  
- **원문**

  > *“…we further refine the model via on‑policy DPO… we **select pairs with smaller edit distance**… correcting them can lead to rapid incremental improvements.*  

---

## 4. 주요 결과

| 모델 & 단계 | BIRD EX % | Spider EX % |
|-------------|-----------|-------------|
| Baseline (No‑CoT) | 57.37 | 78.81 |
| + SFT(GPT‑4o) | 62.03 | 83.00 |
| + Off‑policy DPO | 66.30 | 82.49 |
| **+ On‑policy DPO (최종)** | **68.51** | **86.59** |

> *“Stage 3: On‑Policy Iterative DPO — LLaMA‑3.1 70B Complex → 68.51 (EX %), Valid 98.50 %”*  

또한 CoT 없는 DPO는 **+2 pp** 수준에 그침.  

> *“…baseline trained with DPO on SQLs **without** the CoT reasoning path only outperforms … by 2.6 % and 2.2 % on BIRD and Spider, respectively.”*  

---

## 5. 실무 인사이트

- **실행 결과만으로 자가 학습**  
  - 사내 샌드박스 DB만 확보하면 **라벨 비용 없이** 맞춤형 Text‑to‑SQL LLM을 지속 개선할 수 있다.  
- **구조화된 CoT가 핵심**  
  - Divide‑and‑Conquer 형태의 길고 구조적인 CoT가 가장 큰 성능 향상을 보였다.  
- **데이터 효율성**  
  - GPT‑4o 32 샘플 중 소수(약 5 개)만 정답이어도 충분한 성능 상승.  
- **SQL 유효성**  
  - Validity ≈ 99 % → 향상은 단순 문법 교정보다 *논리* 개선에 기인.  

> *“…the average chain‑of‑thought (CoT) length increases from 560 tokens … to 910 tokens by the final on‑policy round…”*  

---

## 6. 제약 및 향후 과제

- **복잡 스키마**에서는 추가 도메인 예시·지식 필요.  
- CoT가 길어지면 **중복·모순**이 생길 수 있어 후처리 검증 로직 필수.  
- Online‑DPO, PPO, 자유형 CoT 등 **강화학습 결합**은 앞으로의 연구 과제.  

> *“…our approach may still struggle under highly intricate schemas… CoT traces may contain partial truths, redundant steps, or contradictions…”*  
> *“…this work only applies offline‑DPO… More advanced approach, like Online‑DPO, PPO, and GRPO, are left for future exploration.”*  

---

### 참고 자료
- 전체 실험 설정·비용은 논문 4.2 절과 Table 2, 4 참조.  
- 논문: **arXiv 2503.19988v1** 
