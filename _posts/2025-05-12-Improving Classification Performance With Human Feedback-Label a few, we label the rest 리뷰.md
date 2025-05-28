---
layout: sinlge
title: "Improving Classification Performance With Human Feedback - Label a few, we label the rest 리뷰"
summary: "Improving Classification Performance With Human Feedback - Label a few, we label the rest 리뷰"
author: Kim Changhyeon
categories: papers
published: True
toc: True
toc_sticky: True
comments: True
---

### 1. 소개
- 기존 NLP 모델은 수백만개의 행에 크게 의존.
- 생성형 AI는 환각문제에서 여전히 완전히 자유롭지 못함.
- Programatic Labeling은 사용자가 labeling 함수를 이용해 일반화하는 것이지만, 미묘한 어감의 차이를 감지하기 힘들고, 잘못된 분류가 데이터 전체로 확산되며, 여전히 돈이 더 많이 듦.
- Language Models Are Few Shot Leaners (2020), Want To Reduce Labeling Cost? GPT-3 Can Help (2021) 등에서 수천개의 레이블링으로 AI가 학습하고, 이를 가지고 더 많은 데이터를 학습할 수 있음을 보임.
- HuggingFace가 22년 발표한 SetFit도 같은 맥락.


### 3. 프로세스
- Human Feedback Approach를 사용함. 
- 레이블이 지정된 10개를 반환하고, 각 반환마다 인간 피드백 적용 - 반환 - 오답 다시 모델에 입력 - 반환.

#### 3.1 데이터 셋
##### Amazon review :
  - 6001개의 train
  - 2001개의 test
  - Excellent - Very Good - Good - Neutral - Bad 5개의 Label
##### Banking dataset :
  - 200개의 train
  - 2000개의 test
  - Cash received, fait currency support, bin blocked등 77개의 label.
##### Craigslist: 
  - 201개의 train
  - 1001개의 test
  - 전화, 가구, 주택, 전자제품, 자동차 등의 input에 대한 ABBR / ENTY / DESC / HUM / LOC / NUM의 6개 label
##### financial phrasebank dataset:
![image](https://github.com/user-attachments/assets/1f33382b-726e-4905-a63e-439c780a799e)
  - 4850개의 rows
  - Pos / Neg / Neutral
##### Trec Dataset
![image](https://github.com/user-attachments/assets/6ecaa4ae-bb93-4d48-a583-38a043167de8)
  - 5452개의 train
  - 500개의 test
  - 6개의 labels - 50개의 세부 labels 

#### 4. Approach
- Probability : 모델의 예측에 대한 confidence level
- Entropy : 모델 예측의 Unpredictability - 엔트로피가 높다는 것은 해당 label이 전반적인 분류에 영향을 크게 끼친다는 뜻임.
- 일단 Calude / GPT / BERT / SETFIT 같은 모델로 사용한 제로샷 한 결과
  
| Text Body                                                              | Predicted | Probability | Entropy |
|------------------------------------------------------------------------|-----------|-------------|---------|
| Important notice: Your package has been delivered.                    | not spam  | 0.65        | 0.88    |
| Dear customer, Your account balance is low.                           | not spam  | 0.58        | 0.82    |
| Hi, How are you doing? Let’s catch up soon.                           | not spam  | 0.58        | 0.75    |
| Urgent notice: Last chance to update your personal information.       | spam      | 0.92        | 0.51    |
| Hi there, You have won a free vacation! Claim now!                    | spam      | 0.95        | 0.42    |
| Congratulations! You’ve won a million dollars!                        | spam      | 0.97        | 0.36    |

**Table 1: Zero shot model predictions**

- 이후 Probability가 낮은 케이스를 edge cases로 반환해, 이에 대해 Human Feedback을 제공하는 방식으로 다시 분류.

| Text                                                                 | Actual Label | Predicted  | Probability | Entropy |
|----------------------------------------------------------------------|--------------|------------|-------------|---------|
| Important notice: Your package has been delivered.                  | not spam     | not spam   | 1.00        | 0.00    |
| Urgent notice: Last chance to update your personal information.     |              | not spam   | 0.88        | 0.28    |
| Dear customer, Your account balance is low.                         |              | spam        | 0.70        | 0.76    |
| Hi, How are you doing? Let’s catch up soon.                         |              | not spam   | 0.65        | 0.68    |
| Congratulations! You’ve won a million dollars!                      |              | spam       | 0.92        | 0.22    |
| Hi there, You have won a free vacation! Claim now!                  |              | spam       | 0.80        | 0.25    |

**Table 2: Spam Classification Results**

- 이후 해당 프로세스를 반복하는데, Entropy를 기준으로 정렬하면 모델에 큰 영향을 끼치는 row를 찾을 수 있음.

| Text                                                                 | Actual Label | Predicted  | Probability | Entropy |
|----------------------------------------------------------------------|--------------|------------|-------------|---------|
| Important notice: Your package has been delivered.                  | not spam     | not spam   | 1.00        | 0.00    |
| Dear customer, Your account balance is low.                         |              | spam       | 0.70        | 0.76    |
| Hi, How are you doing? Let’s catch up soon.                         |              | not spam   | 0.65        | 0.68    |
| Urgent notice: Last chance to update your personal information.     |              | not spam   | 0.88        | 0.28    |
| Hi there, You have won a free vacation! Claim now!                  |              | spam       | 0.80        | 0.25    |
| Congratulations! You’ve won a million dollars!                      |              | spam       | 0.92        | 0.22    |

**Table 3: Iterated Model Predictions - First Iteration**

- "Dear Customer, Your account balance is low" 라는 문장이 레이블이 없는 데이터 중에 가장 높은 엔트로피를 가짐.
- 이 데이터를 추가하고 다시 데이터를 조정하면 아래와 같은 2번째 Iteration 결과가 나오게 됨.

| Text                                                                 | Actual Label | Predicted  | Probability | Entropy |
|----------------------------------------------------------------------|--------------|------------|-------------|---------|
| Important notice: Your package has been delivered.                  | not spam     | not spam   | 1.00        | 0.00    |
| Dear customer, Your account balance is low.                         | spam         | spam       | 1.00        | 0.00    |
| Hi, How are you doing? Let’s catch up soon.                         |              | not spam   | 0.80        | 0.62    |
| Urgent notice: Last chance to update your personal information.     |              | spam       | 0.86        | 0.18    |
| Hi there, You have won a free vacation! Claim now!                  |              | spam       | 0.75        | 0.20    |
| Congratulations! You’ve won a million dollars!                      |              | spam       | 0.88        | 0.18    |

**Table 4: Iterated Model Predictions - Second Iteration**
- 이런식으로 반복되면서 모델 엔트로피가 점차 감소하여, 최종적으로 더는 필요하지 않게 되는 시점이 오게 됨.


#### 5.평가방법
- BERT, SETFIT, GPT3.5 터보 모델 사용
- 모델의 잘못된 예측 10개를 수정하고 다시 미세조정  - 피드백 - 수정한 데이터 훈련 세트에 추가 - 반복
- 최대 150개의 레이블까지 적용

<span style="color:red">: 너무...적지 않아요? 150번만 해도 충분히 좋아진다면 데이터가 너무 쉬운거 아냐?</span>

#### 6. 결과
**그림 4,5: Amazon Dataset**
![image](https://github.com/user-attachments/assets/a31480dd-5c63-4313-af63-011dc6fb77ee)
![image](https://github.com/user-attachments/assets/5b63ab5a-4378-4961-aa4b-7c9fe6d619df)

**그림 6,7: Banking Dataset**
![image](https://github.com/user-attachments/assets/c4003bda-35be-4745-8250-da8d5d3e0245)
![image](https://github.com/user-attachments/assets/f6779c18-f753-4e7a-9483-f9521a3885df)

**그림 8,9: Craigslist**
![image](https://github.com/user-attachments/assets/7c05561d-caac-4146-9682-e98d8cd08eef)
![image](https://github.com/user-attachments/assets/557c4bdc-603e-40d4-be30-ae581ad2e01c)

**그림 10, 11 : Financial phrasebank dataset**
![image](https://github.com/user-attachments/assets/70b5e3eb-4ac6-4269-b77f-b0d762c25bd3)
![image](https://github.com/user-attachments/assets/7fb73d75-f550-4772-8459-0fe00eb03c9d)

**그림 12 ~ 15 : Trec Dataset**
![image](https://github.com/user-attachments/assets/4eaf0982-f494-4018-8ed0-5bcec63df62d)
![image](https://github.com/user-attachments/assets/5b8535c9-b179-405a-8697-8dd67763e70f)
![image](https://github.com/user-attachments/assets/2a8c72cb-3bb3-4e55-99da-4c04aeee7695)
![image](https://github.com/user-attachments/assets/1a7a62f9-40c8-4877-b666-8dd71255e087)

#### 6-2 : 소스코드
- 논문에는 어떻게 이 수치들을 뽑았는지 제대로 나오지 않았기 때문에, 주석에 가서 링크를 확인해 보기로 함.
- 가장 복잡한 구조인 banking의 GPT와 BERT버전의 소스코드를 확인해보기로 함.
- https://github.com/iiWhiteii/Anote-Text-Classification/tree/main

##### GPT3.5 Turbo
```python
# GPT가 주석만 번역해준 함수
def text_to_openai_json(data, filename):
    """
    주어진 데이터셋을 OpenAI GPT-3.5 turbo 모델에 적합한 JSON Lines (JSONL) 형식으로 변환합니다.

    인자:
        data (DataFrame 또는 유사 객체): 텍스트와 레이블을 포함한 입력 데이터
        filename (str): 저장할 JSONL 파일 이름

    이 함수는 입력 데이터의 각 행을 순회하면서,
    - system 메시지 (지시문)
    - user 메시지 (텍스트 예시)
    - assistant 메시지 (정답 레이블)
    의 대화 형태로 구성하여 JSONL 포맷으로 저장합니다.
    """

    # 대화 데이터를 저장할 빈 리스트 초기화
    message_list = []

    # 입력 데이터를 행 단위로 반복
    for _, row in data.iterrows():
        # system 메시지 생성: 모델에게 분류 기준 설명
        system_message = {
            "role": "system",
            "content": f"given the following text: find the category in: {categories} that is most closely associated with it. Return only the category name"
        }

        # 대화 형식의 항목 하나 생성 및 system 메시지 추가
        message_list.append({"messages": [system_message]})

        # 사용자의 입력 메시지 추가 (예: 문장 혹은 문서)
        user_message = {
            "role": "user",
            "content": row['example']
        }
        message_list[-1]["messages"].append(user_message)

        # 모델의 정답 메시지 (예: 분류 결과) 추가
        assistant_message = {
            "role": 'assistant',
            "content": row['label']
        }
        message_list[-1]["messages"].append(assistant_message)

    # 생성된 대화 데이터를 JSONL 형식으로 저장
    with open(filename, "w") as json_file:
        for message in message_list:
            json.dump(message, json_file)  # JSON 문자열로 변환 후 저장
            json_file.write("\n")          # 줄바꿈으로 구분 (JSONL 포맷)

# 프롬프트
system_content =  f"given the following text: find the category in: {categories} that is most closely associated with it. Return only the category name only in following format" 
```
- 별로 특별할 거 없는....그냥 평범한 프롬프트로 카테고리 분류를 합니다.

```python
def zero_shot_model(data, model_id):
    pred = []
    for row in data["example"]:
        completion = openai.ChatCompletion.create(
            model= model_id,
            messages=[
                {"role": "system", "content": system_content },
                {"role": "user", "content": row }
            ])
        
        print(f'text: {row}')
        print(completion.choices[0].message.content)
        pred.append(completion.choices[0].message.content)

    pred_df = pd.DataFrame({'example': data["example"], 'label' : data['label'], 'few-shot predictions' : pred })
    return pred_df

def fine_tune_model(model_id, num_label, pred_df):
    incorrection_pred_df = pred_df[pred_df['example'] != pred_df['label']][:num_label]
    filename = f'ft_increment_{num_label}.jsonl'
    text_to_openai_json(incorrection_pred_df, filename)
    loader = openai.File.create(file=open(filename, "rb"), purpose='fine-tune')
    fine_tuning_job = openai.FineTuningJob.create(training_file=loader.id, model="gpt-3.5-turbo")
    return fine_tuning_job.id

def ft_accuracy(data, model_id):
    pred = []
    for row in data["example"]:
        completion = openai.ChatCompletion.create(
            model= model_id,
            messages=[
                {"role": "system", "content": system_content },
                {"role": "user", "content": row }
            ])
        
        print(f'example: {row}')
        print(completion.choices[0].message.content)
        pred.append(completion.choices[0].message.content)

    accuracy = accuracy_score(data['label'], pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    precision, recall, f1, _ = precision_recall_fscore_support(data['label'], pred, average='macro', zero_division=1)
    return accuracy, precision, recall, f1, pred
``` 
- 제로샷 모델로 예측을 한 뒤에
- 예측이 틀린 샘플로 파인튜닝용 데이터를 생성 - 학습 하고
- 그걸로 평가하는 식의 구조.
- 어...너무...단순해서 별로 설명할 게 없음...
- entropy는 왜 안구하는걸까? 못하나?

##### BERT
```python
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", max_length=6000)
train_encodings = tokenizer(train_df['example'].to_list(), truncation=True, padding=True)
test_encodings = tokenizer(test_df['example'].to_list(), truncation=True, padding=True)
train_labels = train_df.label.to_list()
test_labels = test_df.label.to_list()
train_df['label'].unique()
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(total_df['label'].unique()), id2label = id_to_label, label2id = label_to_id)

training_args = TrainingArguments(
    output_dir='./BERTModel2',
    do_train=True,
    do_eval=True,
    num_train_epochs=3,  # Reduced number of epochs.
    per_device_train_batch_size=5,  # Reduced batch size for training.
    per_device_eval_batch_size=20,  # Reduced batch size for evaluation.
    warmup_steps=100,
    weight_decay=0.01,
    logging_strategy='steps',
    logging_dir='./multi-class-logs',
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    fp16=True,  # Enable mixed precision training.
)

class DataLoader(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        """
          This construct a dict that is (index position) to encoding pairs.
          Where the Encoding becomes tensor(Encoding), which is an requirements
          for training the model
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        """
        Returns the number of data items in the dataset.

        """
        
        return len(self.labels)

def compute_metrics(pred):   
    # Extract true labels from the input object
    labels = pred.label_ids

    # Obtain predicted class labels by finding the column index with the maximum probability
    preds = pred.predictions.argmax(-1)

    # Compute macro precision, recall, and F1 score using sklearn's precision_recall_fscore_support function
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro',zero_division=1)

    # Calculate the accuracy score using sklearn's accuracy_score function
    acc = accuracy_score(labels, preds)

    # Return the computed metrics as a dictionary
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }


train_dataloader = DataLoader(train_encodings,train_labels)
test_dataloader = DataLoader(test_encodings,test_labels)
trainer = Trainer(
    #the pre-trained bert model that will be fine-tuned
    model=model,
    #training arguments that we defined above
    args=training_args,
    train_dataset= train_dataloader,
    eval_dataset = test_dataloader,
    compute_metrics= compute_metrics
)
predictions = trainer.predict(test_dataloader)
predicted_labels = predictions.predictions.argmax(axis=1)
accuracy = accuracy_score(test_df['label'].to_list(),predicted_labels)
print('accuracy {}% '.format(np.round(accuracy * 100)))
```
- 아주아주 기본적인 BERT 구조.
- 진짜 딱히 특별할 게 없음.

```python
def entropy_for_each_row(class_probabilities):
    """ Calculate entropy for each row in the array """
    return -tf.reduce_sum(class_probabilities * tf.math.log(class_probabilities),axis=1)

def predict_and_calculate_entropy(data):
    
    ''' 
    Predict and Calculate Entropy
    
    This function makes predictions using a pre-trained BERT model, calculates the entropy (uncertainty) of these predictions, 
    and creates a DataFrame containing relevant information.
    
    Args:
        data (DataFrame): A Pandas DataFrame containing text data and associated labels. The DataFrame should have 
        columns 'example' for text data and 'label' for labels.

    Returns:
        final_df (DataFrame): A Pandas DataFrame containing the following columns:
            - 'example': The original text data.
            - 'predicted_Label': The predicted class labels based on the model's predictions.
            - 'predicted_Probability': The maximum predicted probability for each instance.
            - 'Entropy': The calculated entropy (uncertainty) for each instance.
            - 'label': The original labels from the input data.
    ''' 

    
            
    # Sample the Data 
    data_encodings = tokenizer(data['example'].to_list(), truncation=True, padding=True)
    dataloader = DataLoader(data_encodings, data.label.to_list())

    # Make predictions with class_probabilities and calculate entropy (uncertainty) 
    predictions = trainer.predict(dataloader)
    prediction_probabilities = tf.constant(predictions.predictions)

    # Predicted Labels 
    predicted_labels = predictions.predictions.argmax(axis=1)

    
    # Prediction probabilities, returning the highest probability for each instance
    prediction_probabilities_max = np.amax(prediction_probabilities, axis=1)

    # Calculate entropy for each instance
    entropies = entropy_for_each_row(tf.nn.softmax(prediction_probabilities))

    entropy_df = pd.DataFrame(
        {'example' : data['example'].to_list(),
         'predicted_Label': predicted_labels,
         'predicted_Probability': prediction_probabilities_max,
         'Entropy': entropies},
        index=data.index
    )

    final_df = pd.concat([data['label'], entropy_df], axis=1)

    return final_df.sort_values(by=['Entropy'],ascending=False)

# Initialize empty lists to store metrics for each iteration
accuracy_list = []
precision_list = []
recall_list = []
loss_list = []
x_labels = []
n = 0

for iteration in range(15):
    n += 10
    initial_labeled_encoding = tokenizer(hundreds_rows_pred[0:n]['example'].to_list(), truncation=True, padding=True)
    initial_labeled_labels = hundreds_rows_pred[0:n].label.to_list()
    initial_labeled_dataloader = DataLoader(initial_labeled_encoding,initial_labeled_labels)

   
    trainer = Trainer(
        #the pre-trained bert model that will be fine-tuned
        model=model,
        #training arguments that we defined above
        args=training_args,
        train_dataset= initial_labeled_dataloader,
        eval_dataset = test_dataloader,
        compute_metrics= compute_metrics
    )

    trainer.train()

    metrics = trainer.evaluate()

    print('eval_Accuracy :',metrics['eval_Accuracy'])

    accuracy_list.append(metrics['eval_Accuracy'])
    precision_list.append(metrics['eval_Precision'])
    recall_list.append(metrics['eval_Recall'])
    loss_list.append(metrics['eval_loss'])
    x_labels.append(n)
```
- 엔트로피는 일반적인 정보이론에서의 엔트로피를 사용하는 것을 알 수 있음. 
- 행별로 계산해 불확실성 수준을 반환
- 확률 분포가 균일 할수록, 즉 예측이 하나로 확 쏠리지 않을수록 엔트로피가 높아짐.
- 최종적으로 엔트로피가 높은 순으로 반환하여 10개씩 추가하면서 학습하게 구성됨.

##### 느낀 점
1. BERT / SETFIT은 그림 6,7의 결과만 봐도 알 수 있듯이 안타까운 성능임. 이 과정에 쓸만한건 GPT류의 생성형 AI뿐으로 보이나, 애초에 BERT를 저런데다 쓰는게 맞나? 더 많이 학습시켰어야지 하는 생각이 듦.
2. 문제의 난이도와 관계 없이 GPT가 가장 좋으나, 성능 향상이 꼭 드라마틱하진 않은거 같음. 약간 널뛰는 경향이 있음.
3. label이 더더 다양한 경우에도 유의미할까 싶기도 하네.
4. 각각을 agent에게 시키면 안될까? agent를 judge로 써서?


