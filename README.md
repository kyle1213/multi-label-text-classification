# multi-label-text-classification

데이터
- 커뮤니티나 카페와 같은 인터넷에서 특정 키워드가 포함된 텍스트 데이터 모두 수집(런닝화, 아웃백 신메뉴, bhc 신메뉴 등) / 각각 약 1000개
- 특정 클래스를 가진 데이터가 많았고(클래스간 불균형), multi-label task이다보니 클래스 내 불균형과 클래스 집합 불균형이 있었다.
- 클래스는 30개 이상

데이터 증강
- fine tuning이어도 클래스 수에 비해 데이터가 적은 것 같아 Easy Data Augmentaiton을 사용했다.
- random deletion, random swap 사용하여 각 방법마다 2번 -> 데이터 약 4배로 증강 -> 성능 개선이 있었다
- 트위터에서 특정 클래스를 잘 나타내는 키워드로 데이터를 약한?(noisy? imperfect?) 라벨링하며 데이터를 추가 수집함 -> 라벨링의 시간을 더 투자하지 못해, 불완전한 라벨링인 대신 데이터 양을 늘리겠다는 생각 -> 성능 개선이 있었다
- gpt를 활용한 라벨링도 활용

딥러닝 모델
1. koBert
- pre-trained kobert에 classifier layer를 추가하여 사용
- 무난한 base 느낌의 결과가 나옴
  
2. koBigBird
- 원래 의도는, 최대 512 길이를 받는 bert의 한계를 넘고 더 긴 데이터를 보려고 bigbird를 사용했으나, 그럼에도 개발환경인 colab의 gpu 메모리 문제를 해결할 수 없었다.
- 그 결과 웃기게도 bert와 같은 512 길이, 즉 sparse attention 없는 bigbird를 사용하게 되었다.
- 더 웃긴건 pre-train 데이터의 차이 때문에 bert보다 성능이 좋게 나옴
  
3. kcelectra
- 인터넷 글 특성상 korean commentary를 데이터로 pre-train한 kcelectra가 좋은 성능을 나타낼 것 같아서 사용하게 됨
- 실제로 train, val의 성능이 올랐지만, 트위터에서 수집한 데이터가 없는 test set에서는 성능이 다른 모델에 비해 떨어졌다.

학습 조기 종료(early stopping)
- val loss가 특정 patience 만큼 상승하지 않는다면 학습을 조기 종료하였다

learning rate scheduler
- cosine scheduler

Self-Ensemble, Self-Distilation
- single 모델의 time step 간 self-ensemble 모델 사용
- 해당 모델을 teacher로 하여 output logits을 knowledge distillation에 사용 

details
- batch size: 6
- epochs: 100 -> early stopping때문에 보통 6즈음에서 끝남
- learning rate: 5e-5
- 5-fold 학습을 함
- lr가 계속 변하여 weight decay가 잘 반영되지 않는 adam을 개선한 adamw를 optimizer로 사용
- loss : Asymmetric Loss for Imbalance data

# results  
bhc 데이터의 1번째 fold의 학습 정확도 그래프  
![](https://github.com/kyle1213/multi-label-text-classification/blob/main/imgs/bhc_bb_acc_fold0.png)  


bhc 데이터의 1번째 fold의 학습 loss 그래프  
![](https://github.com/kyle1213/multi-label-text-classification/blob/main/imgs/bhc_bb_loss_fold0.png)  


bhc 데이터의 f1 score  
![](https://github.com/kyle1213/multi-label-text-classification/blob/main/imgs/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202024-05-27%20140414.png)  


accuracy  
![](https://github.com/kyle1213/multi-label-text-classification/blob/main/imgs/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202024-05-27%20140514.png)   


class 별 precision, recall, f1 score와 평균  
![](https://github.com/kyle1213/multi-label-text-classification/blob/main/imgs/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202024-05-27%20140546.png)  
![](https://github.com/kyle1213/multi-label-text-classification/blob/main/imgs/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202024-05-27%20140554.png)  


class 별 정확도  
![](https://github.com/kyle1213/multi-label-text-classification/blob/main/imgs/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202024-05-27%20140627.png)  



# 메모

few dataset
- https://arxiv.org/abs/1901.11196

data imbalance
- https://www.sciencedirect.com/science/article/pii/S0031320321001527
- https://link.springer.com/chapter/10.1007/978-3-642-41822-8_42
- https://www.sciencedirect.com/science/article/pii/S0031320312001471?via%3Dihub
- https://www.sciencedirect.com/science/article/pii/S0167865511003734?via%3Dihub
- https://ieeexplore.ieee.org/document/8672066
- focal loss, asymmetric loss - CE는 틀렸을 때 페널티를 주게 학습해서 맞출 때 이득이 없음. 불균형을 위한 weighted loss는 데이터 불균형을 어느정도 해소하지만 easy, hard samples에서 의미가 없음. focal loss는 easy samples(high pt)에 페널티를 줘서 loss를 줄여줌. 물론 hard samples(low pt)에도 패널티가 약간 생김. ASL은 rare positive samples와 negative samples를 구별해서 패널티를 주기로 함(rare에 적은 패널티). 그럼에도 압도적으로 negative가 많은 경우를 위해 PS(probability shifting)을 사용해서, easy negative sampes(마진을 넘는 pt)의 경우 아예 학습을 안하게 함(loss=0)
- https://arxiv.org/abs/2207.07080
- metric: UWA
  
model over-confidence(uncertainty)
- 베이지안 뉴럴넷 - 학습할 가중치를 non-deterministic한 통계로 표현하기 -> 적분으로 계산이 필요한데 근사 방법이 있다 + 파라미터가 너무 많아서 적용이 어렵다
- 베이지안 optimization - given data를 통해 미지의 함수(블랙박스 함수)를 예측 by 불확실성을 줄이는 데이터 포인트 고르기 등
- gaussian process - 데이터 포인트(x, y)로 함수평균, new point x와 x, 커널 함수로 공분산을 구하고 posterior 평균 분산을 구해서 함수예측(함수의 특정 지점의 통계량을 구할 수 있음)?
- https://openreview.net/forum?id=enKhMfthDFS
- https://arxiv.org/pdf/2210.10160.pdf
- https://www.tensorflow.org/tutorials/understanding/sngp?hl=ko
- bayesian NN, rule etc.

data capacity(long text issue)
- linformer
- 입력 길이 n이라면 트랜스포머(셀프어텐션)의 시간, 공간 복잡도는 O(N^2)인데, 행렬 연산을 low rank로 바꿔서 O(N)인 트랜스포머 구현. BUT no pre-trained 모델
- https://arxiv.org/abs/2006.04768
- bigbird
- sparse attention으로 메모리, 연산량 효율 증가

 
 

problem transformations
- label power set
 - 라벨 수가 너무 많음 -> power set은 더 많을듯(물론 없는 라벨셋도 존재하겠지만 실제 데이터 확인결과 너무 많았음)
- binary relevance
  - 가장 쉽고 이상적인듯, 물론 라벨 수가 많아서(약 30개) 불균형이 심해짐
- classifier chain
  - 사람이 글을 쓰고 그 글의 속성에 상관관계는 없다고 가정함 -> 당연히 있을 수 있음 추후 프로젝트 기간 내에 여유가 있으면 진행할 예정
- multi-label multi-class classification
  - '없음'에 해당하는 클래스도 각 라벨마다 만들어야 하는데, 사실 BR이랑 다를게 없다고 생각함(CE랑 BCE랑 같다고 생각). 오히려 ASL적용이 어려워져서 안좋을거 같음

data
- 양이 적음(total 1,000개)
  - SMOTE -> 데이터가 없거나 매우 적은(10개 이하) 라벨이 많아 적용 힘듦
  - EDA -> 여전히 없는 라벨이 있어서 적용이 어려움 
- 불균형
  - 불균형 지수(IRLbl)가 사실 무한이 나옴 -> 라벨이 없는 데이터 때문 -> 라벨이 없는 경우 1개만 가지고 있다고 가정하고 계산하니 수치가 폴드에 따라 180~500 나옴
  - EDA를 전체 데이터에 적용 / 많은 데이터를 가진 라벨을 제외하고 적용 / 적은 데이터를 가진 라벨에 적용
    - IRLbl이 줄어들긴 함 -> 의미가 있을진 모르겠음(160)
    - 어떻게 적용하냐에 따라 클래스간 1의 불균형이 줄어들긴 하지만 클래스 내(0과 1)의 불균형이 심해져 성능에 악영향을 주었다. -> 클래스 내 불균형이 클래스간 불균형보다 심함 -> 모든 속성을 독립적으로 binary classification해서 그런듯
    - 3가지를 적용해서 실험 결과 EDA가 없는거보다 못한 경우도 있지만 보통 val set에서 성능 향상이 있었다.
  - 하지만 EDA는 문장 자체는 형식이 비슷해서 그런지 EDA를 많이 적용한 데이터일수록 train set에 overfit되서 그런지 val loss가 높게 올랐다 -> EDA없이 val loss = 7 / EDA로 train set이 많아질수록 val loss = 10, 13, 19 . . .
  - val loss가 높아지면서 수렴하지만 val acc도 높아지며 수렴함 -> train set에 오버핏되긴 하나, val set에서 low confidence로 정답을 맞추기는 함 -> 그게 아니라 high confidence로 틀리는듯 -> imbalance때문에 정확도가 높아진거임


model
- 초반엔 kobert, maxlen=512로 진행
  - 인터넷 글 특성상 512 이상의 데이터도 꽤 있음을 확인함 -> bert는 버리겠지만
- 후에 kobigbird 사용
  - kobigbird 사전학습 모델을 가져왔지만, 코랩 환경상 1024로 테스트가 힘들었음 -> 512로 했는데 그럼 사실 의미가 없다
  - 하지만 사전학습 모델이 bert는 사전이 1만개 였는데, bigbird는 3만개였음 -> 그래서 그런가 성능(acc, f1 score)은 bigbird가 좋았다(loss는 bert가 better)
- 길이가 짧은 데이터로 구성된 도메인의 경우 bert를 사용해도 무방할 것이고, gpu 메모리를 늘려 bigbird 1024로 사용해도 될 것 같다


loss
- 기본 loss / weighted loss / asymmetric loss 적용
- 기본의 경우 적은 데이터를 가진 라벨들의 정확도는 0프로로, 전반적인 정확도나 다른 metric에 비해 macro f1 score나 1들의 acc가 저조했음
- weighted loss의 경우 데이터 불균형이 심각해 weight의 수치가 극단적이라 오히려 성능에 안좋았음
- asymmetric loss의 경우 성능 개선이 상당히 좋았다
  - 정확도나 micro, weighted f1 score는 10%p 이상 오르고
  - macro f1 score는 두배 오름(0.06 -> 0.15)
  - 특히 불균형이 심한(적은 수의 데이터를 가진 라벨)에 대한 정확도가 생기기 시작 -> 예를 들어, 학습 데이터 700개 중 3개밖에 없던 라벨의 데이터를 val set에서 맞추기 시작


metric
- 불균형이 심해 하나의 metric에만 의존하기 어려워 여러 metric을 보며 모델 성능을 책정했지만 어려웠다 -> trade off가 발생하기도 했기 때문
- 정확도 : 한 데이터에서 모든 라벨을 맞춘 경우
  - 하지만 불균형 데이터라 정확도만 보기엔 아쉬울 수 있다고 생각함
- f1 score : macro, micro, weighted 사용
  - micro, weighted, acc는 비슷하게 나왔음 항상
  - 하지만 macro는 각 라벨의 f1 score의 평균이다보니, 데이터가 적은 라벨의 정확도에 영향을 받아 상당히 낮게 나옴 항상 -> macro f1을 높이는 걸 주 목표로 삼음(그렇다고 다른 metric을 무시하진 않음)
- class acc : 1들의 정확도, 0들의 정확도
  - 라벨간 정확도가 아닌 1들의 정확도와 0들의 정확도. 1들의 정확도가 높아지면 macro f1도 같이 올라서 주의깊게 봤음
- confusion metrix
  - 클래스 별 지표를 보기 좋았다. 물론 클래스가 너무 많아 힘들었음
- 참고 지표가 너무 많고 적용 방법론에 따라 trade off도 많고 변화량도 미묘하며 데이터 양은 적고 불균형도 심하니 성능 개선도 힘들었다
 
학습 중
- 데이터 fold에 따라 val set 정확도, loss가 엄청 높기도 함
- 오버피팅 문제
  - bert는 오버핏이 거의 안 일어나지만 대부분의 metric에서 bigbird보다 저조함(loss 제외)
  - bigbird의 경우 오버핏이 항상 심하게 일어나지만 metric이 우수함(loss 제외)
  - bigbird가 오버핏이 일어나기 전인 낮은  epochs에서 1의 정확도가 70퍼센트나 됨. 물론 0의 정확도는 95퍼센트로, 대다수가 0의 라벨인 데이터인 현 상황에서 정확도는 10퍼센트로 떨어짐
  - 틀려도 1을 진짜 많이 맞추려면 오버핏이 안 된 모델 사용. 전반적인 metric이 우선이면 그냥 오버핏시켜야 함. 오버핏 해도 ASL 덕분에 적은 라벨을 못맞추는건 아님.
  - non-오버핏 모델은, low confidence로 문제를 해결하는듯 -> 전반적인 acc는 떨어지지만 1의 acc가 상승. 하지만 1에 대해 똑똑해지는게 아니라, 1을 찍는 느낌 -> 0을 많이 틀리니까, 심지어 1을 맞추는게 confident하지도 않음

- 사실 loss 설계를 잘못함.. -> 클래스별 loss의 합을 배치 단위로 평균내야 하는데, 그냥 다 합쳤음 -> loss가 엄청 컸던 이유 -> 사실 다 합치는게 맞음. torch의 adam은 loss sum이나 mean이나 같은 결과를 만듬 -> 그렇기에 아래와 같은 분석을 했던 것
- 정상적으로 구현 후 loss는 기존에 7~17에서 시작하던게 1.6~2.x로 바뀜
- 이렇게되니 오버핏은 쉽게 안되지만 학습 자체가 오래 걸려짐 -> 파인튜닝인데 이렇게 오래 학습시켜야되나 할 정도(데이터가 적어서 그런듯)(50에폭시는 해야 함 적어도)
- 그래도 50에폭시정도 하면 기존 sota정도 나옴
- 여기서 느낀점: 이전에 이상하게 진행하던 경우 loss가 몇배는 높게되서 학습이 가속되서 그런지 수렴이 훨씬 빨랐다. 신기한건 local optimum에 그래도 가긴 했다는 것.

