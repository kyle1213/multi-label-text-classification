# multi-label-text-classification
multi-label text classification

few dataset
- https://arxiv.org/abs/1901.11196
- EDA로 적은 데이터만 증강(frequent 라벨이 있는 데이터는 증강 x) -> 증강해도 데이터가 많이 안생김

data imbalance
- https://www.sciencedirect.com/science/article/pii/S0031320321001527
- BERT사용할 때 임베딩 레이어도 학습을 하는데, SMOTE같은 upscaling은 사용하기 어려워 보임. -> 임베딩 레이어를 지나고 SMOTE하기?
- 그리고 데이터가 없거나 극소수인 라벨은 SMOTE 할 수 없음
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
- gaussian process - 데이터 포인트(x, y)로 함수평균, new point x와 x, 커널 함수로 공분산을 구하고 posterior 평균 분산을 구해서 함수예측?
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

pre-train pre-trained model(PLM)
- 의미 없을듯

 
 

 -----

data
- 양이 적음(total 1,000개)
  - SMOTE -> 데이터가 없거나 매우 적은(10개 이하) 라벨이 많아 적용 힘듦
  - EDA -> 여전히 없는 라벨이 있어서 적용이 어려움 
- 불균형
  - 불균형 지수(IRLbl)가 사실 무한이 나옴 -> 라벨이 없는 데이터 때문 -> 라벨이 없는 경우 1개만 가지고 있다고 가정하고 계산하니 수치가 187 나옴
  - EDA를 전체 데이터에 적용 / 많은 데이터를 가진 라벨을 제외하고 적용 / 적은 데이터를 가진 라벨에 적용
    - IRLbl이 줄어들긴 함 -> 의미가 있을진 모르겠음(160)
    - 어떻게 적용하냐에 따라 클래스간 1의 불균형이 줄어들긴 하지만 클래스 내(0과 1)의 불균형이 심해져 성능에 악영향을 주었다. -> 클래스 내 불균형이 클래스간 불균형보다 심함 -> 모든 속성을 독립적으로 binary classification해서 그런듯
    - 3가지를 적용해서 실험 결과 EDA가 없는거보다 못한 경우도 있지만 보통 val set에서 성능 향상이 있었다.
  - 하지만 EDA는 문장 자체는 형식이 비슷해서 그런지 EDA를 많이 적용한 데이터일 수록 val loss가 높게 올랐다 -> EDA없이 val loss = 7 / EDA로 train set이 많아질수록 val loss = 10, 13, 19 . . .
  - val loss가 높아지면서 수렴하지만 val acc도 높아지며 수렴함 -> train set에 오버핏되긴 하나, val set에서 low confidence로 정답을 맞추기는 함 
- 
