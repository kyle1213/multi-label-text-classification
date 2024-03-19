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

pre-train pre-trained model(PLM)
- 의미 없을듯
