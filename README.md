# multi-label-text-classification
multi-label text classification

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
- https://openreview.net/forum?id=enKhMfthDFS
- https://arxiv.org/pdf/2210.10160.pdf
- https://www.tensorflow.org/tutorials/understanding/sngp?hl=ko
- bayesian NN, rule etc.

data capacity(long text issue)
- linformer
- https://arxiv.org/abs/2006.04768

pre-train pre-trained model(PLM)
