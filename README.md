# H I

- batch size : 1 step에 image 몇개 쓸건지

- bottleneck : (저거만인지는 모름) 마지막에서 2번째 layer -> 실제 classification 하는거를 여기서 한다고함( 머 캐싱 얘기도 있는데 이건 먼지 모름)
  -> bottleneck -> final layer 여기서 prediction을 하고 실제 label과 비교하고 back propagation을 통해서 weight를 업데이트

- cross-entropy : 2개의 확률 분포의 차이(거리) -> cost(loss) function으로 사용 (기대값 <-> 실제값) -> 학습 과정이 잘 되고 있는지 학인할 수 있음 (계속 줄어드는지)

- training accuracy는 높은데 validation accuracy 가 낮으면 -> Overfitting

- transfer learning : pre-trained 모델에서 마지막 부분 (Fully-connected)를 제외하고 나머지 부분 + new 데이터셋 가지고 retrain 하는거 (fine-tuning)
  -> pre-trained 모델의 weight 를 초기값으로 설정하고 학습하면 더 굿
  
- precision : 정확성

- training / validation / test 3개로 나누는 이유 : overfitting 방지 -> 학습 과정에서 막 학습용 이미지에서 뭔가를 기억(cache) 하기도 하기 때문에 그 학습 과정에 쓴 이미지에 대해서는 성능이 엄청 좋음 근데 따른거 넣으면 성능 똥 -> 그래서 3개로 나눠서 한다



