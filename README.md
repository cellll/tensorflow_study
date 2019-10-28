# basic

- cross-entropy : 2개의 확률 분포의 차이(거리) -> cost(loss) function으로 사용 (기대값 <-> 실제값) -> 학습 과정이 잘 되고 있는지 학인할 수 있음 (계속 줄어야함)

- training accuracy는 높은데 validation accuracy 가 낮으면 -> Overfitting

- transfer learning : pre-trained 모델에서 마지막 부분 (Fully-connected)를 제외하고 나머지 부분 + new 데이터셋 가지고 retrain 하는것 (fine-tuning)
  -> pre-trained 모델의 weight 를 초기값으로 설정하고 학습하면 더 좋은 결과를 얻을 수 있다
  
- precision : 정확성

- training / validation / test 3개로 나누는 이유 : overfitting 방지 -> 학습 과정에서 학습용 이미지에서 cache 하기도 하기 때문에 그 학습 과정에 쓴 이미지에 대해서는 성능이 좋음. 다른 이미지를 넣으면 성능 하락 -> 그래서 3개로 나눠서 대체로 80% training / 10%는 트레이닝 과정 중에 validation / 10은 test 


# Tensorflow

- multiple gpu : because no supported kernel for gpu devices is available 나오면 

         config = tf.ConfigProto(allow_soft_placement = True)
         sess = tf.Session(config = config)
    
              for i in xrange(gpu개수):
                  with tf.device('/gpu:%d' % i)
       


 - saver : 변수선언 다 된 후에(세션 전) tf.train.Saver(~) <- max_to_keep=None 하면 스텝 다 저장함(default=5)
        saver.save(sess ~~~)
        
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)
        saver.save(sess, ckpt_path, global_step=step)


# python 

- 파이썬 gc.collect() 해도 해제 안되는 것은 glibc 에서 C에서 malloc -> jemalloc 라이브러리
