# H I

- batch size : 1 step에 image 몇개 쓸건지

- cross-entropy : 2개의 확률 분포의 차이(거리) -> cost(loss) function으로 사용 (기대값 <-> 실제값) -> 학습 과정이 잘 되고 있는지 학인할 수 있음 (계속 줄어드는지)

- training accuracy는 높은데 validation accuracy 가 낮으면 -> Overfitting

- transfer learning : pre-trained 모델에서 마지막 부분 (Fully-connected)를 제외하고 나머지 부분 + new 데이터셋 가지고 retrain 하는거 (fine-tuning)
  -> pre-trained 모델의 weight 를 초기값으로 설정하고 학습하면 더 굿
  
- precision : 정확성

- training / validation / test 3개로 나누는 이유 : overfitting 방지 -> 학습 과정에서 막 학습용 이미지에서 뭔가를 기억(cache) 하기도 하기 때문에 그 학습 과정에 쓴 이미지에 대해서는 성능이 엄청 좋음 근데 따른거 넣으면 성능 똥 -> 그래서 3개로 나눠서 대체로 80퍼 training / 10퍼는 트레이닝 과정 중에 validation / 10은 test로 


# Tensorflow

- multiple gpu : because no supported kernel for gpu devices is available 나오면 

         config = tf.ConfigProto(allow_soft_placement = True)
         sess = tf.Session(config = config)
    
              for i in xrange(gpu개수):
                  with tf.device('/gpu:%d' % i)
       


 - saver : 변수선언 다 된 후에(세션 전에) tf.train.Saver(~) <- max_to_keep=None 하면 스텝 다 저장함(default=5)
        saver.save(sess ~~~)
        
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)
        saver.save(sess, ckpt_path, global_step=step)


# Slim

- TF-Slim : lightweight high-level API -> 모델을 defining, training, evaluating -> pretrained weight 갖고 fine tuning 하는거를 이거 쓰면 되는듯
