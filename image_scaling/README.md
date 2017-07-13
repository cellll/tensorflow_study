hi

- tf.image.decode_jpeg(asdf, channels=depth) : jpeg으로 ~~
- tf.cast(asdf, dtype=tf.float32) -> float으로 cast 
- tf.expand_dims(asdf, 0) -> 차원 증가 

- tf.stack -> stack 쌓는거 : x=[1], y=[2] 면 tf.stack(x,y) -> [[1],[2]]가 된다 (tf.pack 이 업글됨)
- tf.image.resize_bilinear -> bilinear interpolation을 사용해서 resizing 한대는데 먼지모름

- tf.squeeze -> 크기가 1인 멀 삭제한대는데 x = [1,2,1,3,1,1] 이고 tf.squeeze(x) 하면 [2,3]이 된다
인덱스로 삭제할라면 tf.squeeze(x, [2,4]) -> 하면 2,4 인덱스중 1인게 사라짐

- tf.random_crop 이런거들 거의다있음


