- tf.image.decode_jpeg(img, channels=depth) : jpeg decoding
- tf.cast(asdf, dtype=tf.float32) -> float으로 cast 
- tf.expand_dims(img, 0) -> 차원 증가 

- tf.stack -> stack 쌓는것 : x=[1], y=[2] 면 tf.stack(x,y) -> [[1],[2]]가 된다 (tf.pack 의 업그레이드)
- tf.image.resize_bilinear -> bilinear interpolation을 사용해서 resizing

- tf.squeeze -> 크기(값)가 1인 것 삭제 (압축)  
    x = [1,2,1,3,1,1] 이고 tf.squeeze(x) 하면 [2,3]이 된다
    인덱스로 삭제할라면 tf.squeeze(x, [2,4]) -> 하면 2,4 인덱스중 1인 것이 삭제짐

- tf.random_crop
