import tensorflow as tf
import cv2
import glob
import numpy as np
pb_path = 'landmark.pb' # pb模型

sess = tf.Session()
with sess.as_default():
    with tf.gfile.FastGFile(pb_path, 'rb') as f:
        graph_def = sess.graph_def
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def,name='')

# 测试图片
im_list = glob.glob('images/*')

landmark = sess.graph.get_tensor_by_name('fully_connected_11/Relu:0')
for im_url in im_list:
    im_data = cv2.imread(im_url)
    im_data = cv2.resize(im_data,(128,128))
    pred = sess.run(landmark,{'Placeholder:0':np.expand_dims(im_data,0)}) # 图片给网络  -- 增加一维
    print(pred)
    # 反归一化
    pred = pred[0]
    for i in range(0,136,2):
        cv2.circle(im_data,(int(pred[i] * 128),int(pred[i+1] * 128)),2,(0,255,0),2)
    name = im_url.split('\\')[-1]
    cv2.imwrite('./test_result/%s' % name,im_data)
    cv2.imshow('11',im_data)
    cv2.waitKey(200)

