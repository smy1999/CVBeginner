import tensorflow as tf
import matplotlib.pylab as plt
import tensorflow_hub as hub
import logging
import numpy as np
import PIL.Image as Image

"""判断是否是军装"""

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# 从网址导入Transfer Learning模型MobileNet, MobileNet已在ImageNet训练完成, ImageNet有1000个不同的输出种类
CLASSIFIER_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
IMAGE_RES = 224  # 图像尺寸与MobileNet吻合

model = tf.keras.Sequential([
    hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
])

# 读取图片
grace_hopper = tf.keras.utils.get_file('image.jpg', 'https://storage.googleapis.com/download.tensorflow.org'
                                                    '/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize((IMAGE_RES, IMAGE_RES))

# Normalization, grace_hopper.shape = (224, 224, 3)
grace_hopper = np.array(grace_hopper) / 255.0

# 模型批处理, 故增加维度 result.shape = (1, 1001)
result = model.predict(grace_hopper[np.newaxis, ...])

# 预测, 得出最有可能的索引值
predicted_class = np.argmax(result[0], axis=-1)
print("argmax = " + str(predicted_class))

# Download ImageNet labels to check our prediction
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())
plt.show()
