import tensorflow as tf

from object_detection.builders import model_builder
print("GPUs disponibles :", tf.config.list_physical_devices('GPU'))

# Création d'un tenseur simple pour activer le GPU
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print("Résultat du produit matriciel :", c)