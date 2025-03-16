import tensorflow as tf
import time




from tensorflow.keras import layers, models

# Test simple sur GPU
def test_gpu():
    if tf.config.list_physical_devices('GPU'):
        print("GPU détecté. Test en cours...")
    else:
        print("Aucun GPU détecté. Vérifiez votre configuration.")
        return

    # Test de calcul
    with tf.device('/GPU:0'):

        # Définir un modèle simple
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax'),
        ])

        # Compiler le modèle
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Dummy data
        import numpy as np
        x = np.random.rand(10, 128, 128, 3)
        y = np.random.randint(10, size=(10,))

        # Entraîner le modèle
        model.fit(x, y, epochs=1)


test_gpu()
