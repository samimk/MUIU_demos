import os
#os.environ["KERAS_BACKEND"] = "jax"
#os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["KERAS_BACKEND"] = "torch"

def demonstrate_backends():
    """
    Demonstrira kako Keras 3.0 radi sa različitim backend-ima
    """

    # Ova funkcija treba biti u ZASEBNIM skriptama!
    # Ne može se mijenjati backend u toku izvršavanja

    print("="*50)
    print(f"Keras 3.0 Backend Demo")
    print("="*50)

    import keras
    from keras import layers, models
    import numpy as np

    # Provjera backend-a
    backend = keras.backend.backend()
    print(f"\n✓ Aktivni backend: {backend}")

    # Kreiranje modela (ISTI KOD za sve backend-e!)
    model = models.Sequential([
        layers.Input(shape=(10,)),
        layers.Dense(20, activation='relu', name='hidden'),
        layers.Dense(1, activation='sigmoid', name='output')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print(f"\n✓ Model uspješno kreiran")
    model.summary()

    # Test podataka
    X_test = np.random.randn(5, 10)
    y_pred = model.predict(X_test, verbose=0)

    print(f"\n✓ Predikcija uspješna")
    print(f"Shape predikcije: {y_pred.shape}")
    print(f"Tip podataka: {type(y_pred)}")

    # Backend-specifične informacije
    if backend == "tensorflow":
        import tensorflow as tf
        print(f"\n[TensorFlow Info]")
        print(f"TF verzija: {tf.__version__}")
        print(f"GPU dostupan: {len(tf.config.list_physical_devices('GPU')) > 0}")

    elif backend == "jax":
        import jax
        print(f"\n[JAX Info]")
        print(f"JAX verzija: {jax.__version__}")
        print(f"JAX devices: {jax.devices()}")

    elif backend == "torch":
        import torch
        print(f"\n[PyTorch Info]")
        print(f"PyTorch verzija: {torch.__version__}")
        print(f"CUDA dostupan: {torch.cuda.is_available()}")

    print("\n" + "="*50)

# Pokretanje
demonstrate_backends()
