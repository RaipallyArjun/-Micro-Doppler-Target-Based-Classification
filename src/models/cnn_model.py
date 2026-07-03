from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam


def create_cnn_model():
    """
    CNN model for Bird vs Drone classification
    """

    model = Sequential(name="MicroDopplerCNN")

    # -------------------------------------------------
    # Input Layer
    # -------------------------------------------------

    model.add(Input(shape=(128, 128, 1)))

    # -------------------------------------------------
    # Convolution Block 1
    # -------------------------------------------------

    model.add(Conv2D(
        filters=16,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ))

    # momentum lowered from default 0.99 -> 0.8 so running stats
    # converge fast enough on a small dataset with few steps/epoch
    model.add(BatchNormalization(momentum=0.8))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    # -------------------------------------------------
    # Convolution Block 2
    # -------------------------------------------------

    model.add(Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ))

    model.add(BatchNormalization(momentum=0.8))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    # -------------------------------------------------
    # Convolution Block 3
    # -------------------------------------------------

    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ))

    model.add(BatchNormalization(momentum=0.8))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    # -------------------------------------------------
    # Classification Head
    # -------------------------------------------------

    model.add(GlobalAveragePooling2D())

    model.add(Dense(
        64,
        activation="relu"
    ))

    model.add(Dropout(0.30))

    model.add(Dense(
        2,
        activation="softmax"
    ))

    # -------------------------------------------------
    # Compile
    # -------------------------------------------------

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -------------------------------------------------
# Test Model
# -------------------------------------------------

if __name__ == "__main__":

    model = create_cnn_model()

    model.summary()