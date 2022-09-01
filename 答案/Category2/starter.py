# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#
import tensorflow as tf


def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # YOUR CODE HERE
    import os
    from tensorflow import keras
    (training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()
    training_images = training_images / 255.0
    test_images = test_images / 255.0
    # Build the classification model
    # model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
    #                                     tf.keras.layers.Dense(128, activation=tf.nn.relu),
    #                                     tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    model = tf.keras.models.Sequential([

        # Add convolutions and max pooling
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            min_delta=1e-4,
            patience=4,
            verbose=1
        ),
        ModelCheckpoint(
            filepath='mymodel.h5',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]

    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(training_images, training_labels, validation_data=(test_images, test_labels), epochs=15, callbacks=callbacks)
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    # model.save("mymodel.h5")
