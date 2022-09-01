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
# Computer vision with CNNs
#
# Create and train a classifier for horses or humans using the provided data.
# Make sure your final layer is a 1 neuron, activated by sigmoid as shown.
#
# The test will use images that are 300x300 with 3 bytes color depth so be sure to
# design your neural network accordingly

import tensorflow as tf
import urllib
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def solution_model():
    # _TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
    # _TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
    # urllib.request.urlretrieve(_TRAIN_URL, 'horse-or-human.zip')
    # local_zip = 'horse-or-human.zip'
    # zip_ref = zipfile.ZipFile(local_zip, 'r')
    # zip_ref.extractall('tmp/horse-or-human/')
    # zip_ref.close()
    # urllib.request.urlretrieve(_TEST_URL, 'testdata.zip')
    # local_zip = 'testdata.zip'
    # zip_ref = zipfile.ZipFile(local_zip, 'r')
    # zip_ref.extractall('tmp/testdata/')
    # zip_ref.close()

    train_datagen = ImageDataGenerator(
        #Your code here. Should at least have a rescale. Other parameters can help with overfitting.)
        rescale=1. / 255.,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
        )
    validation_datagen = ImageDataGenerator(#Your Code here)
        rescale = 1.0/255. )
    train_generator = train_datagen.flow_from_directory(
        #Your Code Here
        directory='tmp/horse-or-human/',
        batch_size=32,
        class_mode='binary',
        target_size=(300, 300)
        )

    validation_generator = validation_datagen.flow_from_directory(
        #Your Code Here)
        directory='tmp/testdata/',
        batch_size=32,
        class_mode='binary',
        target_size=(300, 300)
        )
    # Import the inception model
    from tensorflow.keras.applications.inception_v3 import InceptionV3

    # Create an instance of the inception model from the local pre-trained weights
    local_weights_file = 'data/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    pre_trained_model = InceptionV3(input_shape=(300, 300, 3),
                                    include_top=False,
                                    weights=None)

    pre_trained_model.load_weights(local_weights_file)

    # 使预训练模型中的所有层都不可训练
    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_desired_layer = pre_trained_model.get_layer('mixed7')
    last_output = last_desired_layer.output

    # model = tf.keras.models.Sequential([
    #     # Note the input shape specified on your first layer must be (300,300,3)
    #     # Your Code here
    #     tf.keras.layers.Flatten(),
    #     # This is the last layer. You should not change this code.
    #     tf.keras.layers.Dense(1, activation='sigmoid')
    # ])
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

    x = tf.keras.layers.Flatten()(last_output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(pre_trained_model.input, x)
    model.summary()
    from tensorflow.keras.optimizers import RMSprop
    model.compile(optimizer="adam",
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_generator,
                        validation_data=validation_generator,
                        epochs=20,
                        verbose=1,
                        callbacks=callbacks)
    return model

    # NOTE: If training is taking a very long time, you should consider setting the batch size
    # appropriately on the generator, and the steps per epoch in the model.fit() function.

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    # model.save("mymodel.h5")