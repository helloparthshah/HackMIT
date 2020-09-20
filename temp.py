import tensorflow as tf
from tensorflow.keras import layers

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './data1',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(96, 96),
    batch_size=64)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './data1',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(96, 96),
    batch_size=64)

class_names = train_ds.class_names
print(class_names)

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(
    1./255)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

''' model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
]) '''
model = tf.keras.Sequential()
# model.add(layers.Conv2D(filters=64, kernel_size=5, padding='same', activation='relu'))
model.add(layers.Conv2D(filters=64, kernel_size=5, activation='relu'))
model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.5))
# model.add(layers.Conv2D(filters=128, kernel_size=5, activation='relu'))
model.add(layers.Conv2D(filters=64, kernel_size=5, activation='relu'))
model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(filters=64, kernel_size=5, activation='relu'))
model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(num_classes))

''' model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy']) '''
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)

model.save("test1.h5")
