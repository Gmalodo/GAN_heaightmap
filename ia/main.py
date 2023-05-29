import os
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from model import make_generator_model, make_discriminator_model
from IPython import display

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# def training_images():
#     with gzip.open('../training_set/train-images-idx3-ubyte.gz', 'r') as f:
#         magic_number = int.from_bytes(f.read(4), 'big')
#         image_count = int.from_bytes(f.read(4), 'big')
#         row_count = int.from_bytes(f.read(4), 'big')
#         column_count = int.from_bytes(f.read(4), 'big')
#         image_data = f.read()
#         images = np.frombuffer(image_data, dtype=np.uint8).reshape((image_count, row_count, column_count))
#         max = images.max()
#         return (images - max / 2) / max
#
#
# def training_labels():
#     with gzip.open('../training_set/train-labels-idx1-ubyte.gz', 'r') as f:
#         magic_number = int.from_bytes(f.read(4), 'big')
#         label_count = int.from_bytes(f.read(4), 'big')
#         label_data = f.read()
#         labels = np.frombuffer(label_data, dtype=np.uint8)
#         return labels, LabelBinarizer().fit_transform(labels)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train(dataset, epochs):
    mgl, mdl = ([], [])
    for epoch in tqdm(range(epochs)):
        ggl, gdl = ([], [])
        for image_batch in dataset:
            gl, dl = train_step(image_batch)
            ggl.append(gl)
            gdl.append(dl)
        mgl.append(sum(ggl) / len(ggl))
        mdl.append(sum(gdl) / len(gdl))
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.plot(mgl)
    plt.subplot(1, 2, 2)
    plt.plot(mdl)
    plt.savefig("chart_loss.png")


def display_image(epoch_no):
    return Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


noise_dim = 1000
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])

if __name__ == "__main__":
    # os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    generator = make_generator_model()
    discriminator = make_discriminator_model()

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator, discriminator=discriminator)

    BUFFER_SIZE = 84
    BATCH_SIZE = 16
    train_images = []

    for f in os.listdir("../dataset3"):
        temp = np.array(list(Image.open(os.path.join("../dataset3", f)).getdata()))
        max = temp.max()
        train_images.append((temp - (max / 2) / max).reshape(128, 128, 1))

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    train(train_dataset, 500)
