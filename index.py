import os
import cv2
import numpy as np

from model import make_model

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import LabelBinarizer

from loader import data_loader
from constants.index import age_list, gender_list

BS = 32
EPOCHS = 50

IMAGE_DIMS = (256, 256, 3)
IMAGE_DIMS_2D = (256, 256)

IMAGE_EXTENSION = '.jpg'

def create_model(classes, loss):

    model = make_model(classes = classes, input_shape = IMAGE_DIMS)
    model.compile(loss = loss, optimizer = 'adam', metrics = ["accuracy"])

    return model


def make_original_image(image_name):
    name = image_name.split('.')[2]
    return f'{name}{IMAGE_EXTENSION}'


def find_image_data(data, user_id, original_image):
    for record in data:
        if record['user_id'] == user_id and record['original_image'] == original_image:
            return record
    
    return None

age_mapping = [
    ('(0, 2)', '0-2'),
    ('2', '0-2'),
    ('3', '0-2'),
    ('(4, 6)', '4-6'),
    ('(8, 12)', '8-13'),
    ('13', '8-13'),
    ('22', '15-20'),
    ('(8, 23)', '15-20'),
    ('23', '25-32'),
    ('(15, 20)', '15-20'),
    ('(25, 32)', '25-32'),
    ('(27, 32)', '25-32'),
    ('32', '25-32'),
    ('34', '25-32'), 
    ('29', '25-32'),
    ('(38, 42)', '38-43'),
    ('35', '38-43'),
    ('36', '38-43'),
    ('42', '48-53'),
    ('45', '38-43'),
    ('(38, 43)', '38-43'),
    ('(38, 42)', '38-43'),
    ('(38, 48)', '48-53'),
    ('46', '48-53'),
    ('(48, 53)', '48-53'), 
    ('55', '48-53'),
    ('56', '48-53'),
    ('(60, 100)', '60+'),
    ('57', '60+'),
    ('58', '60+')
]
age_mapping_dict = {each[0]: each[1] for each in age_mapping}

def make_train_data(data, images_path):
    age_train_data = []
    age_train_labels = []

    gender_train_data = []
    gender_train_labels = []

    for user_id in os.listdir(images_path):
        user_images_path = f'{images_path}{user_id}/'
        for image_name in os.listdir(user_images_path):
            
            # skip non .jpg files
            if not image_name.endswith(IMAGE_EXTENSION):
                continue

            original_image = make_original_image(image_name)
            data_record = find_image_data(data, user_id, original_image)

            if (data_record):
                age_label_source = data_record['age']
                gender_label_source = data_record['gender']

                if age_label_source == 'None' or gender_label_source not in gender_list:
                    continue

                age_label = age_mapping_dict[age_label_source]
                age_train_labels.append(age_label)

                gender_label = gender_label_source
                gender_train_labels.append(gender_label)

                image_path = os.path.join(user_images_path, image_name)

                image = cv2.imread(image_path)
                image = cv2.resize(image, IMAGE_DIMS_2D)
                image = img_to_array(image)

                age_train_data.append(image)
                gender_train_data.append(image)

        # temporary, check memory problems
        if len(age_train_data) > 100:
            return (age_train_data, age_train_labels, gender_train_data, gender_train_labels)

    return (age_train_data, age_train_labels, gender_train_data, gender_train_labels)

def train_age_model(age_train_data, age_train_labels):
    age_train_data = np.array(age_train_data, dtype = "float") // 255
    age_train_labels = np.array(age_train_labels)

    age_label_binarizer = LabelBinarizer()
    age_train_labels = age_label_binarizer.fit_transform(age_train_labels)

    age_train_x = age_train_data[:int(len(age_train_data)*.8)]
    age_train_y = age_train_labels[:int(len(age_train_data)*.8)]
    age_test_x = age_train_data[int(len(age_train_data)*.8):]
    age_test_y = age_train_labels[int(len(age_train_data)*.8):]

    age_model = create_model(
        classes = len(age_label_binarizer.classes_),
        loss = "categorical_crossentropy"
    )

    aug = ImageDataGenerator(
        rotation_range = 25, 
        width_shift_range = 0.1,
        height_shift_range = 0.1, 
        shear_range = 0.2, 
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = "nearest"
    )

    _ = age_model.fit(
        aug.flow(age_train_x, age_train_y, batch_size = BS),
        validation_data = (age_test_x, age_test_y),
        steps_per_epoch = len(age_train_x) // BS,
        epochs = EPOCHS, verbose = 1)

    age_model.save('age_model.h5')

def train_gender_model(gender_train_data, gender_train_labels):
    gender_train_data = np.array(gender_train_data, dtype = "float") // 255
    gender_train_labels = np.array(gender_train_labels)

    gender_label_binarizer = LabelBinarizer()
    gender_train_labels = gender_label_binarizer.fit_transform(gender_train_labels)

    print(len(gender_train_labels))

    gender_train_x = gender_train_data[:int(len(gender_train_data)*.8)]
    gender_train_y = gender_train_labels[:int(len(gender_train_data)*.8)]
    gender_test_x = gender_train_data[int(len(gender_train_data)*.8):]
    gender_test_y = gender_train_labels[int(len(gender_train_data)*.8):]

    gender_model = create_model(
        classes = len(gender_label_binarizer.classes_),
        loss = "sparse_categorical_crossentropy"
    )

    aug = ImageDataGenerator(
        rotation_range = 25, 
        width_shift_range = 0.1,
        height_shift_range = 0.1, 
        shear_range = 0.2, 
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = "nearest"
    )

    _ = gender_model.fit(
        aug.flow(gender_train_x, gender_train_y, batch_size = BS),
        validation_data = (gender_test_x, gender_test_y),
        steps_per_epoch = len(gender_train_x) // BS,
        epochs = EPOCHS, verbose = 1)

    gender_model.save('gender_model.h5')

def main():
    data = data_loader()

    images_path = 'images/faces/'
    (age_train_data, age_train_labels, gender_train_data, gender_train_labels) = make_train_data(data, images_path)
    
    print('>> training age model...')
    train_age_model(age_train_data, age_train_labels)
    print('>> age model trained!')

    print('>> training gender model...')
    train_gender_model(gender_train_data, gender_train_labels)
    print('>> gender model trained!')

if __name__ == "__main__":
    main()