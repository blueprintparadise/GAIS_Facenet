import os
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
# from numba import cuda
from pylab import rcParams
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
import albumentations as A
import random

# CREATE CLASS-------------------------------------------------------------------------------------------------------------------------------
image_directory = r'D:\DeepFace\deepface\deepface\images'

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
BOX_COLOR = (255, 0, 0)
dir_list = os.listdir(image_directory)


def remove_augmentation():
    for man in dir_list:
        image_dir = image_directory + "\\" + str(man)
        images = os.listdir(image_dir)
        filtered_files = [file for file in images if file.startswith("form_aug")]
        for file in filtered_files:
            path_to_file = os.path.join(image_dir, file)
            os.remove(path_to_file)
    return print('Cleared Augmentation')


remove_augmentation()


def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=2):
    x_min, y_min, x_max, y_max = map(lambda v: int(v), bbox)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    return img


def show_image(image, bbox):
    image = visualize_bbox(image.copy(), bbox)
    f = plt.figure(figsize=(18, 12))
    plt.imshow(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        interpolation='nearest'
    )
    plt.axis('off')
    f.tight_layout()
    plt.show()


def show_augmented(augmentation, image, bbox):
    augmented = augmentation(image=image, bboxes=[bbox], field_id=['1'])
    show_image(augmented['image'], augmented['bboxes'][0])


# [x_min, y_min, x_max, y_max], e.g. [97, 12, 247, 212].
bbox_params = A.BboxParams(
    format='pascal_voc',
    min_area=1,
    min_visibility=0.5,
    label_fields=['field_id']
)
doc_aug = A.Compose([
    A.Flip(p=0.25),
    A.RandomGamma(gamma_limit=(20, 300), p=0.3),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=35, p=0.9),
    A.RandomRotate90(p=0.25),
    # A.RGBShift(p=0.25),
    A.GaussNoise(p=0.25),
    A.RandomToneCurve(p=0.4),
    A.Solarize(p=0.4, threshold=235),
    A.CLAHE(p=0.5),
    # A.PiecewiseAffine(p=0.5),
    A.CoarseDropout(p=0.5),
    # A.MotionBlur(p=0.5),
    # A.OpticalDistortion(p=0.5),
    A.GridDistortion(p=0.2),
    A.OneOf([
        A.Blur(blur_limit=3, p=0.5),
        A.ColorJitter(p=0.5)
    ], p=1.0)

])

dir_list = os.listdir(image_directory)


def process(dir_list):
    for man in (dir_list):
        image_dir = image_directory + "\\" + str(man)
        images = os.listdir(image_dir)
        for i, image_name in enumerate(images):
            img_dir = image_directory + "\\" + str(man) + "/" + image_name
            form = cv2.imread(img_dir)
            STUDENT_ID_BBOX = [500, 600, 400, 400]
            # show_image(form, bbox=STUDENT_ID_BBOX)
            # show_augmented(doc_aug, form, STUDENT_ID_BBOX)
            DATASET_PATH = 'data/augmented'
            IMAGES_PATH = f'{image_dir}'

            os.makedirs(DATASET_PATH, exist_ok=True)
            os.makedirs(IMAGES_PATH, exist_ok=True)

            rows = []
            for i in tqdm(range(10)):
                try:
                    augmented = doc_aug(image=form, field_id=['1'])
                except:
                    print("\nSkipped :")
                # print(form)
                file_name = f'form_aug_{i}.jpg'
                # print(image_name+" Done")

                cv2.imwrite(f'{IMAGES_PATH}/{file_name}', augmented['image'])

            # pd.DataFrame(rows).to_csv(f'{DATASET_PATH}/annotations.csv', header=True, index=None)


process(dir_list)
