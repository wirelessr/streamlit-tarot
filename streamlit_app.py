import os
import random
import streamlit as st
from PIL import Image
import numpy as np

# Classic Spreads
cols = st.columns(3)
cols = [x.empty() for x in cols]

base_dir = '.'
img_dir = os.path.join(base_dir, 'img')
card_img_dir = os.path.join(img_dir, 'big')
files = os.listdir(card_img_dir)
files = [x for x in files if '_' not in x]
random.shuffle(files)

def open_image(fp):
    image = Image.open(fp)

    image_array = np.array(image)

    flipped_image = image.transpose(Image.ROTATE_180)
    flipped_image_array = np.array(flipped_image)

    return image_array, flipped_image_array

def pick_one():
    card_path = files.pop()
    img = open_image(os.path.join(card_img_dir, card_path))
    img_idx = random.choice([0, 1])
    return img[img_idx]

for col in cols:
    col.image(os.path.join(img_dir, 'cover.png'))

if st.button('Go', use_container_width=True):
    for col in cols:
        img = pick_one()
        col.image(img)
