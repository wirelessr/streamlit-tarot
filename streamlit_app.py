import os
import random
import json
import streamlit as st
from PIL import Image
import numpy as np

@st.cache_data
def init_desc():
    with open('desc.json', encoding='utf-8') as f:
        ret = json.load(f)
    return ret

desc = init_desc()

# Classic Spreads
cols = st.columns(3)
cols = [x.empty() for x in cols]

titles = st.columns(3)

base_dir = '.'
img_dir = os.path.join(base_dir, 'img')
card_img_dir = os.path.join(img_dir, 'big')

@st.cache_data
def init_file(dir_path):
    ret = os.listdir(dir_path)
    ret = [x for x in ret if '_' not in x]
    return ret

files = init_file(card_img_dir)
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
    return img[img_idx], card_path

for col in cols:
    col.image(os.path.join(img_dir, 'cover.png'))

if st.button('Go', use_container_width=True):
    desc_keys = []
    for col in cols:
        img, pk = pick_one()
        col.image(img, caption=desc[pk]['t'])
        desc_keys.append(pk)

    for idx, title in enumerate(titles):
        title.write(desc[desc_keys[idx]]['p'])
        title.write(desc[desc_keys[idx]]['n'])
