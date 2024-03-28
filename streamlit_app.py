import os
import random
import json
import streamlit as st
from PIL import Image
import numpy as np

@st.cache_data
def init_desc():
    with open('desc.json', encoding='utf-8') as f:
        brief = json.load(f)
    with open('full_desc.json', encoding='utf-8') as f:
        full = json.load(f)
    return brief, full

desc, full_desc = init_desc()

# Classic Spreads
def init_spread():
    card_cols = st.columns(3)
    card_cols = [x.empty() for x in card_cols]

    title_cols = st.columns(3)
    assert len(title_cols) == len(card_cols)

    return [(card_cols[i], title_cols[i]) for i in range(len(title_cols))]

spread = init_spread()

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

# config
with st.sidebar:
    is_full_desc = st.toggle('Full Description')

# Init card
for card, _ in spread:
    card.image(os.path.join(img_dir, 'cover.png'))

if st.button('Go', use_container_width=True):
    st.session_state['cards'] = []
    for card, title in spread:
        img, pk = pick_one()
        st.session_state['cards'].append((img, pk))


if 'cards' in st.session_state:
    card_state = st.session_state['cards']
    for idx, (card, title) in enumerate(spread):
        card_img = card_state[idx][0]
        card_pk = card_state[idx][1]
        card.image(card_img, caption=desc[card_pk]['t'])
        title.write(desc[card_pk]['p'])
        title.write(desc[card_pk]['n'])
        if is_full_desc:
            for l in full_desc[desc[card_pk]['t']]:
                title.write(l)
