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
    is_gemini = st.toggle('Gemini Assistant')
    if is_gemini:
        gemini_token = st.text_input('Gemini Token', type='password')

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
            title.text('\n'.join(full_desc[desc[card_pk]['t']]))

    if is_gemini:
        with st.chat_message("assistant"):
            st.markdown('''**過去（寶劍七）：**

這張牌代表狡猾、欺騙和背叛。它表明你在過去曾經歷過一段困難時期，你感到自己被欺騙或背叛。你可能感到不信任或懷疑，並且可能對未來持悲觀態度。

**現在（逆位魔術師）：**

逆位的魔術師代表缺乏創造力、動力和行動。它表明你現在可能感到被困住或受限制，無法充分發揮自己的潛力。你可能會感到沮喪或灰心，並可能在努力實現目標方面遇到困難。

**未來（倒吊者）：**

倒吊者代表犧牲、耐心和逆境中的成長。它表明你未來可能會經歷一段挑戰時期，但它也會給你成長和轉變的機會。你可能會被迫重新評估你的價值觀和優先事項，並可能需要做出一些犧牲。然而，這段經歷最終將幫助你成長並變得更加強大。

**綜合解釋：**

這三個牌陣表明，你過去曾經歷過一段困難時期，感到被欺騙或背叛。現在，你可能感到被困住或受限，無法充分發揮自己的潛力。然而，未來會給你帶來成長和轉變的機會，儘管這可能需要一些犧牲和耐心。

**建議：**

* 信任你的直覺並小心那些讓你感到不信任的人。
* 找到激勵你的方法，並採取行動實現你的目標。
* 擁抱逆境，因為它們提供了成長和學習的機會。
* 保持耐心和積極，相信事情最終會好起來。''')
