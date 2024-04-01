import os
import random
from collections import namedtuple
import streamlit as st

@st.cache_resource
def init_embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

@st.cache_resource
def init_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)

def init_retriever(embeddings):
    from langchain_community.vectorstores import MongoDBAtlasVectorSearch

    DB_NAME = "langchain_db"
    COLLECTION_NAME = "test"
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "index_name"

    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        st.secrets["mongo"]["uri"],
        DB_NAME + "." + COLLECTION_NAME,
        embeddings,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )
    return vector_search.as_retriever()


@st.cache_data
def init_desc():
    import json
    with open('desc.json', encoding='utf-8') as f:
        brief = json.load(f)
    with open('full_desc.json', encoding='utf-8') as f:
        full = json.load(f)
    return brief, full

desc, full_desc = init_desc()

# Classic Spreads
SpreadSlot = namedtuple('SpreadSlot', ['card', 'desc'])
def init_time_flow_spread():
    card_cols = st.columns(3)
    card_cols = [x.empty() for x in card_cols]

    desc_cols = st.columns(3)
    assert len(desc_cols) == len(card_cols)

    return [SpreadSlot(card_cols[i], desc_cols[i]) for i in range(len(desc_cols))]

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
    from PIL import Image
    import numpy as np
    image = Image.open(fp)

    image_array = np.array(image)

    flipped_image = image.transpose(Image.ROTATE_180)
    flipped_image_array = np.array(flipped_image)

    return image_array, flipped_image_array

CardStat = namedtuple('CardStat', ['img', 'img_name', 'reversed'])
def pick_one():
    card_path = files.pop()
    img = open_image(os.path.join(card_img_dir, card_path))
    img_idx = random.choice([0, 1])
    return CardStat(img[img_idx], card_path, img_idx == 1)

# config
with st.sidebar:
    spread_type = st.selectbox('Using Spread', ['Time Flow'])
    is_full_desc = st.toggle('Full Description')
    is_gemini = st.toggle('Gemini Assistant')
    if is_gemini:
        gemini_token = st.text_input('Gemini Token', type='password')

if spread_type == 'Time Flow':
    spread = init_time_flow_spread()
    sys_prompt = '這是一個時間之流的塔羅牌牌陣，抽三張牌分別代表過去、現在和未來，根據使用者的問題、使用者抽出三張牌的意思以及以下{context}，給出一個合理的解釋'

# Init card
for slot in spread:
    slot.card.image(os.path.join(img_dir, 'cover.png'))

if st.button('Go', use_container_width=True):
    st.session_state['cards'] = []
    for _ in spread:
        stat = pick_one()
        st.session_state['cards'].append(stat)


if 'cards' in st.session_state:
    card_state = st.session_state['cards']
    for idx, slot in enumerate(spread):
        card_img = card_state[idx].img
        card_pk = card_state[idx].img_name
        slot.card.image(card_img, caption=desc[card_pk]['t'])

        slot.desc.write(desc[card_pk]['p'])
        slot.desc.write(desc[card_pk]['n'])
        if is_full_desc:
            slot.desc.text('\n'.join(full_desc[desc[card_pk]['t']]))

    if is_gemini and gemini_token:
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains import create_retrieval_chain
        from langchain_core.prompts import ChatPromptTemplate

        os.environ['GOOGLE_API_KEY'] = gemini_token

        llm = init_llm()
        embeddings = init_embeddings()
        retriever = init_retriever(embeddings)

        input_text = ', '.join([f"{'逆位' if stat.reversed else ''}{desc[stat.img_name]['t']}" for stat in card_state])

        prompt = ChatPromptTemplate.from_messages([
            ('system', sys_prompt),
            ('user', 'Question: {input}'),
        ])
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        with st.chat_message("user"):
            st.markdown(input_text)

        with st.empty():
            with st.status("Processing..."):
                response = retrieval_chain.invoke({
                    'input': input_text,
                    'context': []
                })
            
            with st.chat_message("assistant"):
                st.markdown(response['answer'])
