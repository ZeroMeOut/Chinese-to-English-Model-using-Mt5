from transformers import AutoTokenizer, MT5ForConditionalGeneration, logging
import streamlit as st
import base64

logging.set_verbosity_warning()

model_path_or_name = 'outputs/best_model'


@st.cache_resource
def load_model():
    mt5_trained = MT5ForConditionalGeneration.from_pretrained(model_path_or_name)
    return mt5_trained


@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("./image/background.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("EN-ZH Translator")
model = load_model()
tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, use_fast=False)
query = st.text_input("Input data", value="Welcome :)")

if query:
    input_ids = tokenizer(query, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    st.write(tokenizer.decode(outputs[0], skip_special_tokens=True))
