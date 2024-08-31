import streamlit as st
import pandas as pd
import numpy as np
import time
import torch
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer
from streamlit_lottie import st_lottie
import json

#Model
class SimpleGPT2SequenceClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes:int ,max_seq_len:int, gpt_model_name:str):
        super(SimpleGPT2SequenceClassifier,self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(gpt_model_name)
        self.fc1 = nn.Linear(hidden_size*max_seq_len, num_classes)


    def forward(self, input_id, mask):
        """
        Args:
                input_id: encoded inputs ids of sent.
        """
        gpt_out, _ = self.gpt2model(input_ids=input_id, attention_mask=mask, return_dict=False)
        batch_size = gpt_out.shape[0]
        linear_output = self.fc1(gpt_out.view(batch_size,-1))
        return linear_output

# Load the model
def prediction(text):
  model_new = SimpleGPT2SequenceClassifier(hidden_size=768, num_classes=5, max_seq_len=128, gpt_model_name="gpt2")
  model_new.load_state_dict(torch.load('model/simple_gpt2_ .pt'))
  model_new.eval()
  #test ="house prices show slight increase prices of homes in the uk rose a seasonally adjusted 0.5%"
  fixed_text = "".join(text.lower().split())
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  tokenizer.padding_side = "left"
  tokenizer.pad_token = tokenizer.eos_token
  model_input = tokenizer(fixed_text, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
  mask = model_input['attention_mask'].cpu()
  input_id = model_input["input_ids"].squeeze(1).cpu()

  output = model_new(input_id, mask)
  labels_map = {
    0: "business",
    1: "entertainment",
    2: "politics",
    3: "sport",
    4: "tech",
    }
  pred_label = labels_map[output.argmax(dim=1).item()]
  return pred_label

#Web
st.set_page_config(
    page_title="Text Classification", 
    page_icon=":peacock:", 
    layout="wide", 

)

def load_lottiefile(filePath: str):
    with open(filePath, "r") as f:
        return json.load(f)

def main():  
    with st.container():
        left, cent, right = st.columns(3)  
        with cent:
            st.header("Text Classification")
        with right:
            lot = load_lottiefile("image/animation.json")
            st_lottie(
                lot,
                speed=1,
                reverse=False,
                loop=True,
                quality="low",
                height=None,
                width=100,
                key=None,
            )  
    with st.container():
        st.text("Using to distinguish your article, whether there is business type, sport type, entertainment type...")
        st.info("Usage: Copy a paragraph of the article, then paste it into the box left below, click the 'Predict' button. The result will be displayed on the other side.")
    with st.container():
        left_column, center, right_column = st.columns((5, 1, 5))
        result = "" #ket qua
        input ="" #Kiem tra da nhap input chua
        predict = 0 #kiem tra da nhan vao predict hay chua, khoi tao
        with left_column:
            st.subheader("Your article")
            input = st.text_area("Enter a paragraph of your article:")
        with center:
            st.subheader("")
            st.text("")
            st.text("")
            if st.button("Predict"):
                if input!="":
                    predict = 1 #da nhap input
                else:
                    predict = 2 #chua nhap input
            if st.button("Refresh"):
                result = ""
        with right_column:
            st.subheader("Result")
            st.text("")
            st.text("")
            if predict == 1:
                result = prediction(input)
                with st.spinner('Wait for it...'):
                    time.sleep(3)
                st.success('The result is {}'.format(result))
            if predict == 2:
                st.warning("Enter your article to validate !!!")

if __name__=='__main__':
    main()


