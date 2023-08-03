import streamlit as st
import streamlit.components.v1 as components
import base64
from tempfile import NamedTemporaryFile
from audiorecorder import audiorecorder
from whispercpp import Whisper
from gtts import gTTS

#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())

import os
import openai

openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENROUTER_REFERRER = "https://github.com/alonsosilvaallende/langchain-streamlit"

from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="google/palm-2-chat-bison",
                 streaming=True,
                 temperature=2,
                 headers={"HTTP-Referer": OPENROUTER_REFERRER})

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

html1="""
<a href='https://ko-fi.com/S6S3C06PD' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi1.png?v=3' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>
<br />
<a href="https://twitter.com/alonsosilva?ref_src=twsrc%5Etfw" class="twitter-follow-button" data-show-count="false">Follow @alonsosilva</a><script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
"""

# Streamlit
with st.sidebar:
    audio = audiorecorder("Click to send voice message", "Recording... Click when you're done", key="recorder")
    st.title("Custom Instructions for the Peasants")
    prompt_user = "You are a mean AI. Phrase all your replies as insults."
    prompt_received = st.text_area("How would you like the AI assistant to respond?", max_chars=2000)
    if len(prompt_received)>0:
        prompt_user = prompt_received

    st.subheader("Current instructions:\n\n"+f":blue[{prompt_user}]")
    tts1 = st.checkbox("Text-to-Speech")
with st.sidebar:
    components.html(html1)

prompt_1 = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(prompt_user),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt_1, llm=llm)

def my_response(llm, text: str) -> str:
    aux = llm.predict(input=text)
    counter = 0
    while ("I'm not able to help" in aux) and counter<2:
        counter += 1
        aux = llm.predict(input=text)
    return aux


# Download whisper.cpp
w = Whisper('tiny')

def inference(audio):
    # Save audio to a file:
    with NamedTemporaryFile(suffix=".mp3") as temp:
        with open(f"{temp.name}", "wb") as f:
            f.write(audio.tobytes())
        result = w.transcribe(f"{temp.name}")
        text = w.extract_text(result)
    return text[0]

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )




# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if (prompt := st.chat_input("Your message")) or len(audio):
    # If it's coming from the audio recorder transcribe the message with whisper.cpp
    if len(audio)>0:
        prompt = inference(audio)

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = my_response(conversation, prompt)
    # response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
        if tts1:
            tts = gTTS(response, lang='en')
            with NamedTemporaryFile(suffix=".mp3") as temp:
                tempname = temp.name
                tts.save(tempname)
                autoplay_audio(tempname)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
