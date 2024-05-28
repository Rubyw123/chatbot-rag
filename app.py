import streamlit as st
from llm_chains import load_normal_chain, chatChain
from langchain.memory import StreamlitChatMessageHistory
from streamlit_mic_recorder import mic_recorder
from utils import save_chat_history_json,get_timestamp,load_chat_history_json
from audio import decode_audio
import yaml
import os
with open("config.yaml","r") as f:
    config = yaml.safe_load(f)

def load_chain(chat_history):
    return load_normal_chain(chat_history)

def clear_input_field():
    st.session_state['user_question'] = st.session_state['user_input']
    st.session_state['user_input'] = ""

def set_send_input():
    st.session_state['send_input'] = True
    clear_input_field()

def save_chat_history():
    if st.session_state["history"] != []:
        if st.session_state["session_key"] == "new_session":
            st.session_state["new_session_key"] = get_timestamp() + ".json"
            save_chat_history_json(st.session_state["history"],config["chat_history_path"] + st.session_state["new_session_key"])
        else:
            save_chat_history_json(st.session_state["history"],config["chat_history_path"] + st.session_state["session_key"])


def load_chat_history():
        # Getting json file to history
        if st.session_state["session_key"] != "new_session":
            st.session_state["history"] = load_chat_history_json(config["chat_history_path"]+st.session_state["session_key"])
        else:
            st.session_state["history"] = []

def track_index():
    st.session_state["session_index_tracker"] = st.session_state["session_key"]

def main():
    # App title
    st.title("Local Chat App")
    chat_container=st.container()   

    # Sidebar
    st.sidebar.title("Chat Sessions")
    chat_sessions = ["new_session"] + os.listdir(config["chat_history_path"])

    # session states keys initialize
    if 'send_input' not in st.session_state:
        st.session_state["session_key"] = "new_session"
        st.session_state['send_input'] = False
        st.session_state['user_question']=""
        st.session_state["new_session_key"] = None
        st.session_state["session_index_tracker"] = "new_session"
    if st.session_state["session_key"] == "new_session" and st.session_state["new_session_key"] != None:
        st.session_state["session_index_tracker"] = st.session_state["new_session_key"]
        st.session_state["new_session_key"] = None


    index = chat_sessions.index(st.session_state["session_index_tracker"])
    st.sidebar.selectbox("Select a chat session", chat_sessions,key="session_key",index = index, on_change=track_index)

    # Load chat history
    load_chat_history()

    chat_history = StreamlitChatMessageHistory(key="history")
    llm_chain = load_chain(chat_history)

    # User input handling
    user_input = st.chat_input("Type your message here",key="user_input")

    # Voice recording
    voice_rec_col,send_col = st.columns(2)
    with voice_rec_col:
        record_audio = mic_recorder(
                start_prompt="Record Audio",
                stop_prompt="Stop recording",
                just_once=True
            )

    audio_file = st.sidebar.file_uploader("Upload an audio file",type=["wav","mp3","ogg","m4a"])

    if audio_file:
        transcript = decode_audio(audio_file.getvalue())
        llm_chain.run(" Summarize the following: " + transcript,chat_history)
    if record_audio:
        transcript = decode_audio(record_audio["bytes"])
        print(transcript)
        llm_chain.run(transcript,chat_history)

    # Chat history page
    if chat_history.messages != []:
        with chat_container:
            st.write("Chat history:")
            for message in chat_history.messages:
                st.chat_message(message.type).write(message.content)

    # LLM Chain invoke
    with chat_container:
        if user_input:
            st.chat_message("user").write(user_input)
            #print(f"user_input: {user_input}")
            #print(f"chat_history: {chat_history}")
            llm_response = llm_chain.run(user_input=user_input,history=chat_history)
            st.chat_message("ai").write(llm_response)
            user_input = None
    
    # Save chat history
    save_chat_history() 

if __name__ == "__main__":
    main()