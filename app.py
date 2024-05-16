import streamlit as st
from llm_chains import load_normal_chain, chatChain
from langchain.memory import StreamlitChatMessageHistory


def load_chain(chat_history):
    return load_normal_chain(chat_history)

def clear_input_field():
    st.session_state['user_question'] = st.session_state['user_input']
    st.session_state['user_input'] = ""


def set_send_input():
    st.session_state['send_input'] = True
    clear_input_field()

def main():
    st.title("Local Chat App")
    chat_container=st.container()

    if 'send_input' not in st.session_state:
        st.session_state['send_input'] = False
        st.session_state['user_question']=""

    chat_history = StreamlitChatMessageHistory(key="history")
    llm_chain = load_chain(chat_history)
    user_input = st.chat_input("Type your message here",key="user_input")

    #send_button = st.button("Send",key="send_button")



    with chat_container:
        if user_input:
            st.chat_message("user").write(user_input)
            print(f"user_input: {user_input}")
            print(f"chat_history: {chat_history}")
            llm_response = llm_chain.run(user_input=user_input,history=chat_history)
            st.chat_message("ai").write(llm_response)
            user_input = None
    
    if chat_history.messages != []:
        with chat_container:
            st.write("Chat history:")
            for message in chat_history.messages:
                st.chat_message(message.type).write(message.content)

if __name__ == "__main__":
    main()