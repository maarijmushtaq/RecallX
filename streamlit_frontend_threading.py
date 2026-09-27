import streamlit as st
from langgraph_backend import chatbot, generate_chat_title
from langchain_core.messages import HumanMessage
import uuid


# =========================================================
# Utility Functions
# =========================================================

def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):

    if thread_id not in st.session_state["chat_threads"]:

        st.session_state["chat_threads"].append(thread_id)

        # Assign default title to new conversation
        st.session_state["chat_titles"][thread_id] = "New Chat"


def reset_chat():

    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id

    add_thread(thread_id)

    st.session_state["message_history"] = []


def load_conversation(thread_id):

    return chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    ).values["messages"]


# =========================================================
# Initialize Session State
# =========================================================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []


if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}


# Make sure current thread exists
add_thread(st.session_state["thread_id"])


# =========================================================
# Sidebar
# =========================================================

st.sidebar.title("LangGraph Chatbot")


# New Chat
if st.sidebar.button("New Chat"):

    reset_chat()

    # Refresh UI
    st.rerun()


st.sidebar.header("My Conversations")


# Display conversations
for thread_id in st.session_state["chat_threads"][::-1]:

    # Get title
    title = st.session_state["chat_titles"].get(
        thread_id,
        "New Chat"
    )

    # Display title instead of thread ID
    if st.sidebar.button(
        title,
        key=f"thread_{thread_id}"
    ):

        # Select this conversation
        st.session_state["thread_id"] = thread_id

        # Load messages from LangGraph
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:

            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"

            temp_messages.append({
                "role": role,
                "content": msg.content
            })

        st.session_state["message_history"] = temp_messages

        # Refresh UI
        st.rerun()


# =========================================================
# Main UI
# =========================================================

# Display previous messages
for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):
        st.text(message["content"])


# =========================================================
# Chat Input
# =========================================================

user_input = st.chat_input("Type Here")


if user_input:

    # Current conversation
    current_thread = st.session_state["thread_id"]


    # -----------------------------------------------------
    # Generate title ONLY for the first message
    # -----------------------------------------------------

    if st.session_state["chat_titles"][current_thread] == "New Chat":

        title = generate_chat_title(user_input)

        st.session_state["chat_titles"][current_thread] = title


    # -----------------------------------------------------
    # Add user message to UI history
    # -----------------------------------------------------

    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })


    # Show user message
    with st.chat_message("user"):
        st.text(user_input)


    # -----------------------------------------------------
    # LangGraph configuration
    # -----------------------------------------------------

    config = {
        "configurable": {
            "thread_id": current_thread
        }
    }


    # -----------------------------------------------------
    # Generate AI response
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        ai_message = st.write_stream(

            message_chunk.content

            for message_chunk, metadata

            in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },

                config=config,

                stream_mode="messages"
            )
        )


    # -----------------------------------------------------
    # Save AI response in UI history
    # -----------------------------------------------------

    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })