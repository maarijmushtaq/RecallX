import uuid

import streamlit as st

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from final_backend_with_memory import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)


# =========================================================
# Page Config
# =========================================================

st.set_page_config(
    page_title="RecallX Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Custom Theme / CSS / Animations
# =========================================================

CUSTOM_CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --grad-1: #7F5AF0;
    --grad-2: #2CB67D;
    --accent: #7F5AF0;
    --accent-soft: rgba(127, 90, 240, 0.12);
    --bg-dark: #0f1117;
    --card-bg: rgba(255, 255, 255, 0.04);
    --text-light: #e4e4e7;
}

html, body, [class*="css"] {
    font-family: 'Inter', 'Poppins', sans-serif;
}

/* Flat background matching the app header color */
.stApp {
    background-color: var(--bg-dark);
    color: var(--text-light);
}

/* Make the top header bar match the same background */
header[data-testid="stHeader"] {
    background-color: var(--bg-dark);
}

/* Bottom chat input bar - Streamlit's own fixed container, force same flat color */
div[data-testid="stBottom"],
div[data-testid="stBottom"] > div,
div[data-testid="stBottomBlockContainer"] {
    background-color: var(--bg-dark) !important;
}

/* Main title */
h1 {
    font-family: 'Poppins', sans-serif !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, var(--grad-1), var(--grad-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: fadeInDown 0.8s ease;
    padding-bottom: 4px;
}

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-16px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Sidebar styling - same flat color as header/background */
section[data-testid="stSidebar"] {
    background-color: var(--bg-dark);
    border-right: 1px solid rgba(127, 90, 240, 0.15);
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    background: linear-gradient(90deg, var(--grad-1), var(--grad-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}

/* Sidebar title glow */
section[data-testid="stSidebar"] .stMarkdown p {
    animation: fadeIn 1s ease;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--grad-1), var(--grad-2));
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    padding: 0.55em 1.1em;
    transition: transform 0.2s ease, box-shadow 0.25s ease;
    box-shadow: 0 4px 14px rgba(127, 90, 240, 0.25);
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 8px 22px rgba(127, 90, 240, 0.45);
    filter: brightness(1.08);
}

.stButton > button:active {
    transform: translateY(0) scale(0.98);
}

/* Thread list buttons - subtler style */
section[data-testid="stSidebar"] .stButton > button {
    background: var(--card-bg);
    color: var(--text-light);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    text-align: left;
    box-shadow: none;
    font-weight: 500;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--accent-soft);
    border-color: var(--accent);
    color: #fff;
    transform: translateX(4px);
    box-shadow: 0 0 0 1px var(--accent);
}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background: var(--card-bg);
    border: 1.5px dashed rgba(127, 90, 240, 0.5) !important;
    border-radius: 14px;
    transition: border-color 0.3s ease, background 0.3s ease;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--grad-2) !important;
    background: rgba(44, 182, 125, 0.06);
}

/* Success / info boxes */
div[data-testid="stAlert"] {
    border-radius: 14px;
    animation: popIn 0.35s ease;
    backdrop-filter: blur(6px);
}

@keyframes popIn {
    from { opacity: 0; transform: scale(0.96); }
    to { opacity: 1; transform: scale(1); }
}

/* Status widget (tool usage) */
div[data-testid="stStatusWidget"] {
    border-radius: 14px;
    border: 1px solid rgba(127, 90, 240, 0.3);
    animation: pulseGlow 2s infinite;
}

@keyframes pulseGlow {
    0%   { box-shadow: 0 0 0 0 rgba(127, 90, 240, 0.35); }
    70%  { box-shadow: 0 0 0 8px rgba(127, 90, 240, 0); }
    100% { box-shadow: 0 0 0 0 rgba(127, 90, 240, 0); }
}

/* Chat messages */
div[data-testid="stChatMessage"] {
    background: var(--card-bg);
    border-radius: 16px;
    padding: 0.4rem 0.6rem;
    margin-bottom: 0.6rem;
    border: 1px solid rgba(255,255,255,0.06);
    animation: slideUpFade 0.35s ease;
    backdrop-filter: blur(8px);
    transition: border-color 0.2s ease;
}

div[data-testid="stChatMessage"]:hover {
    border-color: rgba(127, 90, 240, 0.4);
}

@keyframes slideUpFade {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Chat input box */
div[data-testid="stChatInput"] textarea {
    background: var(--card-bg) !important;
    border: 1.5px solid rgba(127, 90, 240, 0.35) !important;
    border-radius: 14px !important;
    color: var(--text-light) !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}

div[data-testid="stChatInput"] textarea:focus {
    border-color: var(--grad-2) !important;
    box-shadow: 0 0 0 3px rgba(44, 182, 125, 0.18) !important;
}

/* Sidebar code / thread id chips */
code {
    background: var(--accent-soft) !important;
    color: #b9a7ff !important;
    border-radius: 6px !important;
    padding: 2px 6px !important;
}

/* Caption / metadata footer */
.stCaption, [data-testid="stCaptionContainer"] {
    animation: fadeIn 0.6s ease;
    opacity: 0.85;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--grad-1), var(--grad-2));
    border-radius: 8px;
}

/* Subtle divider glow under title */
.title-divider {
    height: 2px;
    width: 100%;
    background: linear-gradient(90deg, var(--grad-1), var(--grad-2), transparent);
    border-radius: 4px;
    margin-bottom: 1.4rem;
    animation: fadeIn 1s ease;
}

/* Sidebar section headers */
section[data-testid="stSidebar"] .stSubheader, 
section[data-testid="stSidebar"] h3 {
    margin-top: 1.2rem;
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================
# Utilities
# =========================================================


def generate_thread_id():
    return str(uuid.uuid4())



def reset_chat():

    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id

    add_thread(thread_id)

    st.session_state["message_history"] = []




def add_thread(thread_id):

    if thread_id not in st.session_state["chat_threads"]:

        st.session_state["chat_threads"].append(
            thread_id
        )





def load_conversation(thread_id):

    state = chatbot.get_state(

        config={

            "configurable": {

                "thread_id": thread_id,

                "user_id": st.session_state["user_id"]

            }

        }

    )


    return state.values.get(
        "messages",
        []
    )





# =========================================================
# Session Initialization
# =========================================================


if "message_history" not in st.session_state:

    st.session_state["message_history"] = []



if "thread_id" not in st.session_state:

    st.session_state["thread_id"] = generate_thread_id()



# Long-term memory user identifier

if "user_id" not in st.session_state:

    st.session_state["user_id"] = "user_001"




if "chat_threads" not in st.session_state:

    st.session_state["chat_threads"] = retrieve_all_threads()



if "ingested_docs" not in st.session_state:

    st.session_state["ingested_docs"] = {}



add_thread(
    st.session_state["thread_id"]
)



thread_key = str(
    st.session_state["thread_id"]
)


thread_docs = st.session_state["ingested_docs"].setdefault(
    thread_key,
    {}
)



threads = st.session_state["chat_threads"][::-1]


selected_thread = None



# =========================================================
# Sidebar
# =========================================================


st.sidebar.markdown(
    "## 🧠 RecallX AI Chatbot"
)

st.sidebar.caption("Your AI research companion ✨")




st.sidebar.markdown("---")


if st.sidebar.button(
    "➕ New Chat",
    use_container_width=True
):

    reset_chat()

    st.rerun()




if thread_docs:

    latest_doc = list(
        thread_docs.values()
    )[-1]


    st.sidebar.success(

        f"""
📄 **Using:** `{latest_doc.get('filename')}`

🧩 **Chunks:** {latest_doc.get('chunks')}

📃 **Pages:** {latest_doc.get('documents')}
"""

    )


else:

    st.sidebar.info(
        "📭 No PDF indexed yet."
    )




uploaded_pdf = st.sidebar.file_uploader(

    "📤 Upload PDF for this chat",

    type=["pdf"]

)



if uploaded_pdf:


    if uploaded_pdf.name in thread_docs:


        st.sidebar.info(

            f"✅ `{uploaded_pdf.name}` already processed."

        )


    else:


        with st.sidebar.status(
            "🔍 Indexing PDF...",
            expanded=True
        ) as status_box:


            summary = ingest_pdf(

                uploaded_pdf.getvalue(),

                thread_id=thread_key,

                filename=uploaded_pdf.name

            )


            thread_docs[
                uploaded_pdf.name
            ] = summary



            status_box.update(

                label="✅ PDF indexed",

                state="complete",

                expanded=False

            )





# =========================================================
# Previous Conversations
# =========================================================


st.sidebar.markdown("---")
st.sidebar.subheader(
    "💬 Past Conversations"
)



if not threads:


    st.sidebar.caption(
        "No past conversations yet."
    )


else:


    for thread_id in threads:


        if st.sidebar.button(

            f"🗂️ {str(thread_id)[:8]}…",

            key=f"thread-{thread_id}",

            use_container_width=True

        ):

            selected_thread = thread_id





# =========================================================
# Main Chat Interface
# =========================================================


st.title(
    "✨ RecallX AI Agentic Chatbot"
)

st.markdown('<div class="title-divider"></div>', unsafe_allow_html=True)

st.caption("Chat, upload PDFs, and let your assistant use tools in real time 🚀")




for message in st.session_state["message_history"]:


    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"


    with st.chat_message(
        message["role"],
        avatar=avatar
    ):

        st.write(
            message["content"]
        )





user_input = st.chat_input(

    "💭 Ask about your PDF or use a tool..."

)




if user_input:


    st.session_state["message_history"].append(

        {
            "role": "user",
            "content": user_input
        }

    )



    with st.chat_message(
        "user",
        avatar="🧑‍💻"
    ):

        st.write(
            user_input
        )




    CONFIG = {


        "configurable": {


            # SQLite memory

            "thread_id":

                thread_key,


            # PostgreSQL memory

            "user_id":

                st.session_state["user_id"]

        },


        "metadata": {


            "thread_id":

                thread_key,


            "user_id":

                st.session_state["user_id"]

        },


        "run_name":

            "chat_turn"

    }




    with st.chat_message(
        "assistant",
        avatar="🤖"
    ):


        status_holder = {
            "box": None
        }



        def ai_stream():


            for message_chunk, _ in chatbot.stream(

                {
                    "messages":
                    [
                        HumanMessage(
                            content=user_input
                        )
                    ]
                },


                config=CONFIG,


                stream_mode="messages"

            ):



                if isinstance(
                    message_chunk,
                    ToolMessage
                ):


                    tool_name = getattr(
                        message_chunk,
                        "name",
                        "tool"
                    )



                    if status_holder["box"] is None:


                        status_holder["box"] = st.status(

                            f"⚙️ Using tool: {tool_name}",

                            expanded=True

                        )



                if isinstance(
                    message_chunk,
                    AIMessage
                ):


                    yield message_chunk.content





        ai_message = st.write_stream(
            ai_stream()
        )





        if status_holder["box"]:


            status_holder["box"].update(

                label="✅ Tool completed",

                state="complete",

                expanded=False

            )





    st.session_state["message_history"].append(

        {

            "role":
                "assistant",

            "content":
                ai_message

        }

    )





    doc_meta = thread_document_metadata(
        thread_key
    )


    if doc_meta:


        st.caption(

            f"""
📄 Document: **{doc_meta.get('filename')}**  •  🧩 Chunks: **{doc_meta.get('chunks')}**  •  📃 Pages: **{doc_meta.get('documents')}**
"""

        )





# =========================================================
# Load Selected Thread
# =========================================================


if selected_thread:


    st.session_state["thread_id"] = selected_thread



    messages = load_conversation(
        selected_thread
    )



    temp_messages = []



    for msg in messages:


        if isinstance(
            msg,
            HumanMessage
        ):

            role = "user"


        elif isinstance(
            msg,
            AIMessage
        ):

            role = "assistant"


        else:

            continue



        temp_messages.append(

            {

                "role":
                    role,

                "content":
                    msg.content

            }

        )



    st.session_state["message_history"] = temp_messages



    st.session_state["ingested_docs"].setdefault(

        str(selected_thread),

        {}

    )



    st.rerun()
