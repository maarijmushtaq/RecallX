from __future__ import annotations

import os
import uuid
import sqlite3
import tempfile

from typing import (
    Annotated,
    Any,
    Dict,
    Optional,
    TypedDict,
    List
)

import requests

from dotenv import load_dotenv

from pydantic import BaseModel, Field


# LangChain

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

from langchain_community.tools import DuckDuckGoSearchRun

from langchain_community.vectorstores import FAISS


from langchain_core.messages import (
    BaseMessage,
    SystemMessage
)

from langchain_core.tools import tool

from langchain_core.runnables import RunnableConfig


# Models

from langchain_mistralai import ChatMistralAI

from langchain_huggingface import HuggingFaceEmbeddings



# LangGraph

from langgraph.graph import (
    START,
    StateGraph
)

from langgraph.graph.message import add_messages

from langgraph.prebuilt import (
    ToolNode,
    tools_condition
)


# Memory

from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.store.postgres import PostgresStore

from langgraph.store.base import BaseStore



# =====================================================
# ENV
# =====================================================

load_dotenv()



# =====================================================
# PROMPTS
# =====================================================


SYSTEM_PROMPT_TEMPLATE = """

You are a helpful AI assistant with memory.

Use user memory when available.

Known user information:

{user_details_content}


Rules:

- Use only provided memory.
- Do not invent facts.
- Personalize when relevant.


For PDF questions:
use rag_tool with correct thread_id.

"""


MEMORY_PROMPT = """

You maintain long term user memory.


Existing memories:

{user_details_content}


Extract only useful permanent information.

Store:

- name
- profession
- preferences
- projects
- goals


Rules:

- No assumptions.
- Only facts from user.
- Short atomic sentences.
- Mark duplicate information as false.

"""



# =====================================================
# LLM
# =====================================================


llm = ChatMistralAI(

    model="mistral-small-latest",

    mistral_api_key=os.getenv(
        "MISTRAL_API_KEY"
    ),

    temperature=0.3

)



memory_llm = ChatMistralAI(

    model="mistral-small-latest",

    mistral_api_key=os.getenv(
        "MISTRAL_API_KEY"
    ),

    temperature=0

)



# =====================================================
# EMBEDDINGS
# =====================================================


embeddings = HuggingFaceEmbeddings(

    model_name=
    "sentence-transformers/all-MiniLM-L6-v2"

)



# =====================================================
# MEMORY SCHEMA
# =====================================================


class MemoryItem(BaseModel):

    text: str

    is_new: bool



class MemoryDecision(BaseModel):

    should_write: bool

    memories: List[MemoryItem] = Field(
        default_factory=list
    )



memory_extractor = memory_llm.with_structured_output(
    MemoryDecision
)



# =====================================================
# PDF MEMORY
# =====================================================


_THREAD_RETRIEVERS: Dict[str,Any] = {}

_THREAD_METADATA: Dict[str,dict] = {}



def _get_retriever(thread_id):

    if (
        thread_id
        and thread_id in _THREAD_RETRIEVERS
    ):

        return _THREAD_RETRIEVERS[thread_id]


    return None



def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename=None
):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as f:

        f.write(file_bytes)

        path=f.name



    loader = PyPDFLoader(path)

    docs = loader.load()



    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=200

    )


    chunks = splitter.split_documents(
        docs
    )



    vectorstore = FAISS.from_documents(

        chunks,

        embeddings

    )


    _THREAD_RETRIEVERS[str(thread_id)] = (

        vectorstore.as_retriever(
            search_kwargs={"k":4}
        )

    )


    _THREAD_METADATA[str(thread_id)] = {

        "filename":filename,

        "documents":len(docs),

        "chunks":len(chunks)

    }



    os.remove(path)


    return _THREAD_METADATA[str(thread_id)]


# =====================================================
# TOOLS
# =====================================================


search_tool = DuckDuckGoSearchRun()



@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str
):

    """
    Basic calculator.
    """


    if operation == "add":

        result = first_num + second_num


    elif operation == "sub":

        result = first_num - second_num


    elif operation == "mul":

        result = first_num * second_num


    elif operation == "div":

        if second_num == 0:

            return {
                "error":
                "division by zero"
            }

        result = first_num / second_num


    else:

        return {
            "error":
            "invalid operation"
        }



    return {

        "result":
        result

    }





@tool
def get_stock_price(symbol:str):

    """
    Get stock price.
    """

    api_key = os.getenv(
        "ALPHA_VANTAGE_KEY"
    )


    url = (

        "https://www.alphavantage.co/query"

        f"?function=GLOBAL_QUOTE"

        f"&symbol={symbol}"

        f"&apikey={api_key}"

    )


    return requests.get(url).json()






@tool
def rag_tool(
    query:str,
    thread_id:str=None
):

    """
    Retrieve PDF information.
    """


    retriever = _get_retriever(
        thread_id
    )


    if retriever is None:

        return {

            "error":
            "No PDF uploaded"

        }



    docs = retriever.invoke(
        query
    )



    return {

        "context":

        [

            doc.page_content

            for doc in docs

        ]

    }





tools = [

    search_tool,

    calculator,

    get_stock_price,

    rag_tool

]


llm_with_tools = llm.bind_tools(
    tools
)





# =====================================================
# STATE
# =====================================================


class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]





# =====================================================
# REMEMBER NODE
# =====================================================


def remember_node(

    state:ChatState,

    config:RunnableConfig,

    *,

    store:BaseStore

):


    configurable = config.get(
        "configurable",
        {}
    )


    user_id = configurable.get(
        "user_id"
    )


    print(
        "USER ID:",
        user_id
    )



    if not user_id:

        return {}



    namespace = (

        "user",

        user_id,

        "details"

    )



    existing_items = store.search(
        namespace
    )



    existing = "\n".join(

        item.value["data"]

        for item in existing_items

    ) if existing_items else "(empty)"



    last_message = state["messages"][-1]



    if last_message.type != "human":

        return {}



    decision = memory_extractor.invoke(

        [

            SystemMessage(

                content=

                MEMORY_PROMPT.format(

                    user_details_content=

                    existing

                )

            ),


            {

                "role":
                "user",

                "content":
                last_message.content

            }

        ]

    )



    print(
        "MEMORY DECISION:",
        decision
    )



    if decision.should_write:


        for memory in decision.memories:


            if (

                memory.is_new

                and memory.text.strip()

            ):


                store.put(

                    namespace,

                    str(uuid.uuid4()),

                    {

                        "data":

                        memory.text.strip()

                    }

                )


                print(
                    "SAVED:",
                    memory.text
                )


    return {}







# =====================================================
# CHAT NODE
# =====================================================


def chat_node(
    state: ChatState,
    config: RunnableConfig,
    *,
    store: BaseStore
):

    configurable = config.get(
        "configurable",
        {}
    )


    print("\n========== CHAT NODE ==========")
    print("CONFIG:")
    print(configurable)


    thread_id = configurable.get(
        "thread_id"
    )


    user_id = configurable.get(
        "user_id"
    )


    print("THREAD ID:", thread_id)
    print("USER ID:", user_id)



    if not user_id:

        print("NO USER ID FOUND")
        
        user_details = "(empty)"

    else:

        namespace = (
            "user",
            user_id,
            "details"
        )


        print(
            "SEARCHING NAMESPACE:",
            namespace
        )


        memories = store.search(
            namespace
        )


        print(
            "MEMORY COUNT:",
            len(memories)
        )


        for m in memories:
            print(
                "MEMORY ITEM:",
                m.value
            )



        user_details = "\n".join(

            item.value.get(
                "data",
                ""
            )

            for item in memories

        ) if memories else "(empty)"



    print(
        "FINAL MEMORY SENT TO LLM:"
    )

    print(user_details)

    print("==============================\n")



    system_message = SystemMessage(

        content=

        SYSTEM_PROMPT_TEMPLATE.format(

            user_details_content=user_details

        )

        +

        f"""

Thread ID:

{thread_id}

Use rag_tool for PDF questions.

"""

    )


    response = llm_with_tools.invoke(

        [

            system_message,

            *state["messages"]

        ],

        config=config

    )


    return {

        "messages":

        [

            response

        ]

    }


# =====================================================
# TOOL NODE
# =====================================================


tool_node = ToolNode(
    tools
)




# =====================================================
# POSTGRESQL LONG TERM MEMORY STORE
# =====================================================


DB_URI = (
    "postgresql://postgres:postgres@localhost:5442/postgres"
    "?sslmode=disable"
)


# Global store
# This is used by:
# remember_node()
# chat_node()

store_context = PostgresStore.from_conn_string(
    DB_URI
)

store = store_context.__enter__()

store.setup()



# =====================================================
# SQLITE SHORT TERM MEMORY
# =====================================================


conn = sqlite3.connect(

    "chatbot.db",

    check_same_thread=False

)


checkpointer = SqliteSaver(
    conn
)




# =====================================================
# LANGGRAPH BUILD
# =====================================================


graph = StateGraph(
    ChatState
)



# Nodes

graph.add_node(

    "remember",

    remember_node

)


graph.add_node(

    "chat_node",

    chat_node

)


graph.add_node(

    "tools",

    tool_node

)



# Edges

graph.add_edge(

    START,

    "remember"

)


graph.add_edge(

    "remember",

    "chat_node"

)



graph.add_conditional_edges(

    "chat_node",

    tools_condition

)



graph.add_edge(

    "tools",

    "chat_node"

)



# Compile

chatbot = graph.compile(

    checkpointer=checkpointer,

    store=store

)




# =====================================================
# HELPERS
# =====================================================


def retrieve_all_threads():


    threads = set()



    for checkpoint in checkpointer.list(None):


        try:

            thread_id = (

                checkpoint.config

                ["configurable"]

                ["thread_id"]

            )


            threads.add(
                thread_id
            )


        except Exception:

            pass



    return list(threads)





def thread_document_metadata(
    thread_id:str
):


    return _THREAD_METADATA.get(

        str(thread_id),

        {}

    )





def thread_has_document(
    thread_id:str
):


    return (

        str(thread_id)

        in _THREAD_RETRIEVERS

    )





# =====================================================
# LTM TEST
# =====================================================


if __name__ == "__main__":


    config = {


        "configurable": {


            "thread_id":

            "thread_test_001",


            "user_id":

            "user_001"

        }

    }



    print("\n--- Saving Memory ---")



    chatbot.invoke(

        {

            "messages":

            [

                {

                    "role":

                    "user",


                    "content":

                    "My name is Maarij Mushtaq and I am a computer engineer."

                }

            ]

        },

        config

    )




    print("\n--- Asking From New Message ---")



    response = chatbot.invoke(

        {

            "messages":

            [

                {

                    "role":

                    "user",


                    "content":

                    "What is my name and profession?"

                }

            ]

        },

        config

    )



    print(

        response["messages"][-1].content

    )




    print("\n--- PostgreSQL Memory ---")



    memories = store.search(

        (

            "user",

            "user_001",

            "details"

        )

    )



    for memory in memories:


        print(

            memory.value["data"]

        )