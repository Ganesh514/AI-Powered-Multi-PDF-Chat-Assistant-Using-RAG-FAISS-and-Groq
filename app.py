import os
import time
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import (
    FAISS
)

from langchain_groq import (
    ChatGroq
)

load_dotenv()

st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="wide"
)


# ---------- THEME ----------

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True


with st.sidebar:

    st.title(
        "⚙ Settings"
    )

    st.session_state.dark_mode = st.toggle(
        "🌙 Dark Mode",
        value=st.session_state.dark_mode
    )


# ---------- CSS ----------

if st.session_state.dark_mode:

    bg1 = "#020617"
    bg2 = "#0F172A"
    bg3 = "#1E293B"

    card = "rgba(17,24,39,0.75)"

else:

    bg1 = "#EFF6FF"
    bg2 = "#DBEAFE"
    bg3 = "#BFDBFE"

    card = "rgba(255,255,255,0.85)"


st.markdown(
f"""
<style>

.stApp{{

background:

linear-gradient(
135deg,
{bg1},
{bg2},
{bg3}
);

}}


section[data-testid="stSidebar"]{{

background:

linear-gradient(
180deg,
#020617,
#111827
);

}}


.title{{

text-align:center;

font-size:48px;

font-weight:bold;

color:white;

}}


.message-card{{

background:{card};

backdrop-filter:blur(20px);

padding:20px;

border-radius:25px;

margin-bottom:15px;

box-shadow:

0px 10px 30px rgba(
0,
0,
0,
0.25
);

transition:0.3s;

}}


.source-card{{

background:

rgba(
30,
41,
59,
0.8
);

padding:12px;

border-radius:15px;

margin-top:15px;

color:#93C5FD;

}}

</style>
""",
unsafe_allow_html=True
)


st.markdown(
"<div class='title'>📄 Chat with PDF</div>",
unsafe_allow_html=True
)


# ---------- SESSION ----------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "current_files" not in st.session_state:
    st.session_state.current_files = []


# ---------- VECTORSTORE ----------

@st.cache_resource
def create_vectorstore(
    uploaded_files
):

    all_docs = []

    os.makedirs(
        "uploaded_files",
        exist_ok=True
    )

    for uploaded_file in uploaded_files:

        path = os.path.join(
            "uploaded_files",
            uploaded_file.name
        )

        with open(
            path,
            "wb"
        ) as f:

            f.write(
                uploaded_file.getbuffer()
            )

        loader = PyPDFLoader(
            path
        )

        docs = loader.load()

        all_docs.extend(
            docs
        )

    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
    )

    split_docs = (
        splitter.split_documents(
            all_docs
        )
    )

    embeddings = (
        HuggingFaceEmbeddings(
            model_name=
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    vectorstore = (
        FAISS.from_documents(
            split_docs,
            embeddings
        )
    )

    return vectorstore


# ---------- SIDEBAR ----------

with st.sidebar:

    st.title(
        "📂 PDFs"
    )

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    st.divider()

    if uploaded_files:

        for file in uploaded_files:

            st.write(
                "📄",
                file.name
            )

    st.divider()

    chat_history = ""

    for msg in st.session_state.messages:

        chat_history += (

            msg["role"]

            +

            ": "

            +

            msg["content"]

            +

            "\n\n"

        )

    st.download_button(

        "📥 Download Chat",

        chat_history,

        file_name="conversation.txt"

    )

    if st.button(
        "🗑 Clear Chat"
    ):

        st.session_state.messages = []

        st.rerun()


if uploaded_files:

    file_names = sorted(
        [
            file.name
            for file in uploaded_files
        ]
    )

    if (
        file_names
        !=
        st.session_state.current_files
    ):

        st.session_state.current_files = (
            file_names
        )

        vectorstore = (
            create_vectorstore(
                uploaded_files
            )
        )

        st.session_state.retriever = (
            vectorstore.as_retriever(
                search_kwargs={
                    "k":1
                }
            )
        )

    llm = ChatGroq(
        model=
        "llama-3.1-8b-instant"
    )
    avatar_map = {

        "user": "👤",

        "assistant": "🤖"

    }


    for message in st.session_state.messages:

        with st.chat_message(

            message["role"],

            avatar=avatar_map[
                message["role"]
            ]

        ):

            st.markdown(

                message["content"],

                unsafe_allow_html=True

            )


    question = st.chat_input(

        "✨ Ask anything about your PDFs..."

    )


    if question:

        st.session_state.messages.append(

            {

                "role": "user",

                "content": question

            }

        )


        docs = (

            st.session_state.retriever.invoke(

                question

            )

        )


        context = "\n\n".join(

            [

                doc.page_content

                for doc in docs

            ]

        )


        pages = []


        for doc in docs:

            if (

                "page"

                in

                doc.metadata

            ):

                pages.append(

                    str(

                        doc.metadata[
                            "page"
                        ]

                        + 1

                    )

                )


        previous_context = ""


        if len(

            st.session_state.messages

        ) > 6:

            previous_context = "\n".join(

                [

                    x["content"]

                    for x in

                    st.session_state.messages[-6:]

                ]

            )


        prompt = f"""

You are an AI PDF assistant.

Use the previous conversation and the PDF context.

Do NOT use outside knowledge.

If the answer cannot be found, say:

'I could not find the answer in the uploaded PDF.'

Previous Conversation:

{previous_context}


PDF Context:

{context}


Question:

{question}


Answer:

"""


        with st.chat_message(

            "assistant",

            avatar="🤖"

        ):

            placeholder = st.empty()

            full_response = ""


            response = (

                llm.invoke(

                    prompt

                )

            )


            answer = (

                response.content

            )


            words = answer.split()


            for word in words:

                full_response += (

                    word + " "

                )

                placeholder.markdown(

                    f"""

<div class='message-card'>

{full_response}

</div>

""",

                    unsafe_allow_html=True

                )

                time.sleep(

                    0.02

                )


            if pages:

                st.markdown(

                    f"""

<div class='source-card'>

📄 Source Pages

<br><br>

{", ".join(pages)}

</div>

""",

                    unsafe_allow_html=True

                )


        final_answer = (

            answer

            +

            f"""

<div class='source-card'>

📄 Source Pages

<br><br>

{", ".join(pages)}

</div>

"""

        )


        st.session_state.messages.append(

            {

                "role": "assistant",

                "content": final_answer

            }

        )


        st.rerun()


else:

    st.info(

        "👈 Upload one or more PDFs to begin."

    )