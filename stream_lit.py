# Document Loader
import pandas as pd
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from io import BytesIO
# Text Splitter
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import os, datetime
from langchain.vectorstores import FAISS
import textwrap
# Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st
import shutil
import pandas as pd
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.llms import GPT4All
from langchain import PromptTemplate, LLMChain

import requests

API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
headers = {"Authorization": "Bearer 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"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


LOCAL_DB = 'my_index'
LLAMA_PATH = '../GPT4ALL/models/ggml-model-q4_0.bin'


# Split text
def split_chunks(sources):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks


def create_index(chunks, embeddings):
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    search_index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    return search_index


def similarity_search(query, index):
    # k is the number of similarity searched that matches the query
    # default is 4
    matched_docs = index.similarity_search(query, k=3)
    sources = []
    for doc in matched_docs:
        sources.append(
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    return matched_docs, sources


def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')
    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text


def load_file(file, embeddings):
    general_start = datetime.datetime.now()  # not used now but useful
    print("starting file loading...")
    loop_start = datetime.datetime.now()  # not used now but useful
    print("Split file text...")
    loader = PyPDFLoader(file)
    docs = loader.load()
    chunks = split_chunks(docs)
    print("start index loading...")
    db = create_index(chunks, embeddings)
    if os.path.isdir('./my_index'):
        db0 = FAISS.load_local(LOCAL_DB, embeddings)
        db.merge_from(db0)
        shutil.rmtree(LOCAL_DB)
    db.save_local(LOCAL_DB)
    print("local index updated...")


embeddings = HuggingFaceEmbeddings()

uploaded_file = st.file_uploader('upload file')
if uploaded_file is not None:
    if st.button("Process File"):
        with open(os.path.join("./docs/", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        load_file('./docs/' + uploaded_file.name, embeddings=embeddings)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = GPT4All(model=LLAMA_PATH, backend='llama', callback_manager=callback_manager, verbose=True)

# create the prompt template
template = """
Please use the following context to answer questions.
Context: {context}
---
Question: {question}
Answer: Let's think step by step and logically."""

question = st.text_input("Search in uploaded documents")
if st.button("Submit"):
    index = FAISS.load_local(LOCAL_DB, embeddings)
    # Hardcoded question
    docs = index.similarity_search(question)
    content, documents, page = [], [], []
    for doc in docs:
        # st.subheader('Relevant Context:')
        content.append(wrap_text_preserve_newlines(str(doc.page_content)))
        documents.append(wrap_text_preserve_newlines(str(doc.metadata['source']).split('/')[-1]))
        page.append(wrap_text_preserve_newlines(str(doc.metadata['page'])))
    res = pd.DataFrame.from_dict({'Content': content, 'Document': documents, 'Page': page})
    st.table(res)
    # Creating the context
    context = "\n".join([doc.page_content for doc in docs])
    # instantiating the prompt template and the GPT4All chain
    prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    ans = llm_chain.run(question)
    st.subheader('Answer from chatbot')
    # output = query({
    #     "question": question,
    #     "context": context
    # })
    st.write(ans)
