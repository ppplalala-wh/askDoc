import shutil
from typing import List
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from os import listdir, walk
from os.path import isfile, join
# Document Loader
from langchain.document_loaders import PyPDFLoader
# Text Splitter
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import os, datetime
from langchain.vectorstores import FAISS
import textwrap
# Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
import shutil
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.llms import GPT4All
from schemas.context import Context
from langchain import PromptTemplate, LLMChain

app = FastAPI()

LOCAL_DB = 'my_index'
LLAMA_PATH='../GPT4ALL/models/ggml-model-q4_0.bin'
DOC_PATH='./docs/'

# create the prompt template
template = """
Please use the following context to answer questions.
Context: {context}
---
Question: {question}
Answer: Let's think step by step."""

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],  # Update this with the appropriate frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    message: str


embeddings = HuggingFaceEmbeddings()
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = GPT4All(model=LLAMA_PATH, backend='llama', callback_manager=callback_manager, verbose=True)
index = FAISS.load_local(LOCAL_DB, embeddings)


@app.post("/chat/")
def chat(question: Message, contexts: List[Context]):
    user_message = question.message
    context = "\n".join([doc.content for doc in contexts])
    # instantiating the prompt template and the GPT4All chain
    prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    bot_response = llm_chain.run(user_message)
    # Return the bot's response
    return {"message": bot_response}


@app.post("/search/")
def search(user_message: Message) -> List[Context]:
    # Add your chatbot logic here to process the user's message and generate a response
    # This is just a placeholder function
    # Hardcoded question
    docs = index.similarity_search(user_message.message)
    res=[]
    for doc in docs:
        # st.subheader('Relevant Context:')
        data = {
            "content" : wrap_text_preserve_newlines(str(doc.page_content)),
            "documentName": wrap_text_preserve_newlines(str(doc.metadata['source']).split('/')[-1]),
            "pageNo": wrap_text_preserve_newlines(str(doc.metadata['page']))}
        res.append(Context(**data))
    return res


@app.post("/docs/")
async def create_file(file: UploadFile):
    #try:
    with open(os.path.join(DOC_PATH, file.filename), "wb") as f:
        shutil.copyfileobj(file.file, f)
        file.file.close()
        f.close()
        load_file(os.path.join(DOC_PATH, file.filename), embeddings=embeddings)
    return {"Result": "OK"}
    #except:
    #    return {"Result": "File Uploading Failed"}


@app.get("/docs/")
async def get_files() -> List[str]:
    f = []
    for (dirpath, dirnames, filenames) in walk(DOC_PATH):
        f.extend(filenames)
    return f


@app.delete("/docs/")
async def delete_file(file_name: str):
    f = []
    for (dirpath, dirnames, filenames) in walk(DOC_PATH):
        f.extend(filenames)
    return f



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
    general_start = datetime.datetime.now() #not used now but useful
    print("starting file loading...")
    loop_start = datetime.datetime.now() #not used now but useful
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


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8087)
