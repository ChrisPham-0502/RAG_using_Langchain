import os
import openai
from dotenv import find_dotenv, load_dotenv

from langchain_openai import OpenAI
from langchain_community.chat_models import ChatOpenAI

from load_docs import load_docs

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

import streamlit as st
from streamlit_chat import message


load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

llm_model = "gpt-3.5-turbo"

llm = OpenAI(temperature=.7)
chat = ChatOpenAI(temperature=.7,
                  model=llm_model)

#======================================================================================================================
# Load documents
documents = load_docs()
chat_history = []

# 1. Text splitter
text_splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20,
    length_function = len
)

# 2. Embedding
embeddings = OpenAIEmbeddings()

docs = text_splitter.split_documents(documents)

#=====================================================================================================================
# 3. Storage
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory='./data'
)
vector_store.persist()
# ====================================================================================================================
# 4. Retrieve
retriever = vector_store.as_retriever(search_kwargs={"k":6})
# docs = retriever.get_relevant_documents("Tell me more about Data Science")

# Make a chain to answer questions
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, 
    vector_store.as_retriever(search_kwargs={'k':6}),
    return_source_documents=True,
    verbose=False
)
 

# cite sources - helper function to prettyfy responses
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response['source_documents']:
        print(source.metadata['source'])
        
#==============================FRONTEND=======================================
st.title("Docs QA Bot using Langchain")
st.header("Ask anything about your documents...")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []
    
def get_query():
    input_text = st.chat_input("Ask a question about your documents...")
    return input_text

# retrieve the user input
user_input = get_query()
if user_input:
    result = qa_chain({'question': user_input, 'chat_history': chat_history})
    st.session_state.past.append(user_input)
    st.session_state.generated.append(result['answer'])
    
if st.session_state['generated']:
     for i in range(len(st.session_state['generated'])):
         message(st.session_state['generated'][i], key=str(i))
         message(st.session_state['past'][i], is_user=True, key=str(i)+'_user')