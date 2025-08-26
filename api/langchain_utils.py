# langchain_utils.py
# from langchain_openai import ChatOpenAI  # REMOVE THIS LINE
from langchain_community.chat_models import ChatOllama # <-- ADD THIS IMPORT
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List
from langchain_core.documents import Document
import os
from api.chroma_utils import vectorstore

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
output_parser = StrOutputParser()

# Set up prompts and chains (This remains the same)
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the user's question. If you don't know the answer based on the context, just say so. Don't try to make up an answer."), # Slightly better prompt for open-source models
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

def get_rag_chain(model="phi3:mini"): # <-- Default to fastest model "phi3:mini"
    # REPLACE THE OpenAI LLM WITH OLLAMA LLM
    # llm = ChatOpenAI(model=model)
    llm = ChatOllama(model=model, temperature=0) # <-- Use ChatOllama instead. 'model' must match the name you used with `ollama pull`
    
    # The rest of the chain creation logic remains identical
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)    
    return rag_chain