import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
import time
 
load_dotenv()
 
def load_pipeline():
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
 
    vectorstore = FAISS.load_local("faiss_index", hf_embeddings, allow_dangerous_deserialization=True)
 
    llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", temperature=0.7,
            max_tokens=2048
            )
   
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
 
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":3})
 
    prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""You are an e-commerce product expert specializing in analyzing online retail product data such asproduct descriptions, specifications, prices, availability, ratings, and customer reviews.
    Use ONLY the information provided from the given e-commerce dataset (product catalog).
    Stay strictly within the provided product context, attributes, and customer feedback.
    If the dataset contains the relevant information, provide a clear, short, and accurate answer focused on product details, pricing, features, comparisons, availability, or reviews.
    If the answer is not present in the provided dataset, respond with:
    "I Don't Know as the answer is not available in the provided context."

    Context:
    {context}

    Chat History:
    {chat_history}

    Question:
    {question}

    Answer:
    """
    )
 
    # Conversational RAG Chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    return chain
 
def ask_question(chain, question):
    start = time.time()
    result = chain.invoke({"question": question})
    latency = time.time() - start
    docs = result["source_documents"]
 
    retrieved_docs = [doc.page_content[:200] for doc in docs]
    sources = [doc.metadata for doc in docs]
 
    return {
        "answer": result["answer"],
        "retrieved_docs": retrieved_docs,
        "sources": sources,
        "latency": latency,
    }
 