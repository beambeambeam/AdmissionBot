import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from prompt import custom_prompt

load_dotenv()

app = FastAPI(title="University Admission RAG API", version="1.0")

def SplitText():
    pdf_folder = r'D:\Works\admissionbot-rag\data'  
    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_folder, filename)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    return split_docs

#Load pdf to docs
split_docs = SplitText()

#Store pdf in vectordb
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
vectorstore = Chroma(
    collection_name="collection",
    embedding_function=embeddings,
)
vectorstore.add_documents(documents=split_docs)


#Create model
llm = ChatOpenAI(
    model='typhoon-v2-70b-instruct',            
    base_url='https://api.opentyphoon.ai/v1', 
    api_key=os.getenv('TYPHOON_API_KEY')                    
)

#Create chain
qa_chain = LLMChain(llm=llm, prompt=custom_prompt)

class QueryRequest(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"hello": "world"}

@app.post("/query")
async def query_rag(request: QueryRequest):
    user_question = request.question

    retrieved_docs = vectorstore.similarity_search(user_question, k=3)
    retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])
    print(retrieved_text)
    if not retrieved_text.strip():
        return {"answer": "ไม่ทราบ (No relevant information found)"}

    result = qa_chain.run({
        "retrieved_text": retrieved_text,
        "user_question": user_question
    })

    return {"answer": result}