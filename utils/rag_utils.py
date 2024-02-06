import os
import boto3

from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embeddings and Vector Store
from langchain.vectorstores import FAISS

# LLM Models
from langchain.chains import RetrievalQA

# Create a folder to save uploaded PDF files
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# Bedrock Client
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1" , client=bedrock)


prompt_template = """
Human: use the following pieces of context to provide a concise answer
to the question at the end but atleast summarize with 250 words with
detailed explanations. If you don't know the answer, just say you don't
rather than try to ake up an answer.
<context>
{context}
</context>

Question: {question}

Assistant: 
"""


class RAG:
    def __init__(self):
        self.bedrock = bedrock
        self.bedrock_embeddings = bedrock_embeddings
        self.data_folder_path = data_folder
        self.vector_save_name = "faiss_index"

        self.PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )


    # Data Ingestion
    def data_ingestion(self):
        loader = PyPDFDirectoryLoader(self.data_folder_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
                                            chunk_size=10000,
                                            chunk_overlap=1000
                                        )
        
        docs = text_splitter.split_documents(documents)

        return docs
    
    # Vector Embedding and Vector Store
    def get_vector_store(self, docs):
        vectorstore_faiss = FAISS.from_documents(
            docs,
            self.bedrock_embeddings
        )

        vectorstore_faiss.save_local("faiss_index")

    def load_vectors(self):
        return FAISS.load_local(self.vector_save_name, self.bedrock_embeddings)

    def get_claude_llm(self):
        # Create model
        llm = Bedrock(model_id="ai21.j2-mid-v1",
                    client=self.bedrock, model_kwargs={"maxTokens": 512}
                    )
        
        return llm

    def get_llma2_llm(self):
        # Create model
        llm = Bedrock(model_id="meta.llama2-70b-chat-v1",
                    client=self.bedrock, model_kwargs={"max_gen_len": 512}
                    )
        
        return llm
    
    def get_response_llm(self, llm, vectorstore_faiss, query):
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore_faiss.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.PROMPT}
        )

        answer = qa({"query": query})

        return answer["result"]
    
    def save_uploadedfile(self, uploadedfile):
        with open(os.path.join(self.data_folder_path, uploadedfile.name), "wb") as f:
            f.write(uploadedfile.getbuffer())



        
    




