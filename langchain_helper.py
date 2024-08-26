# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

from langchain_google_genai import GoogleGenerativeAI

# from langchain.llms import GooglePalm
from langchain_community.document_loaders import CSVLoader
# from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# from langchain.embeddings import HuggingFaceInstructEmbeddings




from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import os
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env =
api_key = os.getenv('google_api_key')
llm=GoogleGenerativeAI(model="models/text-bison-001",google_api_key=api_key,tempreture=0.1)
# # Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "Nvidia_QA_Database"

def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column="prompt")
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=instructor_embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings,allow_dangerous_deserialization=True)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.5)

    
    prompt_template = """
    You are a customer support agent, helping by following directives and answering questions.

Generate your response by following the steps below:

1. Recursively break-down the answer into smaller questions/directives

2. For each atomic question/directive:

2a. Select the most relevant information from the context 

3. Generate a draft response using the selected information

4. answer as organized as possible,in  bullet points if you could 

5. Generate your final response after adjusting it to increase accuracy and relevance

6. Now only show your final response! Do not provide any explanations or details

CONTEXT:

{context}

QUESTION: 
{question}

"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})


    print(chain)
    return chain

if __name__ == "__main__":
    chain = get_qa_chain()
    