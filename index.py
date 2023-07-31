import os
from langchain.vectorstores.chroma import Chroma
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
load_dotenv()
embedding_model_name = "hkunlp/instructor-large"
chunk_size = 1000
chunk_overlap  = 200
length_function = len

def create_index(file_path: str) -> None:

    # reader = PdfReader(file_path)
    # text = ''
    # for page in reader.pages:
    #     text += page.extract_text()

    # with open('output.txt', 'w', encoding='utf-8') as file:
    #     file.write(text)
    
    

    loader = DirectoryLoader(
        # './',
        # glob='output.txt',
        # loader_cls=TextLoader
        file_path
    )

    documents = loader.load()

    # text_splitter = CharacterTextSplitter(
    #     separator='\n',
    #     chunk_size=1024,
    #     chunk_overlap=128
    # )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function
        )

    texts = text_splitter.split_documents(documents)

    # embeddings = OpenAIEmbeddings(
    #     openai_api_key=os.getenv('OPENAI_API_KEY')
    # )
    
    
    embedding_function = HuggingFaceInstructEmbeddings(
        model_name=embedding_model_name,
        encode_kwargs={"show_progress_bar": True}
    )

    persist_directory = 'db'

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding_function,
        persist_directory=persist_directory
    )

    vectordb.persist()

create_index('source/')