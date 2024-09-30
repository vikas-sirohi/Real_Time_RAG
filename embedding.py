import os
#--------Embedding Library---------
from langchain_community.embeddings import HuggingFaceEmbeddings

# ----------PDF Loader----------
from langchain_community.document_loaders import PyPDFLoader

#-----. Text Splitter Library .-------
from langchain.text_splitter import RecursiveCharacterTextSplitter

#--------ChromaDb---------------
from langchain_community.vectorstores import Chroma
#--------------------------------

current_dir = os.getcwd()
file_path = os.path.join(current_dir, "Pdfs", "numerical_analysis.pdf")
persist_dir = os.path.join(current_dir, "db", "chroma_db")

if not os.path.exists(persist_dir):
    print("Perssistent directory does not exist. Initializing vector store....")

    # Checking if pdf exists or not.
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist, please check the path."
        )
    

    #----------PDF Reader-----------------
    loader = PyPDFLoader(file_path=file_path)
    docs = loader.load()

    # -----------Text Splitter ---------
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
    )

    splits = text_splitter.split_documents(docs)
    print(f"\n Total number of splits: {len(splits)}\n")

    model_name = "sentence-transformers/all-mpnet-base-v2"
    hf = HuggingFaceEmbeddings(model_name = model_name) 

    print("\n --- Creating Text Embeddings----\n")
    # ----------Embedding------------------
    vectordb = Chroma.from_documents(documents = splits,
                                    embedding = hf,
                                    persist_directory = persist_dir)   
    print("\n---Vector Store Created ---\n")

else:
    print("Vector Store already exists. No need to initialize.")
    
    
