from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(docs, chunk_size=1000, chunck_overlap=200):
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunck_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)
