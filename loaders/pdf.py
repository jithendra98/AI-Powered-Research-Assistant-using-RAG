from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

def load_pdfs(path="./data/uploads"):
    loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    return loader.load()
