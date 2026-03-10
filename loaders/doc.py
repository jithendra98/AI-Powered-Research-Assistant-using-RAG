from langchain_community.document_loaders import UnstructuredWordDocumentLoader, DirectoryLoader

def load_docs(path="./data/uploads"):
    loader = DirectoryLoader(path, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader)
    return loader.load()
