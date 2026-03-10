from langchain_community.document_loaders import TextLoader, DirectoryLoader

def load_txts(path="./data/uploads"):
    loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader)
    return loader.load()
