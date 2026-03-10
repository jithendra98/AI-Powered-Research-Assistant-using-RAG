from loaders.pdf import load_pdfs
from loaders.txt import load_txts
from loaders.doc import load_docs
#from loaders.web import load_webpages  # only if you need to load URLs

def custom_loader(path: str):
    docs = []

    # Load all types of documents
    docs.extend(load_pdfs(path))
    docs.extend(load_txts(path))
    docs.extend(load_docs(path))
    # docs.extend(load_webpages(["https://example.com"]))  # Optional: pass list of URLs

    print(f"📄 Loaded {len(docs)} documents from {path}")
    return docs
