from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def initialize_splitter(chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter


def load_split_pdf_file(pdf_file, text_splitter) -> list[Document]:
    documents = []
    try:
        loader = PyPDFLoader(pdf_file)
        print(f"Loading {pdf_file}")
        documents.extend(loader.load_and_split(text_splitter=text_splitter))
        print(f"Loaded {pdf_file}")
    except Exception as e:
        print(f"Error processing {pdf_file}: {str(e)}")
    return documents
