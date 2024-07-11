import PyPDF2
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    return pages


def split_text(pages):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_documents(pages)
    return texts


def create_embeddings(texts):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docsearch = FAISS.from_documents(texts,embeddings)
    return docsearch


def setup_qa_pipeline():
    model_name = "roberta-large"  # You can choose other models like "roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline


def answer_query(query, docsearch, qa_pipeline):
    # Retrieve relevant documents
    docs = docsearch.similarity_search(query)

    # Combine documents into a single context for the QA pipeline
    context = " ".join([doc["text"] for doc in docs])  # Adjust based on how documents are formatted

    # Perform QA
    result = qa_pipeline(question=query, context=context)
    return result["answer"]


def main():
    # Load PDF
    pdf_path = "attention.pdf"
    pages = load_pdf(pdf_path)

    # Process text
    texts = split_text(pages)

    # Create embeddings and vector store
    docsearch = create_embeddings(texts)

    # Set up QA pipeline
    qa_pipeline = setup_qa_pipeline()

    # User interaction loop
    while True:
        query = input("Enter your question about the PDF (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break

        answer = answer_query(query, docsearch, qa_pipeline)
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
