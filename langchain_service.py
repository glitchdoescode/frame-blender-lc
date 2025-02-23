import os
import json
from langchain.docstore.document import Document

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


class LangChainService:
    def __init__(self):
        self.vector_store = None
        self.embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = SemanticChunker(self.embedder)
        self.llm = ChatOllama(model="llama3.2", temperature=1)

    def load_frame_documents(self, folder_path: str = "frame_json"):
        """
        Loads frame JSON files into LangChain Document objects.
        """
        docs = []
        if not os.path.isdir(folder_path):
            return docs
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    frame_name = data.get('frame_name', '')
                    frame_rel = data.get('fr_rel', {})
                    description = data.get('description', '')
                    content = (
                        f"Frame Name: {frame_name}\n"
                        f"Relations: {frame_rel}\n"
                        f"Description: {description}\n"
                    )
                    doc = Document(page_content=content, metadata={"source": filename})
                    docs.append(doc)
        return docs

    def create_vector_store(self, docs):
        """
        Creates a FAISS vector store from the loaded documents.
        """
        if not docs:
            return "Failed: No documents loaded"
        try:
            splitted_docs = self.text_splitter.split_documents(docs)
            if not splitted_docs:
                return "Failed: No splitted documents"
            self.vector_store = FAISS.from_documents(splitted_docs, self.embedder)
            return "Finished"
        except Exception as e:
            return f"Failed: {str(e)}"

    def build_rag_chain(self, prompt_template):
        """
        Builds a RAG chain with the vector store and prompt template.
        """
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized.")
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        stuff_chain = create_stuff_documents_chain(self.llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, stuff_chain)
        return rag_chain

    def generate_response(self, rag_chain, frames_str):
        """
        Generates a response using the RAG chain.
        """
        try:
            result = rag_chain.invoke({"input": frames_str})
            return result["answer"]
        except Exception as e:
            return f"Error during generation: {str(e)}"

    def load_packages(self):
        """
        Loads frame documents and creates a vector store.
        """
        docs = self.load_frame_documents()
        return self.create_vector_store(docs)