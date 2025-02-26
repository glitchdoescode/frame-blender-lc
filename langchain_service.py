# langchain_service.py
import os
import json
from langchain.docstore.document import Document
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from typing import List, Dict
from config import FRAME_JSON_DIR
from dotenv import load_dotenv
from utils import logs, log_message  # Import logs and log_message from utils.py
import asyncio

class LangChainService:
    def __init__(self, vector_store_path: str):
        self.vector_store = None
        self.vector_store_path = vector_store_path
        self.embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = SemanticChunker(self.embedder)
        self.llm = ChatOllama(model="llama3.2", temperature=1)
        log_message("INFO - LangChainService initialized with embedder and LLM.")

    def load_frame_documents(self, folder_path: str = FRAME_JSON_DIR) -> List[Document]:
        """
        Loads frame JSON files into LangChain Document objects, including all fields.
        """
        docs = []
        if not os.path.isdir(folder_path):
            log_message(f"WARNING - Folder {folder_path} does not exist or is not a directory.")
            return docs
        
        log_message(f"INFO - Loading frame documents from {folder_path}")
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                filepath = os.path.join(folder_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        frame_name = data.get('frame_name', '')
                        frame_def = data.get('frame_def', '')
                        fe_def = data.get('fe_def', {})
                        lexical = data.get('lexical', {})
                        examples = data.get('examples', {})
                        fr_rel = data.get('fr_rel', {})
                        content = (
                            f"Frame Name: {frame_name}\n"
                            f"Definition: {frame_def}\n"
                            f"Frame Elements: {json.dumps(fe_def, indent=2)}\n"
                            f"Lexical Units: {json.dumps(lexical, indent=2)}\n"
                            f"Examples: {json.dumps(examples, indent=2)}\n"
                            f"Relations: {json.dumps(fr_rel, indent=2)}\n"
                        )
                        doc = Document(page_content=content, metadata={"source": filename})
                        docs.append(doc)
                        log_message(f"DEBUG - Loaded document: {filename} with frame_name: {frame_name}")
                except Exception as e:
                    log_message(f"ERROR - Failed to load {filename}: {str(e)}")
        log_message(f"INFO - Loaded {len(docs)} frame documents from {folder_path}")
        return docs

    def create_vector_store(self, docs: List[Document]) -> str:
        """
        Creates a FAISS vector store from the loaded documents and saves it to disk.
        """
        if not docs:
            log_message("WARNING - No documents provided to create vector store.")
            return "Failed: No documents loaded"
        
        log_message(f"INFO - Creating vector store with {len(docs)} documents")
        try:
            splitted_docs = self.text_splitter.split_documents(docs)
            if not splitted_docs:
                log_message("WARNING - No documents after splitting.")
                return "Failed: No splitted documents"
            log_message(f"DEBUG - Split documents into {len(splitted_docs)} chunks")
            self.vector_store = FAISS.from_documents(splitted_docs, self.embedder)
            os.makedirs(self.vector_store_path, exist_ok=True)
            self.vector_store.save_local(self.vector_store_path)
            log_message(f"INFO - Vector store created and saved to {self.vector_store_path}")
            return "Finished"
        except Exception as e:
            log_message(f"ERROR - Failed to create vector store: {str(e)}")
            return f"Failed: {str(e)}"

    def load_vector_store(self) -> bool:
        """
        Loads a previously saved vector store from disk if it exists.
        Returns True if loaded, False otherwise.
        """
        if os.path.exists(self.vector_store_path) and os.path.isdir(self.vector_store_path):
            try:
                self.vector_store = FAISS.load_local(self.vector_store_path, self.embedder, allow_dangerous_deserialization=True)
                log_message(f"INFO - Vector store loaded from {self.vector_store_path}")
                return True
            except Exception as e:
                log_message(f"ERROR - Failed to load vector store from {self.vector_store_path}: {str(e)}")
                return False
        log_message(f"DEBUG - No vector store found at {self.vector_store_path}")
        return False

    def build_rag_chain(self, prompt_template):
        """
        Builds a RAG chain with the vector store and prompt template.
        """
        if self.vector_store is None:
            log_message("ERROR - Vector store is not initialized.")
            raise ValueError("Vector store is not initialized.")
        
        log_message("INFO - Building RAG chain")
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        stuff_chain = create_stuff_documents_chain(self.llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, stuff_chain)
        log_message("INFO - RAG chain built successfully")
        return rag_chain

    async def generate_response_async(self, rag_chain, input_data: Dict) -> str:
        """
        Asynchronously generates a response using the RAG chain.
        """
        log_message(f"INFO - Generating response asynchronously with input: {json.dumps(input_data, indent=2)}")
        try:
            # Assume rag_chain.invoke is synchronous; wrap in asyncio.create_task or use async version if available
            result = await asyncio.create_task(rag_chain.ainvoke(input_data))  # Use .ainvoke if available
            log_message(f"DEBUG - Generated response: {result['answer'][:100]}...")  # Truncate for brevity
            return result["answer"]
        except Exception as e:
            log_message(f"ERROR - Error during async response generation: {str(e)}")
            return f"Error during generation: {str(e)}"

    def generate_response(self, rag_chain, input_data: Dict) -> str:
        """
        Generates a response using the RAG chain (synchronous for other uses).
        """
        log_message(f"INFO - Generating response with input: {json.dumps(input_data, indent=2)}")
        try:
            result = rag_chain.invoke(input_data)
            log_message(f"DEBUG - Generated response: {result['answer'][:100]}...")  # Truncate for brevity
            return result["answer"]
        except Exception as e:
            log_message(f"ERROR - Error during response generation: {str(e)}")
            return f"Error during generation: {str(e)}"

    def load_packages(self):
        """
        Loads frame documents and creates or loads a vector store.
        """
        log_message("INFO - Starting load_packages")
        if self.load_vector_store():
            log_message("INFO - Using existing vector store")
            return "Loaded"
        
        docs = self.load_frame_documents()
        result = self.create_vector_store(docs)
        log_message(f"INFO - load_packages completed with result: {result}")
        return result