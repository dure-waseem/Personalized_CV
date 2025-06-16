import chromadb
import PyPDF2  
import time
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
import os
import io
import streamlit as st

load_dotenv()

class ChromaUploader:
    def __init__(self, collection_name, db_path):
        # Initialize Chroma persistent client and collection name
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection_name = collection_name
        self.collection = None
        # self.openai_ef = os.getenv("OPENAI_API_KEY")
        # self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        #     api_key=self.openai_ef,
        #     model_name="text-embedding-ada-002"
        # )
        self.openai_ef = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
        # self.openai_ef = os.getenv("OPENAI_API_KEY")
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.openai_ef,
            model_name="text-embedding-ada-002"
        )
        self._initialize_collection()
        
    def _initialize_collection(self):
        """
        Initializes the collection if it doesn't exist.
        """
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name, embedding_function=self.openai_ef)
            print(f"Collection '{self.collection_name}' already exists.")
        except ValueError:
            # If collection doesn't exist, create a new one
            self.collection = self.chroma_client.create_collection(name=self.collection_name, embedding_function=self.openai_ef)
            print(f"Created new collection '{self.collection_name}'.")

    def add_documents(self, documents):
        """
        Adds documents to the collection, ensuring no duplicate IDs.
        :param documents: List of document strings to be added
        """
        if documents is None or len(documents) == 0:
            print("No data collected from the document to add.")
            return False

        timestamp = int(time.time() * 1000000)  # microseconds for uniqueness
        ids = [f"doc_{timestamp}_{i}" for i in range(len(documents))]
        # Add documents to collection
        self.collection.add(
            documents=documents,
            ids=ids
        )
        print(f"Added {len(documents)} documents to collection '{self.collection_name}'.")
        return True

    def extract_text_from_pdf_bytes(self, pdf_bytes):
        """
        Extracts text from a PDF file from bytes (for Streamlit uploaded files).
        :param pdf_bytes: PDF file as bytes
        :return: Extracted text from the PDF and the lines as a list
        """
        try:
            # Create a file-like object from bytes
            pdf_file = io.BytesIO(pdf_bytes)
            
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Initialize an empty string to store extracted text
            text = ""
            
            # Extract text from each page
            for page in pdf_reader.pages:
                # Extract text from the page
                page_text = page.extract_text()
                
                # Clean up the extracted text
                cleaned_text = self._clean_extracted_text(page_text)
                
                # Append to the total text
                text += cleaned_text + "\n\n"
            
            # Create the list of pdf lines
            pdf_lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            return text.strip(), pdf_lines
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return "", []

    def extract_text_from_pdf(self, pdf_path):
        """
        Extracts text from a PDF file using PyPDF2 with improved text extraction.
        :param pdf_path: Path to the PDF file
        :return: Extracted text from the PDF and the lines as a list
        """
        try:
            # Open the PDF file
            with open(pdf_path, 'rb') as file:
                # Create a PDF reader object
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Initialize an empty string to store extracted text
                text = ""
                
                # Extract text from each page
                for page in pdf_reader.pages:
                    # Extract text from the page
                    page_text = page.extract_text()
                    
                    # Clean up the extracted text
                    cleaned_text = self._clean_extracted_text(page_text)
                    
                    # Append to the total text
                    text += cleaned_text + "\n\n"
                
                # Create the list of pdf lines
                pdf_lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                return text.strip(), pdf_lines
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return "", []

    def _clean_extracted_text(self, text):
        """
        Clean up extracted text to improve readability and remove unnecessary whitespace.
        :param text: Raw extracted text
        :return: Cleaned text
        """
        # Remove excessive whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Join lines, preserving some structure
        cleaned_text = ' '.join(lines)
        
        return cleaned_text

    def get_collection_count(self):
        """
        Get the number of documents in the collection.
        """
        try:
            return self.collection.count()
        except Exception as e:
            print(f"Error getting collection count: {e}")
            return 0