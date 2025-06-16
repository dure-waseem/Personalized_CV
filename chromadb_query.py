#chromadb_query.py
import chromadb
import pdfplumber  # Library to extract text from PDF
import time
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
import os
from openai import OpenAI  # Updated import

load_dotenv()

class ChromaCollection:
    def __init__(self, collection_name, db_path):
        # Initialize Chroma persistent client and collection name
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection_name = collection_name
        self.collection = None
        self.openai_ef = os.getenv("OPENAI_API_KEY")
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.openai_ef,
            model_name="text-embedding-ada-002"
        )
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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

    def query_collection(self, query_texts, n_results=1):
        """
        Queries the collection with the given text and returns the results.
        :param query_texts: List of query strings
        :param n_results: Number of results to return
        :return: Query results
        """
        results = self.collection.query(
            query_texts=query_texts,  # Chroma will embed this for you
            n_results=n_results  # How many results to return
        )
        return results

    def generate_answer(self, query, results):
        """
        Takes the query and ChromaDB results and generates an accurate answer using the LLM.
        :param query: User's query
        :param results: ChromaDB results
        :return: Generated answer from LLM
        """
        # Prepare the context for LLM by appending the query and results
        context = f"User's Query: {query}\n\nContext from ChromaDB:\n{results['documents']}\n\nJust answer to the asked query without any explanation or context.\n\nAnswer:"
        
        try:
            # Use the new OpenAI API format
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Updated model name
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": context}
                ],
                max_tokens=150,
                # temperature=0.7
            )
            
            # Extract and return the answer from the response using the new API structure
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"