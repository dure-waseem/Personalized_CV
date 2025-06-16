#app.py
import streamlit as st
from chromadb_query import ChromaCollection
from chromadb_upload import ChromaUploader
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@st.cache_resource
def get_chroma_collection(collection_name, db_path):
    """Cache the ChromaCollection to prevent re-initialization"""
    return ChromaCollection(collection_name, db_path)

@st.cache_resource
def get_chroma_uploader(collection_name, db_path):
    """Cache the ChromaUploader to prevent re-initialization"""
    return ChromaUploader(collection_name, db_path)

# Set page title and layout
st.set_page_config(page_title="ChromaDB Q&A", layout="wide")
st.title("ChromaDB Q&A Interface")

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ OpenAI API key not found! Please check your .env file.")
    st.stop()

db_path = "../db"  # Default database path
collection_name = "my_collection"

# Create tabs
tab1, tab2 = st.tabs(["📝 Q&A", "📄 Upload PDF"])

# Tab 1: Q&A Interface
with tab1:
    st.header("Ask Questions About Your Documents")
    
    try:
        # Use cached collection
        chroma_collection = get_chroma_collection(collection_name, db_path)
    except Exception as e:
        st.error(f"Error initializing ChromaDB collection: {str(e)}")
        st.stop()

    # User input for the query
    query = st.text_area("Enter your query:", placeholder="Ask me anything about your documents...")

    n_results = 10  # Default number of results

    # Submit button
    if st.button("Generate Answer", type="primary"):
        if query.strip():
            with st.spinner("Searching and generating answer..."):
                try:
                    # Query the collection for similar documents
                    results = chroma_collection.query_collection([query], n_results=n_results)
                    
                    # Generate an answer using the LLM
                    answer = chroma_collection.generate_answer(query, results)
                    
                    # Display the result
                    st.subheader("🤖 Generated Answer:")
                    st.write(answer)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("⚠️ Please enter a query.")

# Tab 2: Upload PDF Interface
with tab2:
    st.header("Upload PDF Documents to ChromaDB")
    
    try:
        # Use cached uploader
        chroma_uploader = get_chroma_uploader(collection_name, db_path)
        
        # Display current collection info
        # doc_count = chroma_uploader.get_collection_count()
   
        
    except Exception as e:
        st.error(f"Error initializing ChromaDB uploader: {str(e)}")
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type="pdf",
        help="Upload a PDF file to extract text and add to ChromaDB"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        st.markdown("---")
        
      
        
       
        process_btn = st.button(
            "Click to add documents to ChromaDB", 
            type="primary",
            use_container_width=True
        )
        
        
        if process_btn:
            with st.spinner("Processing PDF and adding to ChromaDB..."):
                try:
                    # Read the file bytes
                    pdf_bytes = uploaded_file.read()
                    
                    # Extract text and lines
                    pdf_text, pdf_lines = chroma_uploader.extract_text_from_pdf_bytes(pdf_bytes)
                    
                    if pdf_text and pdf_lines:
                        # Add documents to ChromaDB
                        success = chroma_uploader.add_documents(pdf_lines)
                        
                        if success:
                            st.success(f"✅ Successfully processed and added {len(pdf_lines)} document chunks to ChromaDB!")
                            
                            
                        else:
                            st.error("❌ Failed to add documents to ChromaDB.")
                    else:
                        st.error("❌ Could not extract text from the PDF file.")
                        
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
    
    else:
        st.info("Please upload a PDF file to get started.")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ Information")
    st.markdown("""
    ### How to use:
    
    **Q&A Tab:**
    - Ask questions about your uploaded documents
    - Get AI-powered answers based on your document content
    
    **Upload PDF Tab:**
    - Upload PDF files to add to your knowledge base
    - Documents are automatically chunked and stored into the database.
    
   
    """)
    
