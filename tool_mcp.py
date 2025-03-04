from typing import Any, List, Dict
import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Verify API key is set
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY is not set in environment variables")
    print("Document embedding and semantic search will not work properly")

# Initialize FastMCP server
mcp = FastMCP("document-search")

# Initialize ChromaDB client
try:
    client = chromadb.PersistentClient(path="./chroma_db")
    print("Successfully connected to ChromaDB")
except Exception as e:
    print(f"Error initializing ChromaDB client: {e}")
    client = None

# Get the OpenAI embedding function
try:
    if OPENAI_API_KEY:
        embedding_function = OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-3-small"
        )
        print("Successfully initialized OpenAI embedding function")
    else:
        embedding_function = None
        print("No OpenAI API key provided - embedding function not initialized")
except Exception as e:
    print(f"Error initializing embedding function: {e}")
    embedding_function = None

# Get the collection (assuming it's already created by chroma_setup.ipynb)
collection = None
try:
    if client and embedding_function:
        collection = client.get_collection(
            name="pdf_collection",
            embedding_function=embedding_function
        )
        print(f"Successfully connected to collection with {collection.count()} documents")
    else:
        print("Cannot connect to collection: Client or embedding function not initialized")
except Exception as e:
    print(f"Warning: Error connecting to collection: {e}")
    print("Please run chroma_setup.ipynb first to create the collection.")
    # We'll continue running, but the tools will return appropriate error messages
    # instead of crashing the server

def format_search_result(document: str, distance: float, metadata: Dict[str, Any] = None) -> str:
    """Format a search result into a readable string."""
    result = f"Score: {1 - distance:.4f} (closer to 1 is better)\n"
    
    if metadata:
        page_num = metadata.get('page', 'Unknown')
        result += f"Page: {page_num}\n"
    
    result += f"Content: {document}"
    return result

@mcp.tool()
async def query_document(query_text: str, num_results: int = 5) -> str:
    """Search for information in the document based on semantic similarity.

    Args:
        query_text: The search query text
        num_results: Number of results to return (default: 5)
    """
    try:
        # Verify that ChromaDB and collection are properly initialized
        if not collection:
            return "Error: ChromaDB collection is not initialized. Please run chroma_setup.ipynb first."
            
        # Query the collection
        results = collection.query(
            query_texts=[query_text],
            n_results=num_results
        )
        
        # Process results
        if not results or 'documents' not in results or not results['documents'][0]:
            return "No results found for your query."
        
        formatted_results = []
        for i, (doc, distance, metadata) in enumerate(zip(
            results['documents'][0], 
            results['distances'][0],
            results['metadatas'][0] if 'metadatas' in results else [{}] * len(results['documents'][0])
        )):
            formatted_results.append(f"Result {i+1}:\n{format_search_result(doc, distance, metadata)}")
        
        return "\n\n---\n\n".join(formatted_results)
    
    except Exception as e:
        error_message = f"Error querying document: {str(e)}"
        print(error_message)  # Log to console for debugging
        return error_message  # Return error as plain text, not HTML

@mcp.tool()
async def get_collection_info() -> str:
    """Get information about the ChromaDB collection."""
    try:
        if not collection:
            return "Error: ChromaDB collection is not initialized. Please run chroma_setup.ipynb first."
            
        count = collection.count()
        return f"Collection name: pdf_collection\nNumber of documents: {count}"
    except Exception as e:
        error_message = f"Error getting collection info: {str(e)}"
        print(error_message)
        return error_message

@mcp.tool()
async def diagnostic_info() -> str:
    """Get diagnostic information about the ChromaDB setup."""
    info = []
    
    # Check if collection exists
    try:
        if not collection:
            info.append("❌ ChromaDB collection is not initialized")
        else:
            count = collection.count()
            info.append(f"✅ ChromaDB collection is initialized with {count} documents")
    except Exception as e:
        info.append(f"❌ Error accessing ChromaDB collection: {str(e)}")
    
    # Check OpenAI API key
    if not OPENAI_API_KEY:
        info.append("❌ OpenAI API key is not set in environment variables")
    else:
        info.append("✅ OpenAI API key is set")
    
    # Check PDF file
    pdf_path = "./testing/ft_guide.pdf"
    if os.path.exists(pdf_path):
        info.append(f"✅ PDF file exists at {pdf_path}")
    else:
        info.append(f"❌ PDF file not found at {pdf_path}")
    
    return "\n".join(info)

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')