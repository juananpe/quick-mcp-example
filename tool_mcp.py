import os
import logging
from dotenv import load_dotenv
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    GetPromptResult,
    Prompt,
    PromptMessage,
    PromptArgument
)
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import glob
from importlib import metadata
import time
import re

from langchain_community.document_loaders import PyPDFLoader


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("document-search-mcp")

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize server
server = Server("document-search")

# Initialize ChromaDB client
client = None
embedding_function = None
collection = None

try:
    client = chromadb.PersistentClient(path="./chroma_db")
    logger.info("Successfully connected to ChromaDB")
    
    # Initialize OpenAI embedding function
    if OPENAI_API_KEY:
        embedding_function = OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-3-small"
        )
        logger.info("Successfully initialized OpenAI embedding function")
        
        # Get the collection
        collection = client.get_collection(
            name="pdf_collection",
            embedding_function=embedding_function
        )
        logger.info(f"Successfully connected to collection with {collection.count()} documents")
    else:
        logger.warning("OpenAI API key is not set - embedding function not initialized")
except Exception as e:
    logger.error(f"Error initializing components: {e}")


def format_search_result(document: str, distance: float, metadata: dict[str, object] = None) -> str:
    """Format a search result into a readable string."""
    result = f"Score: {1 - distance:.4f} (closer to 1 is better)\n"
    
    if metadata:
        page_num = metadata.get('page', 'Unknown')
        result += f"Page: {page_num}\n"
    
    result += f"Content: {document}"
    return result


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools in the server."""
    return [
        Tool(
            name="query_document",
            description="Search for information in the document based on semantic similarity",
            inputSchema={
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "The search query text"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)"
                    }
                },
                "required": ["query_text"]
            }
        ),
        Tool(
            name="get_collection_info",
            description="Get information about the ChromaDB collection",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="diagnostic_info",
            description="Get diagnostic information about the ChromaDB setup",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool execution requests."""
    
    if name == "query_document":
        query_text = arguments.get("query_text", "")
        num_results = arguments.get("num_results", 5)
        
        try:
            # Verify that ChromaDB and collection are properly initialized
            if not collection:
                return [TextContent(
                    type="text",
                    text="Error: ChromaDB collection is not initialized. Please run chroma_setup.ipynb first."
                )]
                
            # Query the collection
            results = collection.query(
                query_texts=[query_text],
                n_results=num_results
            )
            
            # Process results
            if not results or 'documents' not in results or not results['documents'][0]:
                return [TextContent(type="text", text="No results found for your query.")]
            
            formatted_results = []
            for i, (doc, distance, metadata) in enumerate(zip(
                results['documents'][0], 
                results['distances'][0],
                results['metadatas'][0] if 'metadatas' in results else [{}] * len(results['documents'][0])
            )):
                formatted_results.append(f"Result {i+1}:\n{format_search_result(doc, distance, metadata)}")
            
            return [TextContent(
                type="text",
                text="\n\n---\n\n".join(formatted_results)
            )]
        
        except Exception as e:
            error_message = f"Error querying document: {str(e)}"
            logger.error(error_message)
            return [TextContent(type="text", text=error_message)]
    
    elif name == "get_collection_info":
        try:
            if not collection:
                return [TextContent(
                    type="text",
                    text="Error: ChromaDB collection is not initialized. Please run chroma_setup.ipynb first."
                )]
                
            count = collection.count()
            return [TextContent(
                type="text",
                text=f"Collection name: pdf_collection\nNumber of documents: {count}"
            )]
        except Exception as e:
            error_message = f"Error getting collection info: {str(e)}"
            logger.error(error_message)
            return [TextContent(type="text", text=error_message)]
    
    elif name == "diagnostic_info":
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
        
        return [TextContent(type="text", text="\n".join(info))]
    
    else:
        raise ValueError(f"Unknown tool: {name}")


@server.list_resources()
async def handle_list_resources() -> list[Resource]:
    """List all available document resources"""
    resources = []
    
    # Add known PDF resources
    resources.append(
        Resource(
            uri="document://pdf/ft_guide",
            name="Fine-tuning Guide",
            description="OpenAI Fine-tuning Guide PDF",
            mimeType="application/pdf"
        )
    )
    
    resources.append(
        Resource(
            uri="document://pdf/bert",
            name="BERT Guide",
            description="BERT Paper PDF",
            mimeType="application/pdf"
        )
    )
    
    # Find all available PDFs in testing directory
    try:
        pdf_files = glob.glob("./testing/*.pdf")
        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            # Skip PDFs we already explicitly added
            if name_without_ext in ["ft_guide", "bert"]:
                continue
                
            resources.append(
                Resource(
                    uri=f"document://pdf/{name_without_ext}",
                    name=name_without_ext.replace('_', ' ').title(),
                    description=f"PDF Document: {name_without_ext}",
                    mimeType="application/pdf"
                )
            )
    except Exception as e:
        logger.error(f"Error scanning for PDF files: {e}")
    
    # Add chunk resources if we have a collection
    if collection:
        try:
            # Add a few sample chunks as resources
            chunks = collection.get(limit=10)
            if chunks and 'ids' in chunks and chunks['ids']:
                for i, chunk_id in enumerate(chunks['ids']):
                    resources.append(
                        Resource(
                            uri=f"document://chunk/{chunk_id}",
                            name=f"Document Chunk {chunk_id}",
                            description=f"Document chunk from ChromaDB collection",
                            mimeType="text/plain"
                        )
                    )
        except Exception as e:
            logger.error(f"Error getting chunks from collection: {e}")
    
    return resources


@server.read_resource()
async def handle_read_resource(uri: str):
    """Handle reading resources from various sources"""
    
    if not str(uri).startswith("document://"):
        raise ValueError(f"Unsupported URI scheme: {uri}")
    
    path_parts = str(uri).split("/")
    if len(path_parts) < 3:
        raise ValueError(f"Invalid URI format: {uri}")
    
    resource_type = path_parts[2]  # pdf or chunk
    
    if resource_type == "pdf":
        if len(path_parts) < 4:
            raise ValueError(f"Missing document name in URI: {uri}")
        
        document_name = path_parts[3]
        
        try:            
            # Construct path - assuming documents are in ./testing/
            path = f"./testing/{document_name}"
            if not path.endswith('.pdf'):
                path += '.pdf'
                
            # Load the document
            loader = PyPDFLoader(path)
            pages = loader.load()
            
            # Combine all pages into one text with page markers
            full_text = ""
            for i, page in enumerate(pages):
                full_text += f"\n\n--- Page {i+1} ---\n\n"
                full_text += page.page_content
            logger.info(f"Loaded document {document_name} with {len(pages)} pages. Preview: {full_text[:200]}...")
            return full_text
        except Exception as e:
            error_message = f"Error loading document: {str(e)}"
            logger.error(error_message)
            return error_message
    
    elif resource_type == "chunk":
        if len(path_parts) < 4:
            raise ValueError(f"Missing chunk ID in URI: {uri}")
        
        chunk_id = path_parts[3]
        
        try:
            if not collection:
                return "Error: ChromaDB collection is not initialized."
                
            result = collection.get(ids=[chunk_id])
            
            if not result or not result['documents'] or len(result['documents']) == 0:
                return f"Chunk with ID {chunk_id} not found."
                
            document = result['documents'][0]
            metadata = result['metadatas'][0] if 'metadatas' in result and result['metadatas'] else {}
            
            output = "--- Chunk Details ---\n"
            output += f"ID: {chunk_id}\n"
            
            if metadata:
                for key, value in metadata.items():
                    output += f"{key.capitalize()}: {value}\n"
                    
            output += "\n--- Content ---\n"
            output += document
            
            return output
        except Exception as e:
            error_message = f"Error retrieving chunk: {str(e)}"
            logger.error(error_message)
            return error_message
    
    else:
        raise ValueError(f"Unsupported resource type: {resource_type}")


@server.list_prompts()
async def handle_list_prompts() -> list[Prompt]:
    """List available prompts from the server"""
    return [
        Prompt(
            name="deep_analysis",
            description="Perform deep analysis on document sections",
            arguments=[
                PromptArgument(
                    name="query",
                    description="What aspect to analyze (e.g., 'main themes', 'methodology')",
                    required=True
                )
            ]
        ),
        Prompt(
            name="extract_key_information",
            description="Extract specific types of information from document sections",
            arguments=[
                PromptArgument(
                    name="info_type",
                    description="Type of information to extract (definitions, people, statistics, processes, arguments)",
                    required=True
                )
            ]
        )
    ]


@server.get_prompt()
async def handle_get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
    """Generate a prompt based on the requested type"""
    # Simple argument handling
    if arguments is None:
        arguments = {}
        
    if name == "deep_analysis":
        # Get query with a fallback default
        query = arguments.get("query", "main themes")
        
        return GetPromptResult(
            description=f"Deep analysis focusing on {query}",
            messages=[
                PromptMessage(
                    role="assistant", 
                    content=TextContent(
                        type="text",
                        text="I am a document analysis expert specializing in identifying key themes, arguments, and evidence in academic and technical documents."
                    )
                ),
                PromptMessage(
                    role="user", 
                    content=TextContent(
                        type="text",
                        text=f"""Please perform a deep analysis of the document section provided in the conversation, focusing on {query}.

Include in your analysis:
- Main themes and arguments presented
- Key evidence and supporting details
- Logical structure and flow of information
- Implicit assumptions made in the text
- Strengths and weaknesses of the arguments
- Connections to broader context if applicable

Format your analysis in a well-structured manner with clear headings and concise explanations.
"""
                    )
                )
            ]
        )
    
    elif name == "extract_key_information":
        # Get info_type with a fallback default
        info_type = arguments.get("info_type", "definitions")
        
        info_instructions = {
            "definitions": "Extract and explain all technical definitions, terminology, and concepts",
            "people": "Extract all people, organizations, and entities mentioned with their relevance",
            "statistics": "Extract all numerical data, statistics, and quantitative information",
            "processes": "Extract all processes, procedures, workflows, and methodologies described",
            "arguments": "Extract the main arguments, claims, and supporting evidence"
        }
        
        instruction = info_instructions.get(
            info_type.lower(),
            f"Extract all {info_type} mentioned in the document"
        )
        
        return GetPromptResult(
            description=f"Extracting {info_type} from document",
            messages=[
                PromptMessage(
                    role="assistant", 
                    content=TextContent(
                        type="text",
                        text="I am a precise information extraction specialist with expertise in technical documents."
                    )
                ),
                PromptMessage(
                    role="user", 
                    content=TextContent(
                        type="text",
                        text=f"""Based on the document section provided in the conversation, please {instruction}.

Format your response as a structured list with:
1. Clear headers for each extracted element
2. Direct quotes or references when applicable
3. Brief explanations of significance where helpful
4. Page or section references if available

Be comprehensive but focus on quality over quantity.
"""
                    )
                )
            ]
        )
    
    else:
        raise ValueError(f"Unknown prompt: {name}")


async def main():
    """Run the MCP server using stdin/stdout streams"""
    # Get the distribution info for versioning
    try:
        dist = metadata.distribution("document-search-mcp")
        version = dist.version
    except:
        version = "0.1.0"
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="document-search-mcp",
                server_version=version,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())