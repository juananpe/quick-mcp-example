# ChromaDB Document Query MCP Tool

This project provides a simple MCP (Machine Controlled Program) server that allows querying documents stored in a ChromaDB vector database using semantic search.

## Prerequisites

- Python 3.10 or later
- ChromaDB
- OpenAI API key (for embeddings)

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

4. Run the ChromaDB setup notebook to prepare your document database:
   ```bash
   jupyter notebook chroma_setup.ipynb
   ```
   
   This notebook demonstrates how to:
   - Load PDF documents
   - Split them into chunks
   - Embed them using OpenAI embeddings
   - Store them in a ChromaDB collection

## Running the Server

To start the MCP server:

```bash
python tool_mcp.py
```

## Using the Client

The included client.py provides an interactive way to query the document database:

```bash
python client.py
```

This will:
1. Connect to the MCP server
2. Show available tools
3. Display collection information
4. Enter an interactive query mode where you can search the document

## Available Tools

- `query_document`: Search for information in the document based on semantic similarity
- `get_collection_info`: Get information about the ChromaDB collection, including the number of stored documents

## Customization

You can modify the `chroma_setup.ipynb` notebook to:
- Change the source document(s)
- Adjust chunking parameters
- Select a different embedding model
- Create custom collections
