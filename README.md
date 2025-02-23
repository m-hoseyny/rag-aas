# RAG as a Service (RAGaaS)

A Flask-based API service that provides Retrieval-Augmented Generation (RAG) capabilities as a service. This project allows you to upload documents, create knowledge bases, and query them using natural language, leveraging the power of large language models and vector databases.

## Features

- 📚 Document Upload: Support for text and CSV file uploads via URLs
- 🔍 Vector Search: Utilizes PGVector for efficient similarity search
- 🤖 Multiple LLM Support: Compatible with OpenAI models and custom endpoints
- 🗄️ PostgreSQL Integration: Persistent storage for embeddings and chat history
- 📝 Chat History: Maintains conversation context and history
- 📊 Swagger Documentation: API documentation and testing interface

## Tech Stack

- **Backend Framework**: Flask
- **Database**: PostgreSQL with pgvector extension
- **Vector Embeddings**: OpenAI Embeddings (text-embedding-3-large)
- **LLM Integration**: LangChain
- **API Documentation**: Flasgger/Swagger
- **Containerization**: Docker

## Prerequisites

- Docker and Docker Compose
- OpenAI API Key or compatible endpoint
- PostgreSQL with pgvector extension

## Environment Variables

Copy `.env.example` to `.env` and configure the following variables:

```env
OPENAI_API_KEY=your_api_key
VECTOR_DB_USERNAME=postgres
VECTOR_DB_PASSWORD=postgres
VECTOR_DB_HOSTNAME=postgres
VECTOR_DB_DATABASE=ragaas
VECTOR_DB_PORT=5432
```

## Installation & Setup

1. Clone the repository
2. Copy `.env.example` to `.env` and configure your environment variables
3. Build and run the services:

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8443`

## API Documentation

The API documentation is available through Swagger UI at:
- `http://localhost:8443/apidocs`

You can use this interactive interface to:
- Explore all available endpoints
- Test API endpoints directly
- View request/response schemas
- Download OpenAPI specification

## API Endpoints

- `GET /`: Welcome endpoint with service status
- `POST /upload-file`: Upload documents to create knowledge base
  - Required parameters:
    - `collection_id`: Unique identifier for the document collection
    - `file_url`: URL of the file to process (supports .txt and .csv)
- `POST /chat-message`: Chat endpoint for querying the knowledge base
  - Supports conversation history and context

## Usage Example

1. Upload a document:
```bash
curl -X POST http://localhost:8443/upload-file \
  -H "Content-Type: application/json" \
  -d '{"collection_id": "my_docs", "file_url": "https://example.com/document.txt"}'
```

2. Query the knowledge base:
```bash
curl -X POST http://localhost:8443/chat-message \
  -H "Content-Type: application/json" \
  -d '{"collection_id": "my_docs", "user_input": "What does the document say about X?"}'
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
