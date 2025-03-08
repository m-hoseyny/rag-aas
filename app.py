from flask import Flask, jsonify, request
from db import BotSetting, engine, Chat
from sqlalchemy.orm import Session
import requests
import pandas as pd
import tempfile
import os
from rag import feed_data, feed_data_batch, retrieve, local_response, dify_response
from flasgger import Swagger, swag_from
import logging
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)


app = Flask(__name__)
swagger = Swagger(app, template={
    "info": {
        "title": "RAG as a Service API",
        "description": "API for managing and querying RAG (Retrieval-Augmented Generation) services",
        "version": "1.0.0",
        "contact": {
            "email": "your-email@example.com"
        }
    },
    "schemes": ["http", "https"]
})

@app.route('/')
@swag_from({
    "responses": {
        200: {
            "description": "Welcome message with latest chats",
            "schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "message": {"type": "string"},
                }
            }
        }
    }
})
def index():
    with Session(engine) as session:
        
        return jsonify({
            'status': 'success',
            'message': 'Welcome to RAG as a Service API',
        })

@app.route('/feed', methods=['POST'])
@swag_from({
    "parameters": [
        {
            "name": "body",
            "in": "body",
            "required": True,
            "schema": {
                "type": "object",
                "properties": {
                    "collection_id": {"type": "string", "description": "Collection ID to store the documents"},
                    "data": {
                        "type": "array", 
                        "description": "List of documents to add to the collection",
                        "items": {
                            "type": "object",
                            "properties": {
                                "page_content": {"type": "string", "description": "The content to be stored and retrieved"},
                                "metadata": {"type": "object", "description": "Additional metadata associated with the content"}
                            },
                            "required": ["page_content"]
                        }
                    }
                },
                "required": ["collection_id", "data"]
            }
        }
    ],
    "responses": {
        200: {
            "description": "Data successfully added to the collection",
            "schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "message": {"type": "string"},
                    "ids": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        400: {
            "description": "Invalid input",
            "schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "message": {"type": "string"}
                }
            }
        }
    }
})
def feed():
    try:
        data = request.get_json()
        if not data or 'collection_id' not in data or 'data' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields: collection_id or data'
            }), 400

        collection_id = data['collection_id']
        documents = data['data']
        
        if not isinstance(documents, list) or len(documents) == 0:
            return jsonify({
                'status': 'error',
                'message': 'Data must be a non-empty list of documents'
            }), 400
        
        # Validate each document in the list
        for i, doc in enumerate(documents):
            if 'page_content' not in doc:
                return jsonify({
                    'status': 'error',
                    'message': f'Document at index {i} is missing required field: page_content'
                }), 400
            if 'metadata' not in doc:
                documents[i]['metadata'] = {}
        
        logger.info(f"Feeding {len(documents)} documents to collection: {collection_id}")
        
        # Use the feed_data_batch function to add the data to the database
        result = feed_data_batch(
            documents=documents,
            collection_name=collection_id
        )
        
        return jsonify({
            'status': 'success',
            'message': f'{len(documents)} documents successfully added to collection {collection_id}',
            'ids': result
        })

    except Exception as e:
        logger.error(f"Error feeding data: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/retrieval', methods=['POST'])
@swag_from({
    "parameters": [
        {
            "name": "body",
            "in": "body",
            "required": True,
            "schema": {
                "type": "object",
                "properties": {
                    "knowledge_id": {"type": "string", "description": "Knowledge/Collection ID to search in"},
                    "collection_id": {"type": "string", "description": "Alternative name for knowledge_id"},
                    "query": {"type": "string", "description": "The query text to search for"},
                    "retrieval_setting": {
                        "type": "object",
                        "properties": {
                            "top_k": {"type": "integer", "description": "Maximum number of documents to return", "default": 5},
                            "score_threshold": {"type": "number", "description": "Minimum similarity score threshold", "default": 0.5}
                        }
                    }
                },
                "required": ["query"]
            }
        }
    ],
    "responses": {
        200: {
            "description": "Retrieved documents based on the query",
            "schema": {
                "type": "object",
                "properties": {
                    "records": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "metadata": {"type": "object", "description": "Metadata associated with the document"},
                                "score": {"type": "number", "description": "Similarity score between the query and document"},
                                "title": {"type": "string", "description": "Document title or ID"},
                                "content": {"type": "string", "description": "Document content"}
                            }
                        }
                    }
                }
            }
        },
        400: {
            "description": "Invalid input",
            "schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "message": {"type": "string"}
                }
            }
        }
    }
})
def retrieval():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing required field: query'
            }), 400

        # Accept either knowledge_id or collection_id
        collection_id = data.get('knowledge_id') or data.get('collection_id')
        if not collection_id:
            return jsonify({
                'status': 'error',
                'message': 'Missing required field: knowledge_id or collection_id'
            }), 400

        query = data['query']
        retrieval_setting = data.get('retrieval_setting', {})
        
        # Extract retrieval settings with defaults
        top_k = int(retrieval_setting.get('top_k', 5))
        score_threshold = float(retrieval_setting.get('score_threshold', 0.5))
        
        logger.info(f"Retrieving documents from collection '{collection_id}' for query: '{query}'")
        
        # Use the retrieve function to get documents
        records = retrieve(
            collection_name=collection_id,
            query=query,
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        return jsonify({
            'records': records
        })

    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/upload-file', methods=['POST'])
@swag_from({
    "parameters": [
        {
            "name": "body",
            "in": "body",
            "required": True,
            "schema": {
                "type": "object",
                "properties": {
                    "collection_id": {"type": "string", "description": "Collection ID to store the documents"},
                    "file_url": {"type": "string", "description": "URL of the file to process (must be .txt or .csv)"}
                },
                "required": ["collection_id", "file_url"]
            }
        }
    ],
    "responses": {
        200: {
            "description": "File processed successfully",
            "schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "message": {"type": "string"}
                }
            }
        },
        400: {
            "description": "Invalid input",
            "schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "message": {"type": "string"}
                }
            }
        }
    }
})
def upload_file():
    try:
        data = request.get_json()
        if not data or 'collection_id' not in data or 'file_url' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields: collection_id or file_url'
            }), 400

        collection_id = data['collection_id']
        file_url = data['file_url']
        print(f"Processing URL: {file_url}")

        # Download file with streaming and get filename from headers
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            
            # Try to get filename from Content-Disposition header
            content_disposition = r.headers.get('Content-Disposition')
            if content_disposition and 'filename=' in content_disposition:
                original_filename = content_disposition.split('filename=')[-1].strip('"\'')
            else:
                # Fallback to URL path
                url_path = requests.utils.urlparse(file_url).path
                original_filename = os.path.basename(url_path)
                if not original_filename:
                    original_filename = 'downloaded_file'
            
            print(f"Original Filename: {original_filename}")
            
            file_extension = os.path.splitext(original_filename)[1].lower().lstrip('.')
            if not file_extension:
                file_extension = file_url.split('.')[-1].lower()
            print(f"File Extension: {file_extension}")

            if file_extension not in ['txt', 'csv']:
                return jsonify({
                    'status': 'error',
                    'message': 'Only txt and csv files are supported'
                }), 400

            # Create a temporary file with the original filename
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, original_filename)
            
            print(f"Downloading to: {temp_file_path}")
            
            with open(temp_file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        try:
            # Process the file based on its type
            if file_extension == 'csv':
                df = pd.read_csv(temp_file_path)
                if 'question' not in df.columns or 'answer' not in df.columns:
                    return jsonify({
                        'status': 'error',
                        'message': 'CSV must contain "question" and "answer" columns'
                    }), 400
                
                for _, row in df.iterrows():
                    print(_)
                    feed_data(
                        page_content=row['question'],
                        metadata={'answer': row['answer'], 
                                'question': row['question'],
                                'file_url': file_url,
                                'original_filename': original_filename},
                        collection_name=collection_id
                    )
            else:  # txt file
                with open(temp_file_path, 'r') as file:
                    content = file.read()
                    documents = content.split('\n\n')
                    for doc in documents:
                        if doc.strip():  # Skip empty documents
                            feed_data(
                                page_content=doc.strip(),
                                metadata={
                                    'file_url': file_url,
                                    'original_filename': original_filename
                                },
                                collection_name=collection_id
                            )

            return jsonify({
                'status': 'success',
                'message': f'File {original_filename} processed successfully'
            })

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                print(f"Cleaned up temporary file: {temp_file_path}")

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/chat-message', methods=['POST'])
@swag_from({
    "parameters": [
        {
            "name": "body",
            "in": "body",
            "required": True,
            "schema": {
                "type": "object",
                "properties": {
                    "collection_id": {"type": "string", "description": "Collection ID to search in"},
                    "user_input": {"type": "string", "description": "User's question"}
                },
                "required": ["collection_id", "user_input"]
            }
        }
    ],
    "responses": {
        200: {
            "description": "Successfully processed chat message",
            "schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "faq_match": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "answer": {"type": "string"}
                        }
                    },
                    "detailed_answer": {"type": "string"},
                    "token_usage": {
                        "type": "object",
                        "properties": {
                            "input_tokens": {"type": "integer"},
                            "output_tokens": {"type": "integer"},
                            "total_tokens": {"type": "integer"}
                        }
                    }
                }
            }
        }
    }
})
def chat_message():
    try:
        data = request.get_json()
        if not data or 'collection_id' not in data or 'user_input' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields: collection_id or user_input'
            }), 400

        collection_id = data['collection_id']
        user_input = data['user_input']
        responses = None
        
        with Session(engine) as session:
            bot_settings = session.query(BotSetting).filter(BotSetting.collection_id == collection_id).first()
            if not bot_settings:
                bot_settings = BotSetting(collection_id=collection_id,
                                        threshold=0.2,
                                        k=5,
                                        temperature=0.5,
                                        model_name='gpt-4o')
                session.add(bot_settings)
                session.commit()
                
            if not bot_settings.use_dify:
                responses = local_response(user_input, bot_settings)
            else:
                logger.info('Using dify')
                responses = dify_response(user_input, bot_settings)
                
            # Save chat to database
            chat = Chat(
                user_input=user_input,
                system_answer=responses['detailed_answer'],
                collection_id=collection_id,
                llm_input_token=responses['token_usage']['input_tokens'],
                llm_output_token=responses['token_usage']['output_tokens'],
                llm_model=bot_settings.model_name,
                extra_data={
                    'collection_id': collection_id,
                    'faq_match': {
                        'question': responses['faq_match']['question'],
                        'answer': responses['faq_match']['answer']
                    },
                    'score': responses['faq_match']['score']
                }
            )
            session.add(chat)
            session.commit()

        return jsonify(responses)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)