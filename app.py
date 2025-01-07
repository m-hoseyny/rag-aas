from flask import Flask, jsonify, request
from db import BotSetting, engine, Chat
from sqlalchemy.orm import Session
import requests
import pandas as pd
import tempfile
import os
from rag import feed_data, get_faq, answer_me, local_response
from flasgger import Swagger, swag_from

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
                    "latest_chats": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "user_input": {"type": "string"},
                                "system_answer": {"type": "string"},
                                "created_at": {"type": "string"},
                                "llm_model": {"type": "string"}
                            }
                        }
                    }
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