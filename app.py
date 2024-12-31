from flask import Flask, jsonify, request
from db import engine, Chat
from sqlalchemy.orm import Session
import requests
import pandas as pd
import tempfile
import os
from rag import feed_data, get_faq, answer_me

app = Flask(__name__)

@app.route('/')
def index():
    with Session(engine) as session:
        # Get the latest 5 chats
        latest_chats = session.query(Chat).order_by(Chat.created_at.desc()).limit(5).all()
        chats = [{
            'id': str(chat.id),
            'user_input': chat.user_input,
            'system_answer': chat.system_answer,
            'created_at': chat.created_at.isoformat(),
            'llm_model': chat.llm_model
        } for chat in latest_chats]
        
        return jsonify({
            'status': 'success',
            'message': 'Welcome to RAG as a Service API',
            'latest_chats': chats
        })

@app.route('/upload-file', methods=['POST'])
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

        # Download the file
        response = requests.get(file_url)
        if response.status_code != 200:
            return jsonify({
                'status': 'error',
                'message': f'Failed to download file from URL: {file_url}'
            }), 400

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        file_extension = file_url.split('.')[-1].lower()
        
        if file_extension not in ['txt', 'csv']:
            os.unlink(temp_file_path)
            return jsonify({
                'status': 'error',
                'message': 'Only txt and csv files are supported'
            }), 400

        # Process the file based on its type
        if file_extension == 'csv':
            df = pd.read_csv(temp_file_path)
            if 'question' not in df.columns or 'answer' not in df.columns:
                os.unlink(temp_file_path)
                return jsonify({
                    'status': 'error',
                    'message': 'CSV must contain "question" and "answer" columns'
                }), 400
            
            for _, row in df.iterrows():
                feed_data(
                    page_content=row['question'],
                    metadata={'answer': row['answer'], 
                              'question': row['question'], 
                              'file_url': file_url},
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
                            metadata={'file_url': file_url},
                            collection_name=collection_id
                        )

        # Clean up the temporary file
        os.unlink(temp_file_path)

        return jsonify({
            'status': 'success',
            'message': 'File processed successfully'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/chat-message', methods=['POST'])
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

        # Get FAQ match
        faq_match = get_faq(collection_id, user_input)
        
        # Get detailed answer
        detailed_answer, callback = answer_me(
            collection_name=collection_id,
            user_input=user_input,
            temperature=0.4
        )

        response = {
            'status': 'success',
            'faq_match': {
                'question': faq_match.metadata.get('question', ''),
                'answer': faq_match.metadata.get('answer', '')
            },
            'detailed_answer': detailed_answer['answer'],
            'token_usage': {
                'input_tokens': callback.total_tokens,
                'output_tokens': callback.completion_tokens,
                'total_tokens': callback.total_tokens
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)