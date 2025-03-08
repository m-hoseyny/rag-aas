from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from langchain_postgres.vectorstores import PGVector
import os

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import AIMessage, HumanMessage

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

OPENAI_BASE_URL='https://litellm.tosiehgar.ir'
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY', '')


VECTOR_DB_USERNAME=os.environ.get('VECTOR_DB_USERNAME', 'postgres')
VECTOR_DB_PASSWORD=os.environ.get('VECTOR_DB_PASSWORD', '1234')
VECTOR_DB_HOSTNAME=os.environ.get('VECTOR_DB_HOSTNAME', '1234')
VECTOR_DB_DATABASE=os.environ.get('VECTOR_DB_DATABASE', 'karbala')
VECTOR_DB_PORT=os.environ.get('VECTOR_DB_PORT', '6604')

THRESHOLD = 0.4

# See docker command above to launch a postgres instance with pgvector enabled.
connection = f"postgresql+psycopg://{VECTOR_DB_USERNAME}:{VECTOR_DB_PASSWORD}@{VECTOR_DB_HOSTNAME}:{VECTOR_DB_PORT}/{VECTOR_DB_DATABASE}"  # Uses psycopg3!

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)


def get_vector_db(collection_name):
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection
    )
    return vectorstore

def get_faq(collection_name, user_input):
    vectorstore = get_vector_db(collection_name)
    print('Checking FAQ')
    qrels = vectorstore.similarity_search_with_score(user_input)
    # print(qrels)
    choosen_qrel = [None, None]
    for qrel in qrels:
        if 1- qrel[1] > THRESHOLD:
            choosen_qrel = qrel[0]
            return qrel
    return choosen_qrel


def answer_me(collection_name, user_input, 
              system_prompt=None, k=5, 
              temperature=0.5, history=None, model='gpt-4o',
              threshold=THRESHOLD):
    
    vectorstore = get_vector_db(collection_name)

    if not system_prompt:
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer. "
            "Do not tell about yourself, your model. If the user ask about them, tell the user you cannot answer it. "
            "Just answer based on the user question. "
            "You should not answer anything that is not related to the user question. "
            "Always answer respectfully. If user is aggressive or insolence or bully, just answer it politely."
            "your main language is farsi/persian. Answer all questions and queries in persian. "
            # "If the context not related to the question, do not answer and tell در باره این موضوع اطلاعی ندارم."
        )
        
    system_prompt = system_prompt + '\n\n{context}'

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    llm = ChatOpenAI(model=model, max_tokens=300, base_url=OPENAI_BASE_URL, 
                     api_key=OPENAI_API_KEY,
                     temperature=temperature)

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    # retriever = vectorstore.as_retriever(search_kwargs={'k': k, 'fetch_k': k})
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", 
                                         search_kwargs={"score_threshold": threshold})
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    chat_history = []
    if history:
        for h in history:
            chat_history.extend(
                [
                    HumanMessage(content=h.user_input),
                    AIMessage(content=h.system_answer),
                ]
            )
    openai_callback = None
    # user_input = add_english_numbers_to_text(user_input)
    with get_openai_callback() as cb:
        response = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
        if not response['context']:
            response = {'input': user_input, 'chat_history': [], 'context': [], 'answer': 'درباره این موضوع اطلاعی ندارم.'}
        openai_callback = cb
        return response, openai_callback
    

def feed_data(page_content, metadata, collection_name):
    vectorstore = get_vector_db(collection_name)
    docs = []
    docs.append(Document(page_content=page_content, metadata=metadata))
    result = vectorstore.add_documents(docs)
    return result


def feed_data_batch(documents, collection_name):
    """
    Add multiple documents to the vector database in a single operation.
    
    Args:
        documents (list): List of dictionaries, each containing 'page_content' and 'metadata'
        collection_name (str): Name of the collection to add documents to
        
    Returns:
        list: List of document IDs added to the database
    """
    vectorstore = get_vector_db(collection_name)
    docs = []
    for doc in documents:
        docs.append(Document(page_content=doc['page_content'], metadata=doc['metadata']))
    result = vectorstore.add_documents(docs)
    return result


def retrieve(collection_name, query, top_k=5, score_threshold=0.5):
    """
    Retrieve documents from the vector database based on similarity to the query.
    
    Args:
        collection_name (str): Name of the collection to search in
        query (str): The query text to search for
        top_k (int): Maximum number of documents to return
        score_threshold (float): Minimum similarity score threshold
        
    Returns:
        list: List of dictionaries containing retrieved documents with metadata and scores
    """
    vectorstore = get_vector_db(collection_name)
    
    # Perform similarity search with scores
    results = vectorstore.similarity_search_with_score(
        query=query,
        k=top_k
    )
    
    # Format the results according to the specified response format
    records = []
    for doc, score in results:
        # Convert score from distance to similarity (1 - distance)
        similarity_score = 1 - score
        
        # Skip documents below the threshold
        if similarity_score < score_threshold:
            continue
        
        # Extract title from metadata or use empty string
        title = ""
        if hasattr(doc, 'id') and doc.id:
            title = str(doc.id)
        
        record = {
            "metadata": doc.metadata,
            "score": round(float(similarity_score), 2),
            "title": title,
            "content": doc.page_content
        }
        records.append(record)
    
    return records


def local_response(user_input, bot_settings):
    answer = ''
    question = ''
    collection_id = bot_settings.collection_id
    # Get FAQ match
    faq_match = get_faq(collection_id, user_input)
    output = ''
    if faq_match[0]:
        question = faq_match[0].metadata.get('question', '')
        answer = faq_match[0].metadata.get('answer', '')

    # Get detailed answer
    model = bot_settings.model_name 
    detailed_answer, callback = answer_me(
        collection_name=collection_id,
        user_input=user_input,
        temperature=bot_settings.temperature,
        model=model,
        k=bot_settings.k,
        system_prompt=bot_settings.prompt,
        threshold=bot_settings.threshold
    )
    if detailed_answer['answer']:
        # output = 'به صورت خلاصه\n{}'.format(detailed_answer['answer'])
        output = detailed_answer['answer']
    # Prepare response
    response = {
        'status': 'success',
        'faq_match': {
            'question': question,
            'answer': answer,
            'score': faq_match[1]
        },
        'detailed_answer': output,
        'token_usage': {
            'input_tokens': callback.total_tokens,
            'output_tokens': callback.completion_tokens,
            'total_tokens': callback.total_tokens
        },
        
    }
    
    return response


def dify_response(user_input, bot_settings):
    import requests
    
    headers = {
        'Authorization': f'Bearer {bot_settings.dify_api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'inputs': {},
        'query': user_input,
        'response_mode': 'blocking',  # Using blocking instead of streaming for simpler implementation
        'conversation_id': '',
        'user': 'default-user'
    }
    
    try:
        response = requests.post(
            f'{bot_settings.dify_url}/chat-messages',
            headers=headers,
            json=payload
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        
        data = response.json()
        
        # Format response to match local_response structure
        return {
            'status': 'success',
            'faq_match': {
                'question': '',
                'answer': '',
                'score': 0
            },
            'detailed_answer': data.get('answer', ''),
            'token_usage': {
                'input_tokens': data.get('tokens', {}).get('prompt_tokens', 0),
                'output_tokens': data.get('tokens', {}).get('completion_tokens', 0),
                'total_tokens': data.get('tokens', {}).get('total_tokens', 0)
            }
        }
        
    except requests.RequestException as e:
        return {
            'status': 'error',
            'faq_match': {
                'question': '',
                'answer': '',
                'score': 0
            },
            'detailed_answer': f'Error connecting to Morshed API: {str(e)}',
            'token_usage': {
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0
            }
        }