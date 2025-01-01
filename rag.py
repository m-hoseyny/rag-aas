
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

THRESHOLD = 0.7

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
    choosen_qrel = None
    for qrel in qrels:
        if 1 - qrel[1] > THRESHOLD:
            choosen_qrel = qrel[0]
            return choosen_qrel
    return choosen_qrel


def answer_me(collection_name, user_input, system_prompt=None, k=5, temperature=0.5, history=None, model='gpt-4o'):
    
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
            "If the context is not provided or not related to the question, do not answer and tell در باره این موضوع اطلاعی ندارم."
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
                                         search_kwargs={"score_threshold": THRESHOLD})
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