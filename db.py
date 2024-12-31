
from sqlalchemy import Column, UUID, ForeignKey, Text, BigInteger, Integer, String, DateTime, JSON, create_engine
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import os, uuid
from sqlalchemy.orm import DeclarativeBase
from dotenv import load_dotenv

load_dotenv()

MAIN_DB_USERNAME=os.environ.get('MAIN_DB_USERNAME', 'postgres')
MAIN_DB_PASSWORD=os.environ.get('MAIN_DB_PASSWORD', '1234')
MAIN_DB_DATABASE=os.environ.get('MAIN_DB_DATABASE', 'rag-ass-chats')
MAIN_DB_HOSTNAME=os.environ.get('MAIN_DB_HOSTNAME', 'localhost')
MAIN_DB_PORT=os.environ.get('MAIN_DB_PORT', '5432')


SQLALCHEMY_DATABASE_URI = f"postgresql+psycopg://{MAIN_DB_USERNAME}:{MAIN_DB_PASSWORD}@{MAIN_DB_HOSTNAME}:{MAIN_DB_PORT}/{MAIN_DB_DATABASE}"
engine = create_engine(SQLALCHEMY_DATABASE_URI)


class Base(DeclarativeBase):
    pass

class Chat(Base):
    __tablename__ = 'chats'
    id = Column(UUID(as_uuid=True), nullable=False, 
                default=uuid.uuid4,
                primary_key=True)
    user_input = Column(Text, nullable=False)
    system_answer = Column(Text, nullable=False)
    collection_id = Column(String(200), nullable=False)
    
    llm_input_token = Column(Integer, default=0)
    llm_output_token = Column(Integer, default=0)
    llm_model = Column(String(200))
    
    created_at = Column(DateTime, server_default=func.now())
    extra_data = Column(JSON, default={})  # JSON field to store additional data
    
    def __str__(self) -> str:
        return f'<Chat {self.id}>'
    
    
Base.metadata.create_all(engine)


def add_new_chat(user_input, response, openai_callback, model):

    llm_input_token = openai_callback.prompt_tokens
    llm_output_token = openai_callback.completion_tokens
    chat = Chat(
        user_input=user_input,
        system_answer=response['answer'],
        llm_input_token=llm_input_token,
        llm_output_token=llm_output_token,
        llm_model=model,
    )
    session.add(chat)
    session.commit()
    return chat