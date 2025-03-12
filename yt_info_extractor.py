from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate 
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import FAISS
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize embeddings with API key
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# def create_db_from_youtube_video_url(video_url: str) -> FAISS:
#     loader = YoutubeLoader.from_youtube_url(video_url)
#     transcript = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     docs = text_splitter.split_documents(transcript)

#     db = FAISS.from_documents(docs, embeddings)
#     return db

def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    """Extracts transcript from a YouTube video and stores it in a FAISS vector database."""
    
    # Extract video ID from URL
    yt = YouTube(video_url)
    video_id = yt.video_id
    
    # Get transcript
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_text = " ".join([item["text"] for item in transcript_list])
    
    # Create documents
    from langchain_core.documents import Document
    doc = Document(page_content=transcript_text, metadata={"source": video_url})
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents([doc])
    
    # Create vector database
    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, k=4):
    """Retrieves relevant transcript chunks and generates an AI response using Gemini."""
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    # Initialize Gemini LLM with API key
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key)

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that can answer questions about YouTube videos 
        based on the video's transcript.

        Answer the following question: {question}
        By searching the following video transcript: {docs}

        Only use the factual information from the transcript to answer the question.

        If you feel like you don't have enough information to answer the question, say "I don't know".

        Your answers should be verbose and detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", " ")
    return response, docs
