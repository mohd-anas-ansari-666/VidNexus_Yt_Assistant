import streamlit as st
import yt_info_extractor as yex
import textwrap
import os

st.title("YouTube Assistant")

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.sidebar.text_area(
            label="What is the YouTube video URL?",
            max_chars=50
        )
        query = st.sidebar.text_area(
            label="Ask me about the video?",
            max_chars=50,
            key="query"
        )
        gemini_api_key = st.sidebar.text_input(
            label="Gemini API Key",
            key="langchain_search_api_key_gemini",
            max_chars=50,
            type="password"
        )
        submit_button = st.form_submit_button(label='Submit')

if query and youtube_url:
    if not gemini_api_key:
        st.info("Please add your Gemini API key to continue.")
        st.stop()
    else:
        try:
            # Set the API key in environment for the backend functions
            os.environ["GEMINI_API_KEY"] = gemini_api_key
            
            with st.spinner("Processing video and generating response..."):
                db = yex.create_db_from_youtube_video_url(youtube_url)
                response, docs = yex.get_response_from_query(db, query)
            
            st.subheader("Answer:")
            st.write(response)  # Use st.write instead of textwrap and st.text
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")