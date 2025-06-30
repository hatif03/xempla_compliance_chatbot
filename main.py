import streamlit as st
from agno.agent import Agent, RunEvent
from agno.embedder.google import GeminiEmbedder
from agno.knowledge.url import UrlKnowledge
from agno.models.groq import Groq
from agno.tools.reasoning import ReasoningTools
from agno.vectordb.lancedb import LanceDb, SearchType
from dotenv import load_dotenv
import os

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Agentic RAG with Reasoning", 
    page_icon="🧐", 
    layout="wide"
)

st.title("🧐 Agentic RAG with Reasoning")
st.markdown("""
This app demonstrates an AI agent that:
1. **Retrieves** relevant information from knowledge sources
2. **Reasons** through the information step-by-step
3. **Answers** your questions with citations
""")

