import streamlit as st
from agno.agent import Agent, RunEvent
from agno.embedder.google import GeminiEmbedder
from agno.knowledge.url import UrlKnowledge
from agno.knowledge.website import WebsiteKnowledgeBase
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

groq_key = os.getenv("GROQ_API_KEY")
gemini_key = os.getenv("GOOGLE_API_KEY")

# Check if API keys are provided
if groq_key and gemini_key:

    # Initialize knowledge base (cached to avoid reloading)
    @st.cache_resource(show_spinner="📚 Loading knowledge base...")
    def load_knowledge() -> WebsiteKnowledgeBase:
        """Load and initialize the knowledge base with vector database"""
        kb = WebsiteKnowledgeBase(
            urls=["https://www.bis.gov.in/standards/technical-department/national-building-code/", "https://www.trade.gov/country-commercial-guides/india-environmental-technology"],  # Default URL
            vector_db=LanceDb(
                uri="tmp/lancedb",
                table_name="agno_docs",
                search_type=SearchType.vector,  # Use vector search
                embedder=GeminiEmbedder(
                    api_key=gemini_key
                ),
            ),
        )
        kb.load(recreate=True)  # Load documents into vector DB
        return kb
    
    # Initialize agent (cached to avoid reloading)
    @st.cache_resource(show_spinner="🤖 Loading agent...")
    def load_agent(_kb: WebsiteKnowledgeBase) -> Agent:
        """Create an agent with reasoning capabilities"""
        return Agent(
            model=Groq(
                id="deepseek-r1-distill-llama-70b", 
                api_key=groq_key
            ),
            knowledge=_kb,
            search_knowledge=True,  # Enable knowledge search
            tools=[ReasoningTools(add_instructions=True)],  # Add reasoning tools
            instructions=[
                "Include sources in your response.",
                "Always search your knowledge before answering the question.",
            ],
            markdown=True,  # Enable markdown formatting
        )

    # Load knowledge and agent
    knowledge = load_knowledge()
    agent = load_agent(knowledge)

    # Sidebar for knowledge management
    with st.sidebar:
        st.header("📚 Knowledge Sources")
        st.markdown("Add URLs to expand the knowledge base:")
        
        # Show current URLs
        st.write("**Current sources:**")
        for i, url in enumerate(knowledge.urls):
            st.text(f"{i+1}. {url}")
        
        # Add new URL
        st.divider()
        new_url = st.text_input(
            "Add new URL", 
            placeholder="https://example.com/docs",
            help="Enter a URL to add to the knowledge base"
        )
        
        if st.button("➕ Add URL", type="primary"):
            if new_url:
                with st.spinner("📥 Loading new documents..."):
                    knowledge.urls.append(new_url)
                    knowledge.load(
                        recreate=False,  # Don't recreate DB
                        upsert=True,     # Update existing docs
                        skip_existing=True  # Skip already loaded docs
                    )
                st.success(f"✅ Added: {new_url}")
                st.rerun()  # Refresh to show new URL
            else:
                st.error("Please enter a URL")

    # Main query section
    st.divider()
    st.subheader("🤔 Ask a Question")
    
    # Query input
    query = st.text_area(
        "Your question:",
        value="What are Agents?",
        height=100,
        help="Ask anything about the loaded knowledge sources"
    )
    
    # Run button
    if st.button("🚀 Get Answer with Reasoning", type="primary"):
        if query:
            # Create containers for streaming updates
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 🧠 Reasoning Process")
                reasoning_container = st.container()
                reasoning_placeholder = reasoning_container.empty()
            
            with col2:
                st.markdown("### 💡 Answer")
                answer_container = st.container()
                answer_placeholder = answer_container.empty()
            
            # Variables to accumulate content
            citations = []
            answer_text = ""
            reasoning_text = ""
            
            # Stream the agent's response
            with st.spinner("🔍 Searching and reasoning..."):
                for chunk in agent.run(
                    query,
                    stream=True,  # Enable streaming
                    show_full_reasoning=True,  # Show reasoning steps
                    stream_intermediate_steps=True,  # Stream intermediate updates
                ):
                    # Update reasoning display
                    if chunk.reasoning_content:
                        reasoning_text = chunk.reasoning_content
                        reasoning_placeholder.markdown(
                            reasoning_text, 
                            unsafe_allow_html=True
                        )
                    
                    # Update answer display
                    if chunk.content and chunk.event in {RunEvent.run_response, RunEvent.run_completed}:
                        if isinstance(chunk.content, str):
                            answer_text += chunk.content
                            answer_placeholder.markdown(
                                answer_text, 
                                unsafe_allow_html=True
                            )
                    
                    # Collect citations
                    if chunk.citations and chunk.citations.urls:
                        citations = chunk.citations.urls
            
            # Show citations if available
            if citations:
                st.divider()
                st.subheader("📚 Sources")
                for cite in citations:
                    title = cite.title or cite.url
                    st.markdown(f"- [{title}]({cite.url})")
        else:
            st.error("Please enter a question")

else:
    # Show instructions if API keys are missing
    st.info("""
    👋 **Welcome! To use this app, you need:**
    
    1. **Groq API Key** - For Claude AI model
    
    2. **Gemini** - For embeddings
    
    Once you have both keys, enter them above to start!
    """)

# Footer with explanation
st.divider()
with st.expander("📖 How This Works"):
    st.markdown("""
    **This app uses the Agno framework to create an intelligent Q&A system:**
    
    1. **Knowledge Loading**: URLs are processed and stored in a vector database (LanceDB)
    2. **Vector Search**: Uses Gemini's embeddings for semantic search to find relevant information
    3. **Reasoning Tools**: The agent uses special tools to think through problems step-by-step
    4. **Groq DeepSeek**: DeepSeek processes the information and generates answers
    
    **Key Components:**
    - `WebKnowledgeBase`: Manages data scraping from websites
    - `LanceDb`: Vector database for efficient similarity search
    - `GeminiIEmbedder`: Converts text to embeddings using Gemin embedding model
    - `ReasoningTools`: Enables step-by-step reasoning
    - `Agent`: Orchestrates everything to answer questions
    """)