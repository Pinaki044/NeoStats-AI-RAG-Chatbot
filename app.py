from utils.rag import retrieve_context
from utils.web_search import search_web
import streamlit as st
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.llm import get_chatgroq_model


def get_chat_response(chat_model, messages, system_prompt):
    """Get response from the chat model with RAG"""
    try:
        # Get latest user question
        user_query = messages[-1]["content"]

        # Retrieve context from PDF
        context, sources = retrieve_context(user_query)

        # If no useful context found, search the web
        if not context or "Retrieval error" in context:
            web_context = search_web(user_query)
            context = f"Web Search Results:\n{web_context}"
            sources = ["Web Search (DuckDuckGo)"]

        # Create system prompt with context
        system_prompt = f"""
        {system_prompt}

        Use the following context to answer the question.

        Context:
        {context}

        If the answer is not in the context, answer based on your knowledge.
        """

        # Prepare messages
        formatted_messages = [SystemMessage(content=system_prompt)]

        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))

        # Get response
        response = chat_model.invoke(formatted_messages)

        source_text = "\n\nSources:\n" + "\n".join(sources)

        return response.content + source_text

    except Exception as e:
        return f"Error getting response: {str(e)}"

def instructions_page():
    """Instructions and setup page"""
    st.title("The Chatbot Blueprint")
    st.markdown("Welcome! Follow these instructions to set up and use the chatbot.")
    
    st.markdown("""
    ## 🔧 Installation
                
    
    First, install the required dependencies: (Add Additional Libraries base don your needs)
    
    ```bash
    pip install -r requirements.txt
    ```
    
    ## API Key Setup
    
    You'll need API keys from your chosen provider. Get them from:
    
    ### OpenAI
    - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
    - Create a new API key
    - Set the variables in config
    
    ### Groq
    - Visit [Groq Console](https://console.groq.com/keys)
    - Create a new API key
    - Set the variables in config
    
    ### Google Gemini
    - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
    - Create a new API key
    - Set the variables in config
    
    ## 📝 Available Models
    
    ### OpenAI Models
    Check [OpenAI Models Documentation](https://platform.openai.com/docs/models) for the latest available models.
    Popular models include:
    - `gpt-4o` - Latest GPT-4 Omni model
    - `gpt-4o-mini` - Faster, cost-effective version
    - `gpt-3.5-turbo` - Fast and affordable
    
    ### Groq Models
    Check [Groq Models Documentation](https://console.groq.com/docs/models) for available models.
    Popular models include:
    - `llama-3.1-70b-versatile` - Large, powerful model
    - `llama-3.1-8b-instant` - Fast, smaller model
    - `mixtral-8x7b-32768` - Good balance of speed and capability
    
    ### Google Gemini Models
    Check [Gemini Models Documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for available models.
    Popular models include:
    - `gemini-1.5-pro` - Most capable model
    - `gemini-1.5-flash` - Fast and efficient
    - `gemini-pro` - Standard model
    
    ## How to Use
    
    1. **Go to the Chat page** (use the navigation in the sidebar)
    2. **Start chatting** once everything is configured!
    
    ## Tips
    
    - **System Prompts**: Customize the AI's personality and behavior
    - **Model Selection**: Different models have different capabilities and costs
    - **API Keys**: Can be entered in the app or set as environment variables
    - **Chat History**: Persists during your session but resets when you refresh
    
    ## Troubleshooting
    
    - **API Key Issues**: Make sure your API key is valid and has sufficient credits
    - **Model Not Found**: Check the provider's documentation for correct model names
    - **Connection Errors**: Verify your internet connection and API service status
    
    ---
    
    Ready to start chatting? Navigate to the **Chat** page using the sidebar! 
    """)

def chat_page():
    """Main chat interface page"""
    st.title("🤖 AI ChatBot")

    # Response mode selector
    response_mode = st.radio(
    "Select Response Mode",
    ["Concise", "Detailed"]
    )

    # Default system prompt based on mode
    if response_mode == "Concise":
       system_prompt = "You are a helpful AI assistant. Give short and concise answers."
    else:
       system_prompt = "You are a helpful AI assistant. Provide detailed and thorough explanations."
    
    
    # Determine which provider to use based on available API keys
    chat_model = get_chatgroq_model()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    # if chat_model:
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display bot response
        with st.chat_message("assistant"):
            with st.spinner("Getting response..."):
                response = get_chat_response(chat_model, st.session_state.messages, system_prompt)
                st.markdown(response)
        
        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("🔧 No API keys found in environment variables. Please check the Instructions page to set up your API keys.")

def main():
    st.set_page_config(
        page_title="LangChain Multi-Provider ChatBot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Go to:",
            ["Chat", "Instructions"],
            index=0
        )
        
        # Add clear chat button in sidebar for chat page
        if page == "Chat":
            st.divider()
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    
    # Route to appropriate page
    if page == "Instructions":
        instructions_page()
    if page == "Chat":
        chat_page()

if __name__ == "__main__":
    main()