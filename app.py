# COLAB SETUP SCRIPT - Run this cell first in Google Colab

# Install all required packages
!pip install streamlit>=1.28.0
!pip install langchain>=0.1.0
!pip install langchain-groq>=0.1.0
!pip install langchain-core>=0.1.0

# Download the app file (if you have it hosted somewhere)
# OR create the app.py file directly in Colab

# Create the app.py file directly in Colab
app_code = '''
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# Configure Streamlit page
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)

# App title and description
st.title("🤖 Agentic AI Research Assistant")
st.markdown("Enter a topic and get a structured research summary with key subtopics!")

# Sidebar for API key input
with st.sidebar:
    st.header("🔑 Configuration")
    groq_api_key = st.text_input(
        "Enter your Groq API Key:",
        type="password",
        help="Get your free API key from https://console.groq.com/"
    )
    
    # Model selection
    model_choice = st.selectbox(
        "Choose Model:",
        ["llama3-8b-8192", "mixtral-8x7b-32768"],
        help="LLaMA3 is faster, Mixtral is more capable"
    )

def initialize_agent(api_key, model_name):
    """Initialize the Groq LLM agent"""
    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=0.3,
            max_tokens=1024
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return None

def create_research_prompt():
    """Create the research prompt template"""
    template = """
You are an AI research assistant. Your task is to analyze a given topic and break it down into subtopics with summaries.

TOPIC: {topic}

INSTRUCTIONS:
1. Break the topic into exactly 3 relevant subtopics
2. For each subtopic, provide 3-5 bullet points summary
3. Keep summaries concise and informative
4. Focus on the most important and current aspects

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

## Subtopic 1: [Subtopic Name]
• [Bullet point 1]
• [Bullet point 2]
• [Bullet point 3]
• [Bullet point 4]
• [Bullet point 5]

## Subtopic 2: [Subtopic Name]
• [Bullet point 1]
• [Bullet point 2]
• [Bullet point 3]
• [Bullet point 4]

## Subtopic 3: [Subtopic Name]
• [Bullet point 1]
• [Bullet point 2]
• [Bullet point 3]
• [Bullet point 4]
• [Bullet point 5]

Topic to analyze: {topic}
"""
    return PromptTemplate(template=template, input_variables=["topic"])

def process_research_query(agent, topic):
    """Process the research query using the agent"""
    try:
        # Create prompt
        prompt_template = create_research_prompt()
        formatted_prompt = prompt_template.format(topic=topic)
        
        # Get response from agent
        with st.spinner("🔍 Researching and analyzing..."):
            response = agent.invoke([HumanMessage(content=formatted_prompt)])
            
        return response.content
    
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None

def display_results(results):
    """Display the research results in a formatted way"""
    if results:
        st.markdown("## 📊 Research Summary")
        st.markdown(results)
        
        # Add download option
        st.download_button(
            label="📥 Download Summary",
            data=results,
            file_name="research_summary.md",
            mime="text/markdown"
        )

def main():
    # Check if API key is provided
    if not groq_api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar to get started.")
        st.markdown("""
        ### How to get your Groq API key:
        1. Visit [Groq Console](https://console.groq.com/)
        2. Sign up for a free account
        3. Navigate to API Keys section
        4. Create a new API key
        5. Copy and paste it in the sidebar
        """)
        return
    
    # Initialize the agent
    agent = initialize_agent(groq_api_key, model_choice)
    if not agent:
        return
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Topic input
        topic = st.text_input(
            "🎯 Enter your research topic:",
            placeholder="e.g., Latest AI tools for teachers",
            help="Be specific for better results"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add space
        research_button = st.button("🚀 Start Research", type="primary")
    
    # Process query when button is clicked
    if research_button and topic:
        if len(topic.strip()) < 3:
            st.error("Please enter a more specific topic (at least 3 characters)")
            return
            
        # Process the research query
        results = process_research_query(agent, topic.strip())
        
        if results:
            display_results(results)
    
    elif research_button and not topic:
        st.error("Please enter a research topic first!")
    
    # Example topics
    st.markdown("---")
    st.markdown("### 💡 Example Topics:")
    example_topics = [
        "Latest AI tools for teachers",
        "Sustainable energy solutions 2024",
        "Remote work productivity strategies",
        "Cybersecurity trends for small businesses",
        "Digital marketing for startups"
    ]
    
    cols = st.columns(len(example_topics))
    for i, example in enumerate(example_topics):
        with cols[i]:
            if st.button(f"📝 {example}", key=f"example_{i}"):
                st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using [Streamlit](https://streamlit.io) and [LangChain](https://langchain.com) | "
    "Powered by [Groq](https://groq.com)"
)

if __name__ == "__main__":
    main()
'''

# Write the app code to a file
with open('app.py', 'w') as f:
    f.write(app_code)

print("✅ Setup complete! Now run the next cell to start the app.")
print("📁 Created: app.py")
print("🚀 Next step: Run '!streamlit run app.py --server.port 8501' in a new cell")
