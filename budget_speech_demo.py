# Import necessary libraries
# import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import uuid

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.title("Provide your OpenAI API Key")
    OPENAI_API_KEY = st.text_input(label="Please enter your OpenAI API key", type="password")
if not OPENAI_API_KEY:
    st.info("Please enter your OpenAI API Key")
    st.stop()



# --- 1. Configuration and Setup ---
# Load the OpenAI API key from environment variables
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI embeddings model
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)


# Initialize the OpenAI Chat model (LLM)
@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

llm = get_llm()


# --- 2. Data Loading and Indexing ---
# Load the source document

# Create a retriever from the vector store to fetch relevant documents
@st.cache_resource
def load_vector_store():
    document = PyPDFLoader("budget_speech.pdf").load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(document)
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store.as_retriever()


retriever = load_vector_store()

# --- 3. Prompt Engineering for Conversational Retrieval ---
# Define the prompt template for the main question-answering part of the chain.
# This prompt instructs the LLM on how to use the retrieved context and includes
# a placeholder for the chat history, making it conversation-aware.
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an assistant for answering questions.
        Use the provided context to respond. If the answer 
        isn't clear, acknowledge that you don't know.
        Do not answer any other question apart from the given documentation.
        Do not mention that you are answering from the given documentation.
        Avoid disclosing any information source or methology used.
        Don't say 'The document does not mention' or 'it is not mentioned in doc'.
        avoid using mention of document world at all.
        {context}
        """),
        # MessagesPlaceholder is used to inject the chat history
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

# --- 4. Building the RAG Chain with History Awareness ---
# Create a history-aware retriever. This chain takes the user's input and chat history,
# rephrases the input to be a standalone question, and then uses the retriever
# to fetch relevant documents. This is crucial for follow-up questions.
history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt_template)

# Create a chain that combines (stuffs) the retrieved documents into the prompt
# and sends it to the LLM for a final answer.
qa_chain = create_stuff_documents_chain(llm, prompt_template)

# Create the final retrieval chain by combining the history-aware retriever and the QA chain.
# This is the core of our RAG application.
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# --- 5. Managing Chat History ---
# Initialize a message history object that stores messages in Streamlit's session state.
# This ensures that the history persists across user interactions in the same session.
streamlit_history = StreamlitChatMessageHistory()

# Wrap the RAG chain with a message history manager.
# This `RunnableWithMessageHistory` automatically handles saving user inputs and AI
# responses to the specified history object.
history_chain = RunnableWithMessageHistory(
    rag_chain,
    # A lambda function to retrieve the history object for a given session ID
    lambda session_id: streamlit_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# --- 6. Streamlit Chat Interface with History ---
st.title("💬 Budget Speech Chatbot - 2025")

# Display past messages from chat history
if streamlit_history.messages:
    for message in streamlit_history.messages:
        with st.chat_message(message.type):
            st.markdown(message.content)

# User input box at the bottom of the screen
if prompt := st.chat_input("Ask a question about the Budget Speech..."):
    # Show user's message in chat
    with st.chat_message("human"):
        st.markdown(prompt)

    # Call the RAG chain to get a response
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            response = history_chain.invoke(
                {"input": prompt},
                {"configurable": {"session_id": st.session_state.session_id}}
            )
            raw_answer = response.get("answer", "No answer found.")

            # Split and show response in paragraphs for readability
            paragraphs = raw_answer.strip().split('\n\n')
            for paragraph in paragraphs:
                st.markdown(paragraph.strip())

            # Append user message to history
            st.session_state.chat_history.append({"role": "human", "content": prompt})
            # Append assistant message to history
            st.session_state.chat_history.append({"role": "assistant", "content": raw_answer})

        # Show chat history
st.divider()
# st.text("💬 Chat History")

# Show chat history in a collapsible section (collapsed by default)
with st.expander("💬 Show Chat History"):
    for msg in st.session_state.chat_history:
        if msg["role"] == "human":
            st.markdown(f"**🧑 You:** {msg['content']}")
        else:
            st.markdown(f"**🤖 Bot:** {msg['content']}")
