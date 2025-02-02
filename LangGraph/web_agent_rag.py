import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from typing import List, TypedDict

# Load environment variables
load_dotenv()


# Define State for Query Processing
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Initialize OpenAI LLM and Embeddings
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Setup Qdrant In-Memory Vector Storage
client = QdrantClient(":memory:")
client.create_collection(
    collection_name="web_collection",
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)
vector_store = QdrantVectorStore(
    client=client,
    collection_name="web_collection",
    embedding=embeddings,
)

# Ask for a URL from the user
URL = input("Enter the URL to scrape: ").strip()

# Initialize Playwright Web Scraper
loader = PlaywrightURLLoader(urls=[URL], remove_selectors=["header", "footer"])


# Load and Split Web Content
async def load_documents():
    docs = await loader.aload()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)


all_splits = asyncio.run(load_documents())

# Index Extracted Data in Qdrant
vector_store.add_documents(documents=all_splits)


# Retrieve Relevant Data from Qdrant
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


# Generate AI Response using GPT-4o-mini
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    system_msg = """You are a customer service chatbot named 'Seraphic'.
    Provide clear, concise, and polite responses. If the answer is not found, inform the user that the information is unavailable."""

    template = f"Use the following context to answer the question at the end.\n\nContext: {docs_content}\nQuestion: {state['question']}"

    query_generate = ChatPromptTemplate.from_messages(
        [("system", system_msg), ("placeholder", "{template}")]
    )
    output = query_generate | llm

    response = output.invoke({"template": [HumanMessage(content=template)]})
    return {"answer": response.content}


# Initialize Memory for LangGraph
memory = MemorySaver()

# Build Chatbot Graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile(checkpointer=memory)

# Start Chatbot Interaction
while True:
    question = input("\nYou: ").strip()

    if question.lower() in ["exit", "quit"]:
        print("Exiting chatbot... Have a great day! 😊")
        break

    result = graph.invoke(
        {"question": question}, config={"configurable": {"thread_id": "abc123"}}
    )
    print(f"\nSeraphic: {result['answer']}")
