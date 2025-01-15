import pandas as pd
import nest_asyncio
from langchain import hub
from langchain_core.messages import HumanMessage
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()

df = pd.read_csv("../products.csv")  # Load your dataset here

engine = create_engine("sqlite:///products.db")  # Create a database
df.to_sql("products", engine, index=False)

db = SQLDatabase(engine=engine)
llm = ChatOpenAI(model="gpt-4o-mini")  # Load your model here

toolkit = SQLDatabaseToolkit(db=db, llm=llm)  # Langchain SQL Toolkit
tools = toolkit.get_tools()

prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
system_message = prompt_template.format(dialect="SQLite", top_k=5)

memory = MemorySaver()

agent_executor = create_react_agent(
    llm, tools, state_modifier=system_message, checkpointer=memory
)  # Creating ReAct Agent
config = {"configurable": {"thread_id": "abc123"}}

question = "ENTER YOUR QUESTION HERE"

for step in agent_executor.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
    config=config,
):
    step["messages"][-1].pretty_print()
