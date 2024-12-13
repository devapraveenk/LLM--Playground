import nest_asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()


class ResponseModel(BaseModel):
    """Structured response with metadata."""

    author_name: str
    genre: str
    summary: str = Field(description="Give the summary of the book")
    rating: str


biblio_agent = Agent(
    model=OpenAIModel("gpt-4o-mini"),
    result_type=ResponseModel,
    system_prompt=(
        "You are the Book Analysis Agent."
        "Analyze the book and retrieve detailed information about it from the internet."
    ),
)

response = biblio_agent.run_sync(user_prompt="Verity")  # book name

print(response.data.model_dump_json(indent=2))
# {
#   "author_name": "Colleen Hoover",
#   "genre": "Psychological Thriller, Romance",
#   "summary": "Verity is a psychological thriller that follows Lowen Ashleigh,....
#   "rating": "4.5/5"
# }
