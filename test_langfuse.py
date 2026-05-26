from dotenv import load_dotenv
load_dotenv()   # loads .env from same directory

from langfuse.decorators import observe
from langfuse import Langfuse

@observe()
def test_trace():
    return "hello from RAG app"

test_trace()
Langfuse().flush()
print("Done — check Langfuse dashboard now")
