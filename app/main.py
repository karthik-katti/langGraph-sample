import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from agent import agent_executor

# Initialize FastAPI app
app = FastAPI(title="LangGraph Agent API", version="1.0")


# Request body model
class MessageRequest(BaseModel):
    thread_id: str
    message: str


# In-memory storage for conversational memory
memory: Dict[str, Any] = {}


def get_thread_state(thread_id: str):
    """Retrieves the state for a given thread_id."""
    return memory.get(thread_id, {"messages": []})


@app.post("/chat")
async def chat_with_agent(request: MessageRequest):
    try:
        # Prepare the new input message
        new_input = {"messages": [HumanMessage(content=request.message)]}

        # Invoke the agent executor asynchronously
        output = await agent_executor.ainvoke(new_input, config={"configurable": {"thread_id": request.thread_id}})

        # Retrieve the final AI message from the output
        final_message = output["messages"][-1]

        # Return the content of the AI's response
        return {"response": final_message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "LangGraph Agent API is running. Go to /docs for more info."}


# Run the app with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)