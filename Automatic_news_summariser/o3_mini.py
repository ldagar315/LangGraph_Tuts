import os
from langgraph.graph import Graph
from langchain_groq import ChatGroq
from langchain.tools import TavilySearchResults
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from tavily import TavilyClient

os.environ["GROQ_API_KEY"] = 'GROQ_API_KEY'
os.environ["TAVILY_API_KEY"]= "TAVILY_API_KEY"
# Set up the LLM model
llm = ChatGroq(model="llama-3.3-70b-versatile")  # Add API Key

# Set up the Tavily Search API
search_tool = TavilySearchResults()  # Add API Key
client = TavilyClient()
# Define Pydantic model for quiz questions
class QuizQuestion(BaseModel):
    question: str = Field(description="The text of the question, focusing on core concepts.")
    options: List[str] = Field(description="A list of exactly 4 possible answer strings.")
    correct_option: str = Field(description="The string corresponding to the correct answer, which must be one of the provided options.")

# Define the agent state
class AgentState(TypedDict):
    input_topic: str
    input_num_days: int
    input_region: str
    initial_search_query: Optional[str]
    search_results: Optional[List[Dict[str, Any]]]
    processed_articles: Optional[List[Dict[str, str]]]
    needs_expansion: bool
    expansion_query: Optional[str]
    summary: Optional[str]
    quiz: Optional[List[QuizQuestion]]
    error: Optional[str]
    current_step: Optional[str]

# Define processing functions
#tool = TavilySearchResults(max_results=3)
#llm_with_tools = llm.bind_tools(tool)
def formulate_search_query(state: AgentState) -> AgentState:
    query = f"Find news about '{state['input_topic']}' in '{state['input_region']}' from the past {state['input_num_days']} days."
    state["initial_search_query"] = query
    state["current_step"] = "formulate_search_query"
    return state

def search_news(state: AgentState) -> AgentState:
    try:
        results = client.search(query = state["initial_search_query"])
        #results = llm_with_tools.invoke(state['initial_search_query'])
        #results = search_tool.run(state["initial_search_query"])
        state["search_results"] = results  # Limit to top 3
    except Exception as e:
        state["error"] = str(e)
    state["current_step"] = "search_news"
    return state

def process_articles(state: AgentState) -> AgentState:
    if not state["search_results"]:
        state["error"] = "No articles found."
        return state
    
    processed = []
    for result in state["search_results"]:
        processed.append({
            "url": result.get("url"),
            "snippet": result.get("snippet", "")
        })
    state["processed_articles"] = processed
    state["current_step"] = "process_articles"
    return state

def summarize_news(state: AgentState) -> AgentState:
    articles_text = "\n".join([article["snippet"] for article in state["processed_articles"]])
    prompt = f"Summarize the following news articles:\n{articles_text}"
    summary = llm.invoke(prompt)
    state["summary"] = summary
    state["current_step"] = "summarize_news"
    return state

def generate_quiz(state: AgentState) -> AgentState:
    prompt = f"Generate 3 multiple-choice questions based on:\n{state['summary']}"
    quiz = llm.invoke(prompt)
    state["quiz"] = quiz
    state["current_step"] = "generate_quiz"
    return state

# Define LangGraph workflow
graph = StateGraph(AgentState)
graph.add_node("formulate_search_query", formulate_search_query)
graph.add_node("search_news", search_news)
graph.add_node("process_articles", process_articles)
graph.add_node("summarize_news", summarize_news)
graph.add_node("generate_quiz", generate_quiz)
#tool_node = ToolNode(tools=tool)
#graph.add_node("tools", tool_node)


graph.add_edge(START, "formulate_search_query")
graph.add_edge("formulate_search_query","tool")
graph.add_edge("tool", "process_articles")
graph.add_edge("process_articles", "summarize_news")
graph.add_edge("summarize_news", "generate_quiz")
graph.add_edge("generate_quiz", END)
graph_final = graph.compile()

def run_agent(topic: str, num_days: int, region: str):
    #state = setup_initial_state(topic, num_days, region)
    state = formulate_search_query(state)
    state = search_news(state)
    state = process_articles(state)
    state = summarize_news(state)
    state = generate_quiz(state)
    return state

# Example Usage
if __name__ == "__main__":
    topic = "AI Regulation"
    num_days = 3
    region = "Global"
    result = graph_final.invoke({'input_topic': topic,'input_num_days': num_days,'input_region': region})
    print("Summary:", result["summary"])
    print("Quiz:", result["quiz"])

