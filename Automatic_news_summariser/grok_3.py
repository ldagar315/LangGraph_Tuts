from langgraph.graph import StateGraph, END
from groq import Groq
from tavily import TavilyClient
from typing import Optional, List
import os

# API Key placeholders (replace with your actual keys)
GROQ_API_KEY = "your_groq_api_key_here"
TAVILY_API_KEY = "your_tavily_api_key_here"

# Initialize clients
groq_client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Data Structures
class NewsInput:
    """Class to hold user input for news summarization and quiz generation."""
    def __init__(self, topic: str, region: str, number_of_days: int, quiz: bool = False):
        self.topic = topic
        self.region = region
        self.number_of_days = number_of_days
        self.quiz = quiz

    def validate(self) -> bool:
        """Validate the input parameters."""
        return (self.topic.strip() != "" and 
                self.region.strip() != "" and 
                self.number_of_days > 0 and 
                isinstance(self.number_of_days, int))

class Question:
    """Class to represent a quiz question with options and correct answer."""
    def __init__(self, question: str, options: List[str], correct_option: str):
        self.question = question
        self.options = options
        self.correct_option = correct_option

    def is_valid(self) -> bool:
        """Check if the question is valid."""
        return (len(self.options) == 4 and 
                self.correct_option in self.options and 
                self.question.strip() != "")

class NewsOutput:
    """Class to hold the final output with summary and optional quiz."""
    def __init__(self, summary: str, quiz: List[Question] = None):
        self.summary = summary
        self.quiz = quiz if quiz else []

# State Definition
class State:
    """Class to track the agent's state throughout the workflow."""
    def __init__(self):
        self.input: Optional[NewsInput] = None
        self.query: Optional[str] = None
        self.articles: Optional[List[str]] = None
        self.summary: Optional[str] = None
        self.quiz: Optional[List[Question]] = None
        self.status: str = "initialized"

# Node Functions
def validate_input_node(state: State) -> State:
    """Validate the user input and update the state accordingly."""
    if not state.input.validate():
        state.status = "error"
        state.summary = "Invalid input provided."
    else:
        state.status = "formulating"
    return state

def formulate_query_node(state: State) -> State:
    """Formulate the search query based on user input."""
    if state.status == "formulating":
        state.query = f"{state.input.topic} news in {state.input.region} last {state.input.number_of_days} days"
        state.status = "fetching"
    return state

def call_search_tool_node(state: State) -> State:
    """Fetch news articles using the Tavily Search Tool with retry logic."""
    if state.status == "fetching":
        for attempt in range(3):
            try:
                results = tavily_client.search(state.query)
                # Assuming results is a list of dicts with 'text' key containing article content
                state.articles = [result['text'] for result in results]
                state.status = "summarizing"
                break
            except Exception as e:
                if attempt == 2:
                    state.status = "error"
                    state.summary = "Failed to fetch articles after 3 attempts."
    return state

def summarize_node(state: State) -> State:
    """Summarize the fetched articles using ChatGroq."""
    if state.status == "summarizing":
        prompt = (
            f"Summarize the following news articles about {state.input.topic} in {state.input.region} "
            f"from the last {state.input.number_of_days} days. Make the summary concise (maximum 3 sentences), "
            "simplify complex information, and provide background where necessary to ensure clarity.\n\n"
            "Articles:\n" + "\n".join(state.articles)
        )
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        state.summary = response.choices[0].message.content.strip()
        state.status = "quiz" if state.input.quiz else "complete"
    return state

def generate_quiz_node(state: State) -> State:
    """Generate a quiz based on the summary using ChatGroq."""
    if state.status == "quiz":
        prompt = (
            f"Based on this summary of news about {state.input.topic} in {state.input.region}, "
            "create 3 multiple-choice questions that test understanding of the core context, not just facts. "
            "Each question should have one correct answer and three plausible distractors. "
            "Format each question as: Q: [question] A) [option1] B) [option2] C) [option3] D) [option4] Correct: [A/B/C/D].\n\n"
            f"Summary:\n{state.summary}"
        )
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        quiz_text = response.choices[0].message.content.strip()
        state.quiz = parse_quiz(quiz_text)
        state.status = "complete"
    return state

def parse_quiz(quiz_text: str) -> List[Question]:
    """Parse the quiz text into a list of Question objects."""
    questions = []
    lines = quiz_text.split('\n')
    i = 0
    while i < len(lines):
        if lines[i].startswith('Q:'):
            question = lines[i][3:].strip()
            options = []
            for j in range(4):
                i += 1
                option = lines[i].split(') ', 1)[1].strip()
                options.append(option)
            i += 1
            correct_letter = lines[i].split(': ', 1)[1].strip()
            correct_index = ord(correct_letter) - ord('A')
            correct_option = options[correct_index]
            questions.append(Question(question, options, correct_option))
        i += 1
    return questions

# Condition Functions for Conditional Edges
def validate_input_condition(state: State) -> str:
    """Determine next node after input validation."""
    return END if state.status == "error" else "formulate_query"

def summarize_condition(state: State) -> str:
    """Determine next node after summarization."""
    return "generate_quiz" if state.status == "quiz" else END

# Build the LangGraph Workflow
graph = StateGraph(State)

# Add nodes to the graph
graph.add_node("validate_input", validate_input_node)
graph.add_node("formulate_query", formulate_query_node)
graph.add_node("call_search_tool", call_search_tool_node)
graph.add_node("summarize", summarize_node)
graph.add_node("generate_quiz", generate_quiz_node)

# Set the entry point
graph.set_entry_point("validate_input")

# Define edges
graph.add_conditional_edges("validate_input", validate_input_condition)
graph.add_edge("formulate_query", "call_search_tool")
graph.add_edge("call_search_tool", "summarize")
graph.add_conditional_edges("summarize", summarize_condition)
graph.add_edge("generate_quiz", END)

# Compile the graph
app = graph.compile()

# Sample Usage
if __name__ == "__main__":
    # Initialize state with sample input
    initial_state = State()
    initial_state.input = NewsInput(topic="technology", region="USA", number_of_days=7, quiz=True)
    
    # Run the agent
    final_state = app.invoke(initial_state)
    
    # Construct output from final state
    if final_state.status == "complete":
        output = NewsOutput(summary=final_state.summary, quiz=final_state.quiz)
    elif final_state.status == "error":
        output = NewsOutput(summary=final_state.summary)
    else:
        output = NewsOutput(summary="Unexpected state.")
    
    # Print the results
    print("Summary:")
    print(output.summary)
    if output.quiz:
        print("\nQuiz:")
        for idx, q in enumerate(output.quiz, 1):
            print(f"Question {idx}: {q.question}")
            for opt in q.options:
                print(f"  {opt}")
            print(f"  Correct: {q.correct_option}")


# mistakes 
"""
- A large bug some data type is missing, hard to debug, will take a look once I get some time. 
"""
