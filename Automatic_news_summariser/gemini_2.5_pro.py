import os
import json
from datetime import datetime, timedelta
from typing import TypedDict, List, Optional, Dict, Any, Sequence, Tuple

# Using Pydantic V1 for compatibility with langchain's structured output methods
import pydantic.v1 as pydantic # Use v1 namespace explicitly

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser, JsonKeyOutputFunctionsParser
# from langchain_core.pydantic_v1 import PydanticOutputFunctionsParser # Not strictly needed if using JsonKeyOutputFunctionsParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig # For potential config passing if needed
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END

# --- Configuration ---

# IMPORTANT: Set these environment variables before running the script
# Example: export GROQ_API_KEY="your_groq_api_key"
# Example: export TAVILY_API_KEY="your_tavily_api_key"

os.environ["GROQ_API_KEY"] = 'GROQ_API_KEY'
os.environ["TAVILY_API_KEY"]= "TAVILY_API_KEY"

# Check if keys were actually found and print warnings if missing

LLM_MODEL_NAME = "llama-3.3-70b-versatile" # Groq model identifier

MAX_ARTICLES_TO_PROCESS = 3
NUM_QUIZ_QUESTIONS = 3
MAX_EXPANSION_ATTEMPTS = 1 # Limit context expansion loops

# --- Data Structures ---

class QuizQuestion(pydantic.BaseModel):
    """Represents a single multiple-choice question."""
    question: str = pydantic.Field(description="The text of the question, focusing on core concepts.")
    options: List[str] = pydantic.Field(description=f"A list of exactly 4 possible answer strings.")
    correct_option: str = pydantic.Field(description="The string corresponding to the correct answer, which must be one of the provided options.")

    @pydantic.validator('correct_option')
    def correct_option_must_be_in_options(cls, v, values):
        if 'options' in values and v not in values['options']:
            raise ValueError(f"Correct option '{v}' is not in the provided options: {values.get('options')}")
        return v

    @pydantic.validator('options')
    def options_must_have_four_elements(cls, v):
        if len(v) != 4:
            raise ValueError(f'Options list must contain exactly 4 elements, got {len(v)}')
        return v

class AgentState(TypedDict):
    """Represents the state of the news summarizer agent."""
    input_topic: str
    input_num_days: int
    input_region: str
    initial_search_query: Optional[str]
    search_results: Optional[List[Dict[str, Any]]]
    processed_articles: Optional[List[Dict[str, str]]] # [{'url': ..., 'content': ...}]
    needs_expansion: bool
    expansion_query: Optional[str]
    context_reasoning: Optional[str]
    summary: Optional[str]
    quiz: Optional[List[QuizQuestion]]
    error: Optional[str]
    current_step: Optional[str]
    current_expansion_attempts: int
    max_expansion_attempts: int

# --- Tools and LLM ---

tavily_tool = TavilySearchResults(
    max_results=5)

llm = ChatGroq(
    temperature=0,
    model_name=LLM_MODEL_NAME,
    )

# --- Prompts and Chains ---

search_query_template = "Find relevant news articles about '{topic}' in '{region}' from the past {num_days} days."

summarizer_prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     """You are a skilled news journalist tasked with simplifying complex information.
Given the following news article text(s) (snippets or full text), synthesize this information into a single, coherent, and easy-to-understand summary.
Focus on the core narrative, key events, and most important developments mentioned across the provided articles.
Provide necessary background only if it's directly mentioned or clearly implied in the texts. The summary should be suitable for someone unfamiliar with the details.
Ensure the summary is concise yet comprehensive based *only* on the information given."""),
    ("user", "Here are the news articles/snippets:\n\n---\n{article_texts}\n---\n\nPlease provide the summary.")
])
summarizer_chain = (summarizer_prompt_template | llm | StrOutputParser()) if llm else None

# Context Analysis
context_analysis_prompt_template = ChatPromptTemplate.from_messages([
     ("system", "Analyze the following news content (summary/articles) and determine if it provides reasonably complete context for understanding the main news topic discussed. Consider if crucial background, related events, or key details seem missing. Use the provided 'ContextAnalysis' function structure for your response."),
     ("user", "Content to analyze:\n---\n{content_to_analyze}\n---")
])
class ContextAnalysis(pydantic.BaseModel):
    """Structure for context sufficiency analysis result."""
    sufficient_context: bool = pydantic.Field(description="True if context is reasonably complete, False otherwise.")
    reasoning: str = pydantic.Field(description="Briefly explain why the context is sufficient or insufficient.")

context_analysis_chain = (
    context_analysis_prompt_template
    # Using bind_tools which is flexible for Langchain > 0.1.14
    # It implicitly uses OpenAI functions format when model supports it.
    | llm.bind_tools([ContextAnalysis], tool_choice="ContextAnalysis") # Force calling this tool
    # Extract the arguments directly from the tool call
    | (lambda msg: msg.tool_calls[0]['args'] if msg.tool_calls else {})
) if llm else None


# Expansion Query
expansion_query_prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     """Based on the following news content, which was deemed to have insufficient context:
---
{current_articles_or_summary}
---
And considering the initial search was related to "{topic}" in "{region}" for the last {num_days} days.

The reasoning provided for insufficient context was: '{reasoning}'.

Identify the key missing information needed to provide a more complete understanding based on this reasoning.
Formulate a concise and specific Tavily search query that would likely find this missing context. Output ONLY the search query string."""),
    ("user", "Generate the new search query.")
])
expansion_chain = (expansion_query_prompt_template | llm | StrOutputParser()) if llm else None

# Quiz Creator
quiz_prompt_template = ChatPromptTemplate.from_messages([
    ("system",
    f"""Given the following news summary/article content, create a list of exactly {NUM_QUIZ_QUESTIONS} multiple-choice questions (MCQs).
These questions should test comprehension of the *core concepts* and main points presented in the text.
Do NOT focus on trivial facts, dates, or specific numbers unless they are central to the main point.
Each question must have exactly 4 plausible options, with only one being correct.
Ensure the 'correct_option' value exactly matches one of the strings in the 'options' list.
Use the provided 'QuizQuestion' tool/function structure for *each* question in the list. Output a valid JSON list containing these structures."""),
    ("user", "News Content:\n---\n{summary_or_articles}\n---\n\nGenerate the quiz questions now.")
])

# Use bind_tools with the Pydantic model and a parser to extract the list
quiz_chain = (
    quiz_prompt_template
    | llm.bind_tools([QuizQuestion]) # Bind the Pydantic model as a tool
    # Assuming the LLM might call the tool multiple times for the list,
    # we extract args from all calls and validate/parse them.
    | (lambda msg: [QuizQuestion(**call['args']) for call in msg.tool_calls if call.get('name') == QuizQuestion.__name__] if msg.tool_calls else [])
) if llm else None


# --- Agent Nodes ---

def setup_initial_state(state: AgentState) -> AgentState:
    """Initializes the AgentState with user inputs and defaults."""
    print("--- Setting up Initial State ---")
    # Inputs are expected to be passed when invoking the graph via `app.invoke(inputs)`
    state['current_step'] = "setup_initial_state"
    state['processed_articles'] = []
    state['summary'] = None
    state['quiz'] = None
    state['error'] = None
    state['needs_expansion'] = False
    state['expansion_query'] = None
    state['context_reasoning'] = None
    state['current_expansion_attempts'] = 0
    state['max_expansion_attempts'] = MAX_EXPANSION_ATTEMPTS
    print(f"   Inputs: Topic='{state.get('input_topic', 'N/A')}', Days={state.get('input_num_days', 'N/A')}, Region='{state.get('input_region', 'N/A')}'")
    return state

def formulate_search_query(state: AgentState) -> AgentState:
    """Creates the initial search query string."""
    print("--- Formulating Initial Search Query ---")
    state['current_step'] = "formulate_search_query"
    if state.get('error'): return state # Skip if error occurred earlier

    query = search_query_template.format(
        topic=state['input_topic'],
        region=state['input_region'],
        num_days=state['input_num_days']
    )
    state['initial_search_query'] = query
    print(f"   Initial Query: {query}")
    return state

def search_news(state: AgentState) -> AgentState:
    """Executes the search using the appropriate query."""
    print("--- Searching News ---")
    state['current_step'] = "search_news"
    if state.get('error'): return state
    if not tavily_tool:
        state['error'] = "Tavily Search Tool not configured (API key missing)."
        print(f"   Error: {state['error']}")
        return state

    # Use expansion query if available, otherwise use initial query
    query = state.get('expansion_query', state.get('initial_search_query'))
    if not query:
        state['error'] = "No search query available."
        print(f"   Error: {state['error']}")
        return state

    print(f"   Executing search for: \"{query}\"")
    try:
        # Tavily returns a list of dicts with 'url' and 'content' (snippet)
        results = tavily_tool.invoke({"query": query}) # Pass query in a dict
        state['search_results'] = results if results else [] # Ensure it's a list
        print(f"   Found {len(state['search_results'])} results.")
        # Reset expansion query after use
        state['expansion_query'] = None

    except Exception as e:
        state['error'] = f"Tavily search failed: {e}"
        print(f"   Error: {state['error']}")
        state['search_results'] = [] # Ensure it's an empty list on error

    return state

def process_articles(state: AgentState) -> AgentState:
    """Filters results and prepares article content (using snippets)."""
    print("--- Processing Articles ---")
    state['current_step'] = "process_articles"
    if state.get('error'): return state

    results = state.get('search_results', [])

    if not results:
        print("   No search results to process.")
        # Decide if this is an error or just no news. We'll proceed for now,
        # downstream nodes should handle empty processed_articles.
        state['processed_articles'] = []
        return state

    processed = []
    print(f"   Processing top {MAX_ARTICLES_TO_PROCESS} results...")
    # In a real scenario, you'd fetch full text here. For simplicity, we use the snippet ('content')
    # Also, add basic deduplication based on URL.
    seen_urls = set()
    articles_from_this_search = []
    for article in results:
        url = article.get('url')
        content = article.get('content') # Tavily usually provides snippet here
        if url and content and url not in seen_urls:
            articles_from_this_search.append({"url": url, "content": content})
            seen_urls.add(url)
            if len(articles_from_this_search) >= MAX_ARTICLES_TO_PROCESS:
                break

    # If this was an expansion search, merge results (optional, could also replace)
    # For simplicity, we'll just use the latest results for now.
    state['processed_articles'] = articles_from_this_search

    print(f"   Selected {len(state['processed_articles'])} articles for processing.")
    if not state['processed_articles']:
         print("   Warning: No usable articles found after processing.")
         # Optional: Set error if no articles could be processed at all
         # state['error'] = "Could not process any articles from search results."

    return state

def summarize_news(state: AgentState) -> AgentState:
    """Generates a summary from the processed articles."""
    print("--- Summarizing News ---")
    state['current_step'] = "summarize_news"
    if state.get('error'): return state
    if not llm or not summarizer_chain:
        state['error'] = "LLM or Summarizer Chain not configured."
        print(f"   Error: {state['error']}")
        return state

    articles = state.get('processed_articles', [])
    if not articles:
        print("   No articles to summarize.")
        state['summary'] = "No relevant news articles found or processed."
        # We don't treat this as an error, just report no summary possible
        return state

    # Combine content for the prompt
    article_texts = "\n\n---\n\n".join([f"Article URL: {a['url']}\n\nSnippet:\n{a['content']}" for a in articles])

    print("   Generating summary...")
    try:
        summary = summarizer_chain.invoke({"article_texts": article_texts})
        state['summary'] = summary
        print("   Summary generated.")
    except Exception as e:
        state['error'] = f"Summarization failed: {e}"
        print(f"   Error: {state['error']}")
        state['summary'] = "Error during summarization."

    return state

def check_context_sufficiency(state: AgentState) -> AgentState:
    """Checks if the summary/articles have enough context."""
    print("--- Checking Context Sufficiency ---")
    state['current_step'] = "check_context_sufficiency"
    if state.get('error'): return state
    if not llm or not context_analysis_chain:
        state['error'] = "LLM or Context Analysis Chain not configured."
        print(f"   Error: {state['error']}")
        return state

    content_to_analyze = state.get('summary') or ""
    if not content_to_analyze and state.get('processed_articles'):
         # Fallback to using article snippets if summary failed or is empty
         content_to_analyze = "\n\n---\n\n".join([f"Article URL: {a['url']}\n\nSnippet:\n{a['content']}" for a in state['processed_articles']])

    if not content_to_analyze:
        print("   No content available to analyze context.")
        # Assume insufficient if no content, prevents quiz generation on nothing
        state['needs_expansion'] = True
        state['context_reasoning'] = "No content was generated or processed."
        return state

    print("   Analyzing context...")
    try:
        # The chain now directly returns the dictionary parsed from the tool call args
        analysis_result = context_analysis_chain.invoke({"content_to_analyze": content_to_analyze})

        if isinstance(analysis_result, dict) and 'sufficient_context' in analysis_result:
            state['needs_expansion'] = not analysis_result.get('sufficient_context', True) # Default to sufficient if key missing after parse
            state['context_reasoning'] = analysis_result.get('reasoning', 'No reasoning provided.')
            print(f"   Context sufficient: {not state['needs_expansion']}. Reasoning: {state['context_reasoning']}")
        else:
            raise ValueError(f"Unexpected output format from context analysis: {analysis_result}")

    except Exception as e:
        state['error'] = f"Context sufficiency check failed: {e}"
        print(f"   Error: {state['error']}")
        # Default to needing expansion on error to be safe, or handle differently
        state['needs_expansion'] = True
        state['context_reasoning'] = f"Error during analysis: {e}"

    # Increment expansion attempts *after* checking
    state['current_expansion_attempts'] += 1

    return state

def formulate_expansion_query(state: AgentState) -> AgentState:
    """Generates a new query if context was insufficient."""
    print("--- Formulating Expansion Query ---")
    state['current_step'] = "formulate_expansion_query"
    if state.get('error'): return state
    if not llm or not expansion_chain:
        state['error'] = "LLM or Expansion Chain not configured."
        print(f"   Error: {state['error']}")
        return state

    # Use summary if available, else article snippets
    current_content = state.get('summary') or "\n\n---\n\n".join([a['content'] for a in state.get('processed_articles', [])])
    if not current_content:
         state['error'] = "No content available to formulate expansion query."
         print(f"   Error: {state['error']}")
         return state

    print("   Generating expansion query...")
    try:
        new_query = expansion_chain.invoke({
            "current_articles_or_summary": current_content,
            "topic": state['input_topic'],
            "region": state['input_region'],
            "num_days": state['input_num_days'],
            "reasoning": state.get('context_reasoning', 'Context was deemed insufficient.')
        })
        state['expansion_query'] = new_query.strip()
        print(f"   Expansion Query: {state['expansion_query']}")
    except Exception as e:
        state['error'] = f"Expansion query formulation failed: {e}"
        print(f"   Error: {state['error']}")
        # Prevent retry loop if formulation fails
        state['needs_expansion'] = False # Set to false to avoid retrying search

    return state

def generate_quiz(state: AgentState) -> AgentState:
    """Generates the quiz based on the final summary or articles."""
    print("--- Generating Quiz ---")
    state['current_step'] = "generate_quiz"
    if state.get('error'): return state
    if not llm or not quiz_chain:
        state['error'] = "LLM or Quiz Chain not configured."
        print(f"   Error: {state['error']}")
        return state

    # Use summary if available, else article snippets
    content_for_quiz = state.get('summary')
    if not content_for_quiz and state.get('processed_articles'):
         print("   Using processed article snippets for quiz generation (summary was empty/failed).")
         content_for_quiz = "\n\n---\n\n".join([f"Article URL: {a['url']}\n\nSnippet:\n{a['content']}" for a in state['processed_articles']])

    if not content_for_quiz:
        state['error'] = "No content available to generate quiz."
        print(f"   Error: {state['error']}")
        state['quiz'] = []
        return state

    print(f"   Generating {NUM_QUIZ_QUESTIONS} quiz questions...")
    try:
        # The chain now directly returns the list of parsed Pydantic objects
        quiz_list = quiz_chain.invoke({"summary_or_articles": content_for_quiz})

        # Validate the output structure (list of QuizQuestion objects)
        if isinstance(quiz_list, list) and all(isinstance(q, QuizQuestion) for q in quiz_list):
             # Ensure we have the correct number of questions
             if len(quiz_list) == NUM_QUIZ_QUESTIONS:
                 state['quiz'] = quiz_list
                 print(f"   Successfully generated {len(quiz_list)} quiz questions.")
             else:
                 # Handle case where LLM didn't return exactly the right number
                 print(f"   Warning: LLM generated {len(quiz_list)} questions, expected {NUM_QUIZ_QUESTIONS}. Using generated questions.")
                 state['quiz'] = quiz_list # Keep what was generated or decide to error/retry
        else:
            # Handle unexpected format or parsing errors
            raise TypeError(f"Quiz generation returned unexpected type or content: {type(quiz_list)}")

    except Exception as e:
        state['error'] = f"Quiz generation failed: {e}"
        print(f"   Error: {state['error']}")
        state['quiz'] = [] # Ensure it's an empty list on error

    return state

# --- Conditional Edges ---

def decide_on_expansion(state: AgentState) -> str:
    """Determines the next step after checking context sufficiency."""
    print("--- Deciding on Expansion ---")
    if state.get('error'):
        print("   Error detected, ending.")
        return END # Go to end if any node reported an error

    needs_expansion = state.get('needs_expansion', False)
    current_attempts = state.get('current_expansion_attempts', 0)
    max_attempts = state.get('max_expansion_attempts', 1)

    if needs_expansion and current_attempts <= max_attempts:
        print(f"   Context insufficient, attempt {current_attempts}/{max_attempts}. Formulating expansion query.")
        return "formulate_expansion_query" # Needs expansion and within attempt limit
    elif needs_expansion:
        print(f"   Context insufficient, but max expansion attempts ({max_attempts}) reached. Proceeding to quiz generation.")
        return "generate_quiz" # Needs expansion but limit reached
    else:
        print("   Context sufficient. Proceeding to quiz generation.")
        return "generate_quiz" # Context is sufficient

def decide_after_processing(state: AgentState) -> str:
    """Decides whether to summarize or end if no articles were processed."""
    print("--- Deciding After Processing ---")
    if state.get('error'):
        print("   Error detected, ending.")
        return END
    if not state.get('processed_articles'):
        print("   No articles processed, ending run.")
        # Optionally set a final message/error here if desired
        state['summary'] = "No relevant news articles found or processed."
        state['quiz'] = []
        return END
    else:
        print("   Articles processed, proceeding to summarization.")
        return "summarize_news"

# --- Build the Graph ---

builder = StateGraph(AgentState)

# Add nodes
builder.add_node("setup_initial_state", setup_initial_state)
builder.add_node("formulate_search_query", formulate_search_query)
builder.add_node("search_news", search_news)
builder.add_node("process_articles", process_articles)
builder.add_node("summarize_news", summarize_news)
builder.add_node("check_context_sufficiency", check_context_sufficiency)
builder.add_node("formulate_expansion_query", formulate_expansion_query)
builder.add_node("generate_quiz", generate_quiz)

# Define edges
builder.set_entry_point("setup_initial_state")
builder.add_edge("setup_initial_state", "formulate_search_query")
builder.add_edge("formulate_search_query", "search_news")
builder.add_edge("search_news", "process_articles")

# Conditional edge after processing articles
builder.add_conditional_edges(
    "process_articles",
    decide_after_processing,
    {
        "summarize_news": "summarize_news",
        END: END
    }
)

builder.add_edge("summarize_news", "check_context_sufficiency")

# Conditional edge after checking context
builder.add_conditional_edges(
    "check_context_sufficiency",
    decide_on_expansion,
    {
        "formulate_expansion_query": "formulate_expansion_query",
        "generate_quiz": "generate_quiz",
        END: END # Added END path if error occurred
    }
)

# Loop back from expansion query formulation to search
builder.add_edge("formulate_expansion_query", "search_news")

# Final step leads to END
builder.add_edge("generate_quiz", END)


# Compile the graph
app = builder.compile()

# --- Main Execution Block ---

if __name__ == "__main__":

    print("\n--- Starting News Summarizer Agent ---")

    # Example Inputs
    inputs = {
        "input_topic": "latest developments in renewable energy policy",
        "input_num_days": 7,
        "input_region": "Europe"
    }

    # inputs = {
    #     "input_topic": "Apple Vision Pro reception",
    #     "input_num_days": 3,
    #     "input_region": "Global"
    # }

    # inputs = {
    #      "input_topic": "Recent Mars rover findings",
    #      "input_num_days": 30,
    #      "input_region": "Space" # Test region handling
    # }


    # Stream events for debugging (optional)
    # config = RunnableConfig(recursion_limit=50) # Increase recursion limit if needed
    # for event in app.stream(inputs, config=config):
    #     for node, output in event.items():
    #         print(f"--- Output from node: {node} ---")
    #         # print(output) # Can be verbose
    #         print("{")
    #         for key, value in output.items():
    #              # Truncate long values for cleaner logging
    #              if isinstance(value, str) and len(value) > 200:
    #                  print(f"  '{key}': '{value[:200]}...'")
    #              elif isinstance(value, list) and len(value) > 3:
    #                   print(f"  '{key}': {value[:3]}... (Total: {len(value)})")
    #              else:
    #                  print(f"  '{key}': {value}")
    #         print("}")


    # Or just invoke and get the final state
    final_state = app.invoke(inputs)

    print("\n--- Agent Run Finished ---")

    if final_state.get('error'):
        print(f"\nERROR during execution: {final_state['error']}")

    print("\n--- Final Summary ---")
    print(final_state.get('summary', "No summary generated."))

    print("\n--- Final Quiz ---")
    quiz_questions = final_state.get('quiz')
    if quiz_questions:
        for i, q in enumerate(quiz_questions):
            print(f"\nQ{i+1}: {q.question}")
            for j, opt in enumerate(q.options):
                print(f"  {chr(ord('A') + j)}. {opt}")
            print(f"  Correct Answer: {q.correct_option}")
    else:
        print("No quiz generated.")

    # print("\n--- Full Final State ---")
    # print(json.dumps(final_state, indent=2, default=str)) # Use default=str for non-serializable types like Pydantic models
