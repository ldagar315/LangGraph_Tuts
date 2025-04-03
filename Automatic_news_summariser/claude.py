import os
from typing import List, TypedDict, Optional, Dict, Any, Annotated, Sequence
from datetime import datetime, timedelta

import json
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
import operator
from tavily import TavilyClient
from datetime import datetime, timedelta

from groq import Groq
import requests
from bs4 import BeautifulSoup
import time

os.environ["GROQ_API_KEY"] = 'GROQ_API_KEY'
os.environ["TAVILY_API_KEY"]= "TAVILY_API_KEY"
# Initialize clients
groq_client = Groq()
tavily_client = TavilyClient()

# Data Models
class QuizQuestion(BaseModel):
    """Represents a single multiple-choice question."""
    question: str = Field(description="The text of the question, focusing on core concepts.")
    options: List[str] = Field(description="A list of exactly 4 possible answer strings.")
    correct_option: str = Field(description="The string corresponding to the correct answer, which must be one of the provided options.")

class AgentState(TypedDict):
    """Represents the state of the news summarizer agent."""
    # Inputs
    input_topic: str
    input_num_days: int
    input_region: str

    # Intermediate results
    initial_search_query: Optional[str]
    search_results: Optional[List[Dict[str, Any]]]  # Raw results from Tavily
    processed_articles: Optional[List[Dict[str, str]]]  # Key articles selected
    needs_expansion: bool  # Flag to indicate if more context is needed
    expansion_query: Optional[str]
    expansion_attempts: int  # Count expansion attempts to prevent infinite loops

    # Outputs
    summary: Optional[str]
    quiz: Optional[List[QuizQuestion]]

    # Control flow & Error handling
    error: Optional[str]
    current_step: Optional[str]  # To track progress/debugging


# Helper Functions
def truncate_text(text, max_length=10000):
    """Truncate text to a maximum length to avoid LLM context limitations."""
    if len(text) > max_length:
        return text[:max_length] + "... [truncated]"
    return text

def extract_article_text(url, timeout=10):
    """Extract the main text content from a news article URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            script_or_style.decompose()
            
        # Find the main article content
        # Different sites structure their content differently, so we try common patterns
        article_elements = soup.find_all(['article', 'main', '.content', '.article', '.post'])
        
        if article_elements:
            main_content = article_elements[0].get_text(separator=" ", strip=True)
        else:
            # If no specific article element, get text from paragraphs
            paragraphs = soup.find_all('p')
            main_content = " ".join([p.get_text(strip=True) for p in paragraphs])
            
        # Clean up the text
        main_content = " ".join(main_content.split())
        
        return main_content if main_content else "Failed to extract article content."
    
    except Exception as e:
        return f"Error extracting content: {str(e)}"

def call_llm(prompt, model="llama-3.3-70b-versatile", max_tokens=4000, temperature=0.5):
    """Call the ChatGroq LLM with a prompt."""
    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return f"Error: {str(e)}"

# Node Functions
def setup_initial_state(user_input) -> AgentState:
    """Initialize the agent state with user inputs."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "input_topic": user_input.get("topic_of_interest", ""),
        "input_num_days": user_input.get("num_days"),  # Default to 7 days
        "input_region": user_input.get("region_of_interest"),  # Default to Global
        "initial_search_query": None,
        "search_results": None,
        "processed_articles": None,
        "needs_expansion": False,
        "expansion_query": None,
        "expansion_attempts": 0,
        "summary": None,
        "quiz": None,
        "error": None,
        "current_step": f"Initialized at {current_time}"
    }

def formulate_search_query(state: AgentState) -> AgentState:
    """Formulate the initial search query based on user inputs."""
    topic = state["input_topic"]
    region = state["input_region"]
    days = state["input_num_days"]
    
    # Template-based approach for search query
    query = f"Latest news about {topic} in {region} from the past {days} days"
    
    # Update state
    new_state = state.copy()
    new_state["initial_search_query"] = query
    new_state["current_step"] = "Formulated search query"
    return new_state

def search_news(state: AgentState) -> AgentState:
    """Execute search using Tavily API."""
    new_state = state.copy()
    
    try:
        # Determine which query to use
        if state["expansion_query"] and state["expansion_attempts"] > 0:
            query = state["expansion_query"]
            search_type = "news"  # Use news search for expansion queries
        else:
            query = state["initial_search_query"]
            search_type = "news"  # Use news search type for news articles
        
        # Calculate date range
        days_ago = state["input_num_days"]
        max_age_mins = days_ago * 24 * 60  # Convert days to minutes
        
        # Execute Tavily search
        search_results = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,  # Get top 5 results to filter later
            include_domains=None,  # Could be customized in future
            exclude_domains=None,
            max_age_mins=max_age_mins,
            search_type=search_type
        )
        
        # Update state
        new_state["search_results"] = search_results.get("results", [])
        new_state["current_step"] = "Executed search"
        
        if not new_state["search_results"]:
            new_state["error"] = f"No search results found for query: {query}"
        
    except Exception as e:
        new_state["error"] = f"Search failed: {str(e)}"
        new_state["search_results"] = []
    
    return new_state

def process_articles(state: AgentState) -> AgentState:
    """Process and fetch content for the top articles."""
    new_state = state.copy()
    search_results = state.get("search_results", [])
    
    if not search_results:
        new_state["error"] = "No search results to process"
        new_state["processed_articles"] = []
        return new_state
    
    # Sort by potential relevance (we'll use the score if available)
    sorted_results = sorted(
        search_results, 
        key=lambda x: x.get("score", 0), 
        reverse=True
    )
    
    # Take top 3 results
    top_results = sorted_results[:3]
    
    processed_articles = []
    for article in top_results:
        url = article.get("url")
        snippet = article.get("content", "No snippet available")
        
        try:
            # Try to fetch the full text
            full_text = extract_article_text(url)
            # Truncate to avoid LLM context limitations
            full_text = truncate_text(full_text)
            
            processed_articles.append({
                "url": url,
                "title": article.get("title", "Unknown Title"),
                "snippet": snippet,
                "full_text": full_text
            })
        except Exception as e:
            # If extraction fails, just use the snippet
            processed_articles.append({
                "url": url,
                "title": article.get("title", "Unknown Title"),
                "snippet": snippet,
                "full_text": f"Failed to extract: {str(e)}. Using snippet instead: {snippet}"
            })
    
    new_state["processed_articles"] = processed_articles
    new_state["current_step"] = "Processed articles"
    
    if not processed_articles:
        new_state["error"] = "Failed to process any articles"
    
    return new_state

def summarize_news(state: AgentState) -> AgentState:
    """Summarize the processed articles."""
    new_state = state.copy()
    processed_articles = state.get("processed_articles", [])
    
    if not processed_articles:
        new_state["error"] = "No processed articles to summarize"
        return new_state
    
    # Prepare article texts for summarization
    article_texts = []
    for idx, article in enumerate(processed_articles, 1):
        title = article.get("title", "Untitled")
        source = article.get("url", "Unknown Source")
        # Prefer full text if available, otherwise use snippet
        content = article.get("full_text") if article.get("full_text") else article.get("snippet", "")
        
        article_texts.append(f"ARTICLE {idx}: {title}\nSource: {source}\n\n{content}\n")
    
    all_articles = "\n---\n".join(article_texts)
    
    # Truncate to fit LLM context window if necessary
    all_articles = truncate_text(all_articles)
    
    # Create summarization prompt
    prompt = f"""You are a skilled news journalist tasked with simplifying complex information.
Given the following news article text(s):
---
{all_articles}
---
Synthesize this information into a single, coherent, and easy-to-understand summary. 
Break down complex topics, explain key events clearly, and provide necessary background 
if it's a long-running issue mentioned in the text. Focus on the core narrative and most 
important developments. The summary should be suitable for someone unfamiliar with the details.
"""
    
    # Call LLM for summarization
    summary = call_llm(prompt)
    
    new_state["summary"] = summary
    new_state["current_step"] = "Generated summary"
    
    return new_state

def check_context_sufficiency(state: AgentState) -> AgentState:
    """Check if the current context is sufficient or needs expansion."""
    new_state = state.copy()
    summary = state.get("summary", "")
    processed_articles = state.get("processed_articles", [])
    
    if not summary or not processed_articles:
        new_state["error"] = "Missing summary or processed articles for context check"
        new_state["needs_expansion"] = False  # Can't expand without a base
        return new_state
    
    # Prepare content for analysis
    article_snippets = [article.get("snippet", "") for article in processed_articles]
    content_to_analyze = summary + "\n\nBased on articles covering: " + "; ".join(article_snippets[:200])  # Limit size
    
    # Create context analysis prompt
    prompt = f"""Analyze the following news content (articles/summary):
---
{content_to_analyze}
---
Evaluate if this content provides a reasonably complete context for understanding the main news topic discussed. 
Consider if crucial background, related events, or key details seem missing.

Respond in JSON format with the following structure ONLY:
{{
  "sufficient_context": boolean, // true if context is reasonably complete, false otherwise
  "reasoning": "Briefly explain why the context is sufficient or insufficient." // Optional brief justification
}}
"""
    
    # Call LLM for context analysis
    result = call_llm(prompt)
    
    # Parse JSON result
    try:
        result_json = json.loads(result)
        sufficient_context = result_json.get("sufficient_context", True)  # Default to True if parsing fails
        
        new_state["needs_expansion"] = not sufficient_context
        new_state["current_step"] = f"Context check: {'Sufficient' if sufficient_context else 'Insufficient'}"
        
    except json.JSONDecodeError:
        # Handle case where LLM output isn't valid JSON
        # Default to not needing expansion if we can't parse the result
        new_state["needs_expansion"] = False
        new_state["current_step"] = "Context check failed, defaulting to sufficient"
    
    return new_state

def formulate_expansion_query(state: AgentState) -> AgentState:
    """Formulate a query to expand the context if needed."""
    new_state = state.copy()
    
    # Increment expansion attempts counter
    new_state["expansion_attempts"] = state.get("expansion_attempts", 0) + 1
    
    # Check if we've hit the maximum number of expansion attempts
    if new_state["expansion_attempts"] > 2:  # Limit to 2 expansion attempts
        new_state["needs_expansion"] = False
        new_state["current_step"] = "Max expansion attempts reached"
        return new_state
    
    summary = state.get("summary", "")
    initial_query = state.get("initial_search_query", "")
    topic = state.get("input_topic", "")
    region = state.get("input_region", "")
    days = state.get("input_num_days", 7)
    
    # Create expansion query prompt
    prompt = f"""Based on the following news content, which was deemed to have insufficient context:
---
{summary}
---
And considering the initial search was related to "{topic}" in "{region}" for the last {days} days, with query: "{initial_query}"

Identify the key missing information needed to provide a more complete understanding. Formulate a concise and specific 
search query that would likely find this missing context. Output ONLY the search query string.
"""
    
    # Call LLM for expansion query
    expansion_query = call_llm(prompt, max_tokens=200, temperature=0.3)
    
    new_state["expansion_query"] = expansion_query
    new_state["current_step"] = f"Formulated expansion query: {expansion_query}"
    
    return new_state

def generate_quiz(state: AgentState) -> AgentState:
    """Generate a quiz based on the summary and articles."""
    new_state = state.copy()
    summary = state.get("summary", "")
    processed_articles = state.get("processed_articles", [])
    
    if not summary:
        new_state["error"] = "Missing summary for quiz generation"
        return new_state
    
    # Prepare content for quiz generation
    content_for_quiz = summary
    
    # Include snippets from articles if available
    if processed_articles:
        snippets = "\n\nAdditional context from articles:\n"
        for idx, article in enumerate(processed_articles, 1):
            title = article.get("title", "Untitled")
            snippets += f"{idx}. {title}: {article.get('snippet', '')}\n"
        
        content_for_quiz += snippets
    
    # Create quiz generation prompt
    prompt = f"""Given the following news summary/article content:
---
{content_for_quiz}
---
Create a list of exactly 3 multiple-choice questions (MCQs) designed to test comprehension of the *core concepts* 
and main points presented in the text. Do NOT focus on trivial facts, dates, or specific numbers unless they are 
central to the main point. Each question should have 4 plausible options, with only one being correct.

Output the questions strictly as a JSON list containing objects that conform to the following Pydantic model:

class QuizQuestion(BaseModel):
    question: str = Field(description="The text of the question, focusing on core concepts.")
    options: List[str] = Field(description="A list of exactly 4 possible answer strings.")
    correct_option: str = Field(description="The string corresponding to the correct answer, which must be one of the provided options.")

Example Format:
[
  {{
    "question": "What was the main outcome of the reported event?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_option": "Option B"
  }},
  // ... more questions
]

Generate the JSON list now.
"""
    
    # Call LLM for quiz generation
    quiz_response = call_llm(prompt, max_tokens=2000, temperature=0.7)
    
    # Parse quiz JSON
    try:
        # Try to extract JSON from the response if it's not a clean JSON
        if not quiz_response.strip().startswith('['):
            # Find JSON array in the response
            start_idx = quiz_response.find('[')
            end_idx = quiz_response.rfind(']') + 1
            if start_idx >= 0 and end_idx > start_idx:
                quiz_response = quiz_response[start_idx:end_idx]
        
        quiz_questions = json.loads(quiz_response)
        
        # Validate the structure
        processed_quiz = []
        for q in quiz_questions:
            if isinstance(q, dict) and "question" in q and "options" in q and "correct_option" in q:
                # Ensure correct_option is in options
                if q["correct_option"] not in q["options"]:
                    raise ValueError(f"Correct option '{q['correct_option']}' not in options list")
                
                processed_quiz.append(QuizQuestion(
                    question=q["question"],
                    options=q["options"],
                    correct_option=q["correct_option"]
                ))
        
        # Ensure we have exactly 3 questions
        if len(processed_quiz) < 3:
            # If we don't have enough questions, pad with simple ones from the summary
            while len(processed_quiz) < 3:
                # Generate a simple question about the summary
                default_q = QuizQuestion(
                    question=f"What is a key point mentioned in the news summary? (Question {len(processed_quiz)+1})",
                    options=[
                        "The information is not in the summary", 
                        "This is mentioned in the summary", 
                        "The summary does not cover this topic", 
                        "This is contradicted by the summary"
                    ],
                    correct_option="This is mentioned in the summary"
                )
                processed_quiz.append(default_q)
        elif len(processed_quiz) > 3:
            # If we have too many, take only the first 3
            processed_quiz = processed_quiz[:3]
        
        new_state["quiz"] = processed_quiz
        new_state["current_step"] = "Generated quiz"
        
    except Exception as e:
        new_state["error"] = f"Failed to generate valid quiz: {str(e)}"
        # Create a default quiz in case of failure
        default_quiz = [
            QuizQuestion(
                question="What is the main topic of the news summary?",
                options=[
                    f"News about {state['input_topic']}",
                    "Sports news",
                    "Weather forecast",
                    "Stock market updates"
                ],
                correct_option=f"News about {state['input_topic']}"
            ),
            QuizQuestion(
                question="Which region was covered in this news summary?",
                options=[
                    state["input_region"],
                    "Antarctica",
                    "The Moon",
                    "International Space Station"
                ],
                correct_option=state["input_region"]
            ),
            QuizQuestion(
                question="How recent is the news coverage?",
                options=[
                    f"Within the past {state['input_num_days']} days",
                    "From last year",
                    "From a decade ago",
                    "Historical archives"
                ],
                correct_option=f"Within the past {state['input_num_days']} days"
            )
        ]
        new_state["quiz"] = default_quiz
    
    return new_state

# Define conditions for routing in the graph
def should_expand_context(state: AgentState) -> bool:
    """Determine if context expansion is needed."""
    return state.get("needs_expansion", False) and state.get("expansion_attempts", 0) < 3

def has_error(state: AgentState) -> bool:
    """Check if an error occurred."""
    return state.get("error") is not None

def has_articles(state: AgentState) -> bool:
    """Check if we have processed articles."""
    articles = state.get("processed_articles", [])
    return len(articles) > 0

# Build the LangGraph
def build_news_agent() -> StateGraph:
    """Build and return the LangGraph for the news agent."""
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("setup_initial_state", setup_initial_state)
    workflow.add_node("formulate_search_query", formulate_search_query)
    workflow.add_node("search_news", search_news)
    workflow.add_node("process_articles", process_articles)
    workflow.add_node("summarize_news", summarize_news)
    workflow.add_node("check_context_sufficiency", check_context_sufficiency)
    workflow.add_node("formulate_expansion_query", formulate_expansion_query)
    workflow.add_node("generate_quiz", generate_quiz)
    
    # Define edges
    workflow.add_edge("setup_initial_state", "formulate_search_query")
    workflow.add_edge("formulate_search_query", "search_news")
    workflow.add_edge("search_news", "process_articles")
    
    # Conditional edges from process_articles
    workflow.add_conditional_edges(
        "process_articles",
        has_articles,
        {
            True: "summarize_news",
            False: END  # End if no articles were processed
        }
    )
    
    workflow.add_edge("summarize_news", "check_context_sufficiency")
    
    # Conditional edges from context check
    workflow.add_conditional_edges(
        "check_context_sufficiency",
        should_expand_context,
        {
            True: "formulate_expansion_query",
            False: "generate_quiz"
        }
    )
    
    workflow.add_edge("formulate_expansion_query", "search_news")  # Loop back for more context
    workflow.add_edge("generate_quiz", END)
    
    # Set the entry point
    workflow.set_entry_point("setup_initial_state")
    graph = workflow.compile()
    
    return graph

# Main executor function
def run_news_agent(topic_of_interest: str, num_days: int = 7, region_of_interest: str = "Global"):
    """Run the news agent with the specified inputs."""
    # Build the agent
    agent = build_news_agent()
    
    # Configure inputs
    user_input = {
        "topic_of_interest": topic_of_interest,
        "num_days": num_days,
        "region_of_interest": region_of_interest
    }
    
    # Run the agent
    result = agent.invoke(user_input)
    
    # Format and return results
    if result.get("error"):
        return {
            "status": "error",
            "error_message": result["error"],
            "summary": result.get("summary", "No summary generated"),
            "quiz": result.get("quiz", [])
        }
    
    # Format quiz for display
    formatted_quiz = []
    if result.get("quiz"):
        for q in result["quiz"]:
            if isinstance(q, QuizQuestion):
                formatted_quiz.append({
                    "question": q.question,
                    "options": q.options,
                    "correct_option": q.correct_option
                })
            else:
                # If quiz is already a dict (JSON)
                formatted_quiz.append(q)
    
    return {
        "status": "success",
        "summary": result.get("summary", "No summary generated"),
        "quiz": formatted_quiz,
        "sources": [article.get("url") for article in result.get("processed_articles", [])]
    }

# Example usage
if __name__ == "__main__":
    # Example usage
    topic = "Politics"
    days = 1
    region = "India"
    
    print(f"Running news agent for topic: {topic}, past {days} days, region: {region}")
    result = run_news_agent(topic, days, region)
    
    print("\n===== NEWS SUMMARY =====")
    print(result.get("summary", "No summary available"))
    
    print("\n===== QUIZ =====")
    quiz = result.get("quiz", [])
    for i, q in enumerate(quiz, 1):
        print(f"\nQuestion {i}: {q.get('question')}")
        for j, option in enumerate(q.get('options', []), 1):
            print(f"  {j}. {option}")
        print(f"Correct answer: {q.get('correct_option')}")
    
    print("\n===== SOURCES =====")
    for source in result.get("sources", []):
        print(f"- {source}")
