import os
from langgraph.graph import Graph
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, TypedDict
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import json

os.environ["GROQ_API_KEY"] = 'GROQ_API_KEY'
os.environ["TAVILY_API_KEY"]= "TAVILY_API_KEY"
# Set up the LLM model
llm = ChatGroq(model="llama-3.3-70b-versatile")  # Add API Key

# Set up the Tavily Search API
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

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
def setup_initial_state(topic: str, num_days: int, region: str) -> AgentState:
    return {
        "input_topic": topic,
        "input_num_days": num_days,
        "input_region": region,
        "needs_expansion": False,
        "error": None,
        "current_step": "setup_initial_state"
    }

def formulate_search_query(state: AgentState) -> AgentState:
    query = f"Find news about '{state['input_topic']}' in '{state['input_region']}' from the past {state['input_num_days']} days."
    state["initial_search_query"] = query
    state["current_step"] = "formulate_search_query"
    return state

def is_likely_news_article(soup: BeautifulSoup) -> bool:
    """
    Checks HTML metadata and structure to determine if a page is likely a news article.
    Returns True if it seems like an article, False otherwise.
    """
    if not soup:
        return False

    # 1. Check Open Graph type
    og_type = soup.find("meta", property="og:type")
    if og_type and og_type.get("content", "").lower() in ["article", "news"]:
        print("DEBUG: Found og:type=article/news")
        return True

    # 2. Check for <article> tag
    if soup.find("article"):
        print("DEBUG: Found <article> tag")
        return True

    # 3. Check Schema.org metadata (JSON-LD)
    schema_scripts = soup.find_all("script", type="application/ld+json")
    for script in schema_scripts:
        try:
            data = json.loads(script.string)
            # Handle cases where data is a list or a single dictionary
            if isinstance(data, list):
                items_to_check = data
            elif isinstance(data, dict):
                items_to_check = [data]
            else:
                continue # Skip if format is unexpected

            for item in items_to_check:
                # Check main entity type or graph elements
                if isinstance(item, dict):
                    item_type = item.get("@type")
                    if isinstance(item_type, str) and item_type.lower() in ["newsarticle", "article", "report"]:
                         print(f"DEBUG: Found Schema.org type: {item_type}")
                         return True
                    # Check within @graph structure
                    graph = item.get("@graph")
                    if isinstance(graph, list):
                        for graph_item in graph:
                             if isinstance(graph_item, dict):
                                 graph_item_type = graph_item.get("@type")
                                 if isinstance(graph_item_type, str) and graph_item_type.lower() in ["newsarticle", "article", "report"]:
                                      print(f"DEBUG: Found Schema.org type in @graph: {graph_item_type}")
                                      return True

        except json.JSONDecodeError:
            # Ignore scripts with invalid JSON
            continue
        except Exception as e:
            # Catch other potential errors during schema parsing
            print(f"DEBUG: Error parsing schema JSON: {e}")
            continue

    # 4. Fallback: Check for significant paragraph content (optional, less reliable)
    # paragraphs = soup.find_all('p')
    # if len(paragraphs) > 5: # Arbitrary threshold
    #     print(f"DEBUG: Found {len(paragraphs)} <p> tags (fallback check)")
    #     return True

    print("DEBUG: Did not meet criteria for news article.")
    return False

def extract_article_text_from_soup(soup: BeautifulSoup, max_words=1000):
    """Extract the main text content and headline from a BeautifulSoup object with a word limit."""
    try:
        # Extract headline first
        headline = None
        
        # Try to find headline in common locations
        # 1. Check for h1 tags
        h1_tags = soup.find_all('h1')
        if h1_tags:
            headline = h1_tags[0].get_text(strip=True)
        
        # 2. Check for meta tags (og:title, twitter:title)
        if not headline:
            og_title = soup.find("meta", property="og:title")
            if og_title:
                headline = og_title.get("content", "")
            
            if not headline:
                twitter_title = soup.find("meta", property="twitter:title")
                if twitter_title:
                    headline = twitter_title.get("content", "")
        
        # 3. Check for title tag
        if not headline:
            title_tag = soup.find("title")
            if title_tag:
                headline = title_tag.get_text(strip=True)
        
        # 4. Check for Schema.org headline
        if not headline:
            schema_scripts = soup.find_all("script", type="application/ld+json")
            for script in schema_scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict) and "headline" in data:
                        headline = data["headline"]
                        break
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "headline" in item:
                                headline = item["headline"]
                                break
                except:
                    continue
        
        # If still no headline, use the first h1 or h2
        if not headline:
            for tag in ['h1', 'h2']:
                tag_elements = soup.find_all(tag)
                if tag_elements:
                    headline = tag_elements[0].get_text(strip=True)
                    break
        
        # If still no headline, use a default
        if not headline:
            headline = "News Article"
        
        # Remove script and style elements
        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form', 'button']):
            script_or_style.decompose()
            
        # Find the main article content
        # Different sites structure their content differently, so we try common patterns
        article_elements = soup.find_all(['article', 'main', '.content', '.article', '.post', '.entry-content', '.story-content'])
        
        if article_elements:
            main_content = article_elements[0].get_text(separator=" ", strip=True)
        else:
            # If no specific article element, get text from paragraphs
            paragraphs = soup.find_all('p')
            main_content = " ".join([p.get_text(strip=True) for p in paragraphs])
            
        # Clean up the text
        main_content = " ".join(main_content.split())
        
        # Limit the text to max_words
        words = main_content.split()
        if len(words) > max_words:
            main_content = " ".join(words[:max_words]) + "... (truncated)"
        
        return {
            "headline": headline,
            "content": main_content if main_content else "Failed to extract article content."
        }
    
    except Exception as e:
        return {
            "headline": "Error extracting headline",
            "content": f"Error extracting content: {str(e)}"
        }

def extract_article_text(url, timeout=10, max_words=1000):
    """Extract the main text content and headline from a news article URL with a word limit."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Check content type to ensure it's HTML
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            return {
                "headline": "Not an HTML page",
                "content": f"Not an HTML page. Content-Type: {content_type}"
            }
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check if this is likely a news article
        if not is_likely_news_article(soup):
            return {
                "headline": "Not a news article",
                "content": "Not identified as a news article."
            }
        
        # Extract the article text and headline
        return extract_article_text_from_soup(soup, max_words)
    
    except Exception as e:
        return {
            "headline": "Error",
            "content": f"Error extracting content: {str(e)}"
        }

def search_news(state: AgentState) -> AgentState:
    try:
        response = tavily_client.search(
            query=state["initial_search_query"],
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            max_results=7  # Increased from 3 to 7 to provide more candidates for filtering
        )
        state["search_results"] = response.get("results", [])
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
        url = result.get("url")
        if not url:
            continue
            
        try:
            # Fetch URL content
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Check content type to ensure it's HTML
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                print(f"Skipping {url}: Not an HTML page. Content-Type: {content_type}")
                continue
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Check if this is likely a news article
            if not is_likely_news_article(soup):
                print(f"Skipping {url}: Not identified as a news article.")
                continue
            
            # Extract the article text and headline
            extracted = extract_article_text_from_soup(soup, max_words=1000)
            
            # Add to processed articles
            processed.append({
                "url": url,
                "headline": extracted["headline"],
                "snippet": result.get("content", ""),
                "full_content": extracted["content"]
            })
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {str(e)}")
            continue
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            continue
    
    # Update state with processed articles
    state["processed_articles"] = processed
    
    # Check if any articles passed the filter
    if not processed:
        state["error"] = "No valid news articles found after filtering."
    
    state["current_step"] = "process_articles"
    return state

def summarize_news(state: AgentState) -> AgentState:
    # Check if we have articles to summarize
    if not state.get("processed_articles"):
        state["error"] = "No articles to summarize."
        return state
    
    # Process each article individually to avoid token limits
    for i, article in enumerate(state["processed_articles"]):
        try:
            article_text = article["full_content"]
            
            # Skip if the content indicates it's not a news article
            if article_text == "Not identified as a news article." or article_text.startswith("Not an HTML page"):
                continue
                
            prompt = f"Summarize the following news article in 2-3 sentences:\n{article_text}"
            article_summary = llm.invoke(prompt)
            
            # Parse the summary if it's an AIMessage
            if hasattr(article_summary, 'content'):
                article_summary = article_summary.content
                
            # Add the summary to the article
            state["processed_articles"][i]["summary"] = article_summary
        except Exception as e:
            state["processed_articles"][i]["summary"] = f"Failed to summarize: {str(e)}"
    
    state["current_step"] = "summarize_news"
    return state

def generate_quiz(state: AgentState) -> AgentState:
    # Check if we have articles to generate quiz from
    if not state.get("processed_articles"):
        state["error"] = "No articles to generate quiz from."
        return state
    
    # Collect all article summaries
    article_summaries = []
    for i, article in enumerate(state["processed_articles"]):
        # Use summary if available, otherwise use snippet or full_content
        summary = article.get("summary", article.get("snippet", article.get("full_content", "")))
        
        # Skip if the content indicates it's not a news article
        if summary == "Not identified as a news article." or summary.startswith("Not an HTML page"):
            continue
            
        article_summaries.append(f"Article {i+1}: {summary}")
    
    # Check if we have any valid summaries
    if not article_summaries:
        state["error"] = "No valid article summaries to generate quiz from."
        return state
    
    # Combine all summaries
    combined_summaries = "\n\n".join(article_summaries)
    
    # Generate quiz based on the combined summaries with clear format instructions
    prompt = f"""Generate 3 multiple-choice questions based on these news articles:

{combined_summaries}

Format each question as a JSON object with the following structure:
{{
  "question": "The question text",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct_option": "The correct option (must be exactly one of the options)"
}}

Return the questions as a JSON array of these objects.
"""
    try:
        quiz = llm.invoke(prompt)
        state["quiz"] = quiz
    except Exception as e:
        state["error"] = f"Error generating quiz: {str(e)}"
        # Create a simple fallback quiz
        state["quiz"] = [
            {
                "question": "What is the main topic of these news articles?",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_option": "Option A"
            }
        ]
    
    state["current_step"] = "generate_quiz"
    return state

# Define LangGraph workflow
graph = Graph()
graph.add_node("setup_initial_state", setup_initial_state)
graph.add_node("formulate_search_query", formulate_search_query)
graph.add_node("search_news", search_news)
graph.add_node("process_articles", process_articles)
graph.add_node("summarize_news", summarize_news)
graph.add_node("generate_quiz", generate_quiz)

graph.add_edge("setup_initial_state", "formulate_search_query")
graph.add_edge("formulate_search_query", "search_news")
graph.add_edge("search_news", "process_articles")
graph.add_edge("process_articles", "summarize_news")
graph.add_edge("summarize_news", "generate_quiz")

def run_agent(topic: str, num_days: int, region: str):
    state = setup_initial_state(topic, num_days, region)
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
    result = run_agent(topic, num_days, region)
    print("Summary:", result["summary"])
    print("Quiz:", result["quiz"])

