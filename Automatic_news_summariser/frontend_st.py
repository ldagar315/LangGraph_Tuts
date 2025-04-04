import streamlit as st
import os
from news_summariser_openai import run_agent
import json

# Set page configuration
st.set_page_config(
    page_title="News Summarizer & Quiz",
    page_icon="📰",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .news-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .quiz-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .correct-answer {
        color: #4CAF50;
        font-weight: bold;
    }
    .incorrect-answer {
        color: #F44336;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>News Summarizer & Quiz Generator</h1>", unsafe_allow_html=True)

# Helper function to parse quiz data
def parse_quiz_data(quiz_data):
    """Parse the quiz data from AIMessage to a list of dictionaries."""
    if not quiz_data:
        return []
    
    # If it's already a list, return it
    if isinstance(quiz_data, list):
        return quiz_data
    
    # If it's a string, try to parse it as JSON
    if isinstance(quiz_data, str):
        try:
            # Try to find JSON array in the string
            import re
            json_match = re.search(r'\[(.*)\]', quiz_data, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # Try to parse the entire string as JSON
                return json.loads(quiz_data)
        except json.JSONDecodeError:
            # If it's not valid JSON, try to extract quiz questions using regex
            import re
            questions = []
            
            # Look for JSON-like objects
            json_objects = re.findall(r'\{[^{}]*\}', quiz_data)
            for obj_str in json_objects:
                try:
                    # Try to parse each object
                    obj = json.loads(obj_str)
                    if 'question' in obj and 'options' in obj and 'correct_option' in obj:
                        questions.append(obj)
                except:
                    pass
            
            if questions:
                return questions
            
            # Fallback to regex pattern matching
            question_pattern = r'Question \d+: (.*?)\n'
            options_pattern = r'Options: \[(.*?)\]'
            correct_pattern = r'Correct Answer: (.*?)(?:\n|$)'
            
            question_matches = re.findall(question_pattern, quiz_data)
            options_matches = re.findall(options_pattern, quiz_data)
            correct_matches = re.findall(correct_pattern, quiz_data)
            
            for i in range(min(len(question_matches), len(options_matches), len(correct_matches))):
                question = question_matches[i]
                options_str = options_matches[i]
                correct = correct_matches[i]
                
                # Parse options
                options = [opt.strip().strip("'\"") for opt in options_str.split(',')]
                
                questions.append({
                    "question": question,
                    "options": options,
                    "correct_option": correct
                })
            
            return questions
    
    # If it's an AIMessage object, try to extract content
    if hasattr(quiz_data, 'content'):
        return parse_quiz_data(quiz_data.content)
    
    # Default case - create a simple quiz if nothing else works
    return [
        {
            "question": "What is the main topic of these news articles?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_option": "Option A"
        }
    ]

# Helper function to parse summary
def parse_summary(summary_data):
    """Parse the summary data from AIMessage to a string."""
    if not summary_data:
        return "No summary available."
    
    # If it's already a string, return it
    if isinstance(summary_data, str):
        return summary_data
    
    # If it's an AIMessage object, try to extract content
    if hasattr(summary_data, 'content'):
        return summary_data.content
    
    # Default case
    return str(summary_data)

# Sidebar for input parameters
with st.sidebar:
    st.markdown("<h2 class='sub-header'>Search Parameters</h2>", unsafe_allow_html=True)
    
    topic = st.text_input("Topic", "AI Regulation")
    num_days = st.slider("Number of Days", 1, 30, 3)
    region = st.selectbox("Region", ["Global", "US", "Europe", "Asia", "Africa", "South America", "Australia"])
    
    if st.button("Generate News & Quiz"):
        with st.spinner("Generating news summaries and quiz questions..."):
            # Run the agent
            result = run_agent(topic, num_days, region)
            
            # Parse quiz and summary data
            if 'quiz' in result:
                result['quiz'] = parse_quiz_data(result['quiz'])
            
            if 'summary' in result:
                result['summary'] = parse_summary(result['summary'])
            
            # Store results in session state
            st.session_state.result = result
            st.session_state.quiz_submitted = False
            st.session_state.selected_answers = {}

# Main content area with tabs
if 'result' in st.session_state:
    tab1, tab2 = st.tabs(["News Articles", "Quiz Questions"])
    
    # News Tab
    with tab1:
        st.markdown("<h2 class='sub-header'>News Articles</h2>", unsafe_allow_html=True)
        
        if st.session_state.result.get("error"):
            st.error(f"Error: {st.session_state.result['error']}")
        
        if st.session_state.result.get("processed_articles"):
            for i, article in enumerate(st.session_state.result["processed_articles"]):
                with st.container():
                    st.markdown(f"<div class='news-card'>", unsafe_allow_html=True)
                    
                    # Display article headline as heading
                    headline = article.get("headline", "News Article")
                    st.markdown(f"<h3>{headline}</h3>", unsafe_allow_html=True)
                    
                    # Display article URL as a small link
                    url = article.get("url", "")
                    if url:
                        st.markdown(f"<p><small><a href='{url}' target='_blank'>Source: {url}</a></small></p>", unsafe_allow_html=True)
                    
                    # Display article summary
                    if 'summary' in article:
                        st.markdown(f"<p><strong>Summary:</strong> {article['summary']}</p>", unsafe_allow_html=True)
                    else:
                        # If no individual summary, use the snippet
                        st.markdown(f"<p><strong>Summary:</strong> {article['snippet']}</p>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("No news articles found. Please try different search parameters.")
    
    # Quiz Tab
    with tab2:
        st.markdown("<h2 class='sub-header'>Quiz Questions</h2>", unsafe_allow_html=True)
        
        quiz = st.session_state.result.get("quiz", [])
        
        if quiz:
            # Initialize selected answers if not already done
            if not st.session_state.selected_answers:
                st.session_state.selected_answers = {i: None for i in range(len(quiz))}
            
            # Display each quiz question
            for i, question in enumerate(quiz):
                with st.container():
                    st.markdown(f"<div class='quiz-card'>", unsafe_allow_html=True)
                    st.markdown(f"<h3>Question {i+1}: {question['question']}</h3>", unsafe_allow_html=True)
                    
                    # Create radio buttons for options
                    options = question['options']
                    selected_option = st.radio(
                        f"Select your answer:",
                        options,
                        key=f"q_{i}",
                        index=options.index(st.session_state.selected_answers[i]) if st.session_state.selected_answers[i] in options else None
                    )
                    
                    # Update selected answer
                    st.session_state.selected_answers[i] = selected_option
                    
                    # Show correct answer if quiz is submitted
                    if st.session_state.quiz_submitted:
                        correct_option = question['correct_option']
                        if selected_option == correct_option:
                            st.markdown(f"<p class='correct-answer'>✓ Correct!</p>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<p class='incorrect-answer'>✗ Incorrect. The correct answer is: {correct_option}</p>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Submit button
            if not st.session_state.quiz_submitted:
                if st.button("Submit Quiz"):
                    st.session_state.quiz_submitted = True
                    st.rerun()
            
            # Reset button
            if st.session_state.quiz_submitted:
                if st.button("Reset Quiz"):
                    st.session_state.quiz_submitted = False
                    st.session_state.selected_answers = {i: None for i in range(len(quiz))}
                    st.rerun()
        else:
            st.warning("No quiz questions available. Please generate news and quiz first.")
else:
    st.info("Please enter search parameters and click 'Generate News & Quiz' to get started.") 
