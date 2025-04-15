from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
#from langchain_openai import OpenAI
from openai import OpenAI
import operator
import os
class state(TypedDict):
    problem_to_solve: str
    user_input: str
    technicals: str
    mvp_features: str
    ui_design: str
    ux_design: str
    final_draft: str

#llm = ChatGroq(model = 'meta-llama/llama-4-scout-17b-16e-instruct')
#llm = OpenAI(model_name = "gpt-4o-mini")

os.environ["OPENAI_API_KEY"] = "YOUR_OPEN_AI_API_KEY"

from openai import OpenAI
def openai_completion(input):
    client = OpenAI()
    response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
        "role": "user",
        "content": [
            {
            "type": "input_text",
            "text": input
            }
        ]
        },
    ],
    text={
        "format": {
        "type": "text"
        }
    },
    reasoning={},
    tools=[],
    temperature=1,
    max_output_tokens=2048,
    top_p=1,
    store=True
    )
    return response.output[0].content[0].text

def extract_technicals(state):
    # some code for node_1
    prompt = f"""You are Kevin Weil, CPO of Instagram, adept at prototyping. Select a tech stack for a quick MVP based on the app idea and features.
                    Prototype requirements:
                    - Demonstrate core functionality
                    - Support up to 50 users
                    - Use free resources
                    - Prioritize fast development over robustness
                    Output as a bullet list (e.g., - Gradio (frontend), - Python FastAPI (backend)).
                    App idea: {state['user_input']}
                    MVP features: {state['mvp_features']}"""
    message = openai_completion(prompt)
    state['technicals'] = message
    return state

def mvp_features(state):
    prompt = f"""You are Kevin Weil, CPO of Instagram, skilled at turning ideas into quick prototypes. Given an app idea, identify the 3 most important features for an MVP.
                Prototype requirements:
                - Demonstrate core functionality
                - Support up to 50 users
                - Use free resources
                - Prioritize fast development over robustness
                Be concise, innovative, and output as a numbered list (1., 2., 3.). 
                App idea: {state['user_input']}
                Core problem (if provided): {state['problem_to_solve']}"""
    message = openai_completion(prompt)
    state['mvp_features'] = message
    return state

def UX_designer(state):
    prompt = f"""You are an expert UX designer skilled in creating user-friendly flows. Given an app idea and MVP features, design the user journey and flow.
                    Describe each key screen, its purpose, and how users navigate between them. Output as a concise numbered list (e.g., 1. Screen Name - Purpose - Navigation).
                    App idea: {state['user_input']}
                    MVP features: {state['mvp_features']}"""
    message = openai_completion(prompt)
    state['ux_design'] = message
    return state

def UI_designer(state):
    prompt = f"""You are an expert UI designer skilled in crafting beautiful mockups. Given an app idea and UX design, create the UI layout and visual design for each screen.
                Include screen layout and design elements (e.g., color scheme, typography). Output as a numbered list (e.g., 1. Screen Name - Layout - Design Elements).
                App idea: {state['user_input']}
                UX design: {state['ux_design']}"""
    message = openai_completion(prompt)
    state['ui_design'] = message
    return state

def final_compiler(state):
    prompt = f"""You are an expert prompt engineer. Given a complete MVP plan, create a concise, step-by-step instructional prompt for an LLM to generate the app’s code.
                    Include setup, feature implementation, and design integration. Output as a numbered list (e.g., 1. Step - Description).
                    MVP features: {state['mvp_features']}
                    Tech stack: {state['technicals']}
                    UX design: {state['ux_design']}
                    UI design: {state['ui_design']}
            """
    message = openai_completion(prompt)
    state['final_draft'] = message
    return state 

graph = StateGraph(state)

graph.add_node("mvp",mvp_features)
graph.add_node("cto",extract_technicals)
graph.add_node("ux",UX_designer)
graph.add_node("ui",UI_designer)
graph.add_node("final_compiler", final_compiler)

graph.add_edge(START, "mvp")
graph.add_edge("mvp", "cto")
graph.add_edge("cto", "ux")
graph.add_edge("ux", "ui")
graph.add_edge("ui", "final_compiler")
graph.add_edge("final_compiler", END)
final_graph = graph.compile()


def final_graph(initial_state):
    final_state = final_graph.invoke(initial_state)
    return final_state
