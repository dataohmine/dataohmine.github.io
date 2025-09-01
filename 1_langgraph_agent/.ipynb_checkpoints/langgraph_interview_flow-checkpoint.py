# langgraph_interview_flow.py
from langgraph.graph import StateGraph, END
from typing import Annotated, TypedDict

# Define the state schema as a TypedDict class
class InterviewState(TypedDict):
    role: Annotated[str, "Role of the person (e.g., operator, sponsor)"]
    stage: Annotated[str, "Current interview stage"]
    questions: Annotated[list, "List of questions for the current stage"]
    current_q_idx: Annotated[int, "Index of the current question"]
    transcript: Annotated[list, "List of question/answer pairs"]
    should_skip: Annotated[bool, "Flag to skip the current stage"]
    session: Annotated[dict, "Session metadata"]
    audio_segments: Annotated[list, "List of audio segments collected"]
    questions_complete: Annotated[bool, "Flag indicating if all questions are done"]
    last_audio_path: Annotated[str, "Path to the last recorded audio file"]

def create_interview_graph():
    """Create and compile the interview graph"""
    # Import nodes here to avoid circular imports
    from interview_nodes import (
        load_questions_node,
        ask_question_node,
        summary_node,
        save_session_node,
        route_after_question
    )
    
    # Create the graph with the TypedDict schema
    graph = StateGraph(InterviewState)
    
    # Add nodes to the graph
    graph.add_node("load_questions", load_questions_node)
    graph.add_node("ask_question", ask_question_node)
    graph.add_node("summarize", summary_node)
    graph.add_node("save", save_session_node)
    
    # Set up the flow
    graph.set_entry_point("load_questions")
    
    # Set edges - simplified flow since ask_question_node now handles everything
    graph.add_edge("load_questions", "ask_question")
    graph.add_conditional_edges("ask_question", route_after_question)
    graph.add_edge("summarize", "save")
    graph.add_edge("save", END)
    
    return graph.compile()

# --- Runner ---
def run_stage(role: str, stage: str, session: dict):
    """Run a single interview stage"""
    # Initialize operator name if not set
    if 'operator' not in session:
        session['operator'] = "Unknown_Operator"
    
    # Create the graph dynamically
    interview_graph = create_interview_graph()
    
    # Initialize state for the current interview stage
    state = {
        'role': role,
        'stage': stage,
        'questions': [],
        'current_q_idx': 0,
        'transcript': [],
        'should_skip': False,
        'session': session,
        'audio_segments': [],
        'questions_complete': False,  # Track if questions are complete
        'last_audio_path': None  # Initialize audio path as None
    }
    
    # Run the interview graph with the current state
    final_state = interview_graph.invoke(state)
    
    # Return the updated session data after completing the stage
    return final_state['session']