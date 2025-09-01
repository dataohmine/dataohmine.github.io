import os
import json
import argparse
from langgraph_interview_flow import run_stage
from datetime import datetime
from pathlib import Path

ALL_STAGES = [
    ("candidate", "interview")
]

def create_new_session():
    return {
        "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "candidate": None,
        "start": datetime.now().isoformat(),
        "interview": {}
    }

def load_existing_session(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def save_checkpoint(session, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(session, f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description="Run AI interview system.")
    parser.add_argument("--recursion-limit", type=int, default=50, help="Set recursion limit for LangGraph")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    checkpoint_dir = Path("interview_outputs")
    checkpoint_path = checkpoint_dir / f"checkpoint_session.json"
    
    if checkpoint_path.exists():
        print("Resuming from existing checkpoint...")
        session = load_existing_session(checkpoint_path)
    else:
        session = create_new_session()
    
    try:
        for role, stage in ALL_STAGES:
            if session.get(stage):
                print(f"Skipping completed stage: {stage}")
                continue

            print(f"\nStarting interview session for {role}")

            try:
                session = run_stage(role=role, stage=stage, session=session)
                save_checkpoint(session, checkpoint_path)
                print(f"Interview completed successfully")
                
            except Exception as stage_error:
                print(f"Error during interview: {stage_error}")
                print("Progress saved. You can resume later.")
                raise  # Re-raise to exit the loop
        
        print("\nInterview session completed successfully.")
        
        # Clean up checkpoint file on successful completion
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print("Checkpoint file cleaned up.")
            
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Progress saved. You can resume later.")
        print(f"Checkpoint saved at: {checkpoint_path}")
        
        # Provide debugging info
        if "recursion limit" in str(e).lower():
            print("\nRecursion limit reached. This usually means:")
            print("   • Your workflow has no END state transitions")
            print("   • Nodes are creating infinite loops")
            print("   • Missing stop conditions in your graph")
            print(f"   • Try increasing --recursion-limit (current: {args.recursion_limit})")
