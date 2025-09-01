#!/usr/bin/env python3
"""
Enhanced AI Interview Assistant - Streamlit Application
Built with LangGraph and OpenAI for professional interview preparation
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Set environment variables for better Streamlit deployment
os.environ.setdefault('STREAMLIT_SERVER_PORT', '8501')
os.environ.setdefault('STREAMLIT_SERVER_ADDRESS', '0.0.0.0')

try:
    from frontend import main
    
    if __name__ == "__main__":
        # Run the main application
        main()
        
except ImportError as e:
    import streamlit as st
    st.error(f"""
    ❌ **Import Error**: {str(e)}
    
    **Please ensure all dependencies are installed:**
    ```bash
    pip install -r requirements.txt
    ```
    
    **Required files:**
    - config.json
    - questions/interview.json
    - utils/audio_utils.py
    - utils/text_utils.py
    """)
    
except Exception as e:
    import streamlit as st
    st.error(f"""
    ❌ **Application Error**: {str(e)}
    
    **Troubleshooting:**
    1. Check that all required files exist
    2. Verify OpenAI API key is set: `OPENAI_API_KEY`
    3. Ensure audio permissions are enabled
    4. Check the application logs for more details
    """)
    
    # Show debug information in development
    if os.getenv('STREAMLIT_ENV') == 'development':
        st.exception(e)