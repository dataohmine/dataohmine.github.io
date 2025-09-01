# AI Interview Assistant

A professional voice-driven interview system built with LangGraph and OpenAI's API. This system conducts structured interviews with real-time speech-to-text transcription, intelligent conversation flow management, and comprehensive session recording.

## Overview

The AI Interview Assistant leverages LangGraph's state management capabilities to create a sophisticated interview experience. It handles voice interactions with intelligent transcription error correction, and provides comprehensive session management with automatic checkpointing for professional candidate assessment.

## Key Features

- **LangGraph State Management**: Robust interview flow control with automatic state transitions
- **Voice Processing**: Real-time speech-to-text with OpenAI Whisper integration
- **Intelligent Transcription**: Advanced business acronym detection and correction (CEO, CFO, COO variants)
- **Professional Interviews**: Comprehensive candidate assessment workflow
- **Session Persistence**: Automatic checkpointing and resume functionality
- **Audio Recording**: Complete interview audio capture with MP3 export
- **Response Validation**: Smart validation with retry mechanisms for unclear responses

## System Architecture

### Core Components

- **LangGraph Workflow**: State-driven interview progression using directed graph execution
- **Interview Nodes**: Modular processing units for questions, audio, and validation
- **Audio Handler**: Voice recording and playback with silence detection
- **Configuration System**: JSON-based settings for models, prompts, and validation rules

### Interview Flow

```
Load Questions → Ask Question → Record Audio → Transcribe → Validate → Store Response
      ↓              ↑                                          ↓
   Summary ← ────────┴──────────────── (Loop until complete) ──┘
      ↓
  Save Session
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dataohmine/ai_interviewer.git
   cd ai_interviewer
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment setup**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

## Configuration

### Environment Variables

Create a `.env` file with:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Interview Configuration

The `config.json` file controls:
- **Voice Models**: TTS and transcription model selection
- **Transcription Prompts**: Context-specific prompts for different question types
- **Validation Settings**: Response validation rules and retry logic
- **Audio Settings**: Recording parameters and quality settings
- **Business Acronym Detection**: Enhanced CEO/CFO/COO transcription accuracy

## Usage

### Basic Interview Execution

Run a complete interview cycle:
```bash
python main.py
```

### Session Management

The system automatically creates checkpoints and supports resume functionality:
```bash
# Resume from interrupted interview
python main.py  # Automatically detects existing checkpoint
```

## Question Management

Interview questions are stored in `questions/interview.json`:

```json
{
  "id": "question_id",
  "text": "Question content for text-to-speech"
}
```

### Question Types

- **Role Questions**: Current position identification with enhanced transcription
- **Experience Questions**: Professional background and expertise  
- **Achievement Questions**: Career accomplishments and challenges
- **Goal Questions**: Future aspirations and career objectives

## Output Structure

Interview sessions generate comprehensive outputs:

```
interview_outputs/
└── FirstName_LastName_TIMESTAMP/
    ├── FirstName_LastName_TIMESTAMP_interview.json
    └── FirstName_LastName_TIMESTAMP_audio.mp3
```

### JSON Output Format

```json
{
  "participant_info": {
    "first_name": "John",
    "last_name": "Doe", 
    "timestamp": "20240101_143022",
    "stage": "interview",
    "role": "candidate"
  },
  "transcript": [
    {
      "id": "question_id",
      "q": "Question text",
      "a": "Participant response"
    }
  ],
  "session_data": {
    "session_id": "session_20240101_143022",
    "start": "2024-01-01T14:30:22",
    "interview": {
      "transcript": [...],
      "summary": "AI-generated summary"
    }
  }
}
```

## Advanced Features

### Intelligent Transcription Correction

The system includes sophisticated business acronym detection:

- **CEO Variants**: Handles common misheard variations (io, co, seo, see-oh, etc.)
- **CFO Detection**: Recognizes seafo, seaphone, c-f-o patterns
- **COO Recognition**: Processes ku, koo, c-o-o variations
- **Context-Aware**: Uses question-specific transcription prompts

### Response Validation

Multi-level validation system:

1. **Audio Quality**: Checks recording duration and file size
2. **Content Validation**: Detects empty responses and common misheard phrases  
3. **Role-Specific Logic**: Enhanced validation for executive titles
4. **Retry Mechanism**: Intelligent retry with different clarification prompts

### Error Handling

Comprehensive error management:

- **Graceful Degradation**: Continues interview despite individual question failures
- **Session Persistence**: Saves progress on errors for later resume
- **Detailed Logging**: Comprehensive error reporting and debugging information
- **Recursion Protection**: Built-in safeguards against infinite loops

## Technical Specifications

### Dependencies

- **langgraph**: State machine and workflow management
- **openai**: GPT models and Whisper transcription
- **pyaudio**: Real-time audio recording
- **pydub**: Audio processing and manipulation
- **webrtcvad**: Voice activity detection
- **python-dotenv**: Environment variable management

### Performance Considerations

- **Memory Management**: Efficient audio segment handling
- **API Optimization**: Batched requests and error retry logic  
- **File Cleanup**: Automatic temporary file management
- **Checkpoint Strategy**: Minimal state persistence for resume capability

## Development

### Project Structure

```
ai_interviewer/
├── main.py                     # Entry point and session management
├── langgraph_interview_flow.py # LangGraph workflow definition
├── interview_nodes.py          # Core interview processing logic
├── config.json                 # System configuration
├── requirements.txt            # Python dependencies
├── questions/                  # Interview question sets
│   └── interview.json
├── utils/                      # Utility modules
│   ├── audio_utils.py         # Audio recording and playback
│   └── text_utils.py          # Text processing helpers
└── tests/                      # Test suite
    ├── test_interview.py
    └── test_text_utils.py
```

### Testing

Run the test suite:
```bash
python -m pytest tests/
```

### Customization

1. **Add New Questions**: Modify `questions/interview.json` file
2. **Modify Transcription Logic**: Update prompts in `config.json`
3. **Extend Validation Rules**: Enhance validation functions in `interview_nodes.py`
4. **Custom Audio Processing**: Modify `utils/audio_utils.py`

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

For issues and feature requests, please use the GitHub issue tracker.