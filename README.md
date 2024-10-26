# MARTy (Model Agnostic Research Tool for Hydrology)

MARTy is an AI-powered assistant designed to help researchers and practitioners with hydrological modeling tasks. It provides voice-interactive guidance for setting up and running CONFLUENCE models, analyzing results, and explaining hydrological concepts.

## Features

- Voice-interactive interface for model setup and analysis
- Integration with CONFLUENCE hydrological modeling framework
- Support for multiple model configurations and analysis methods
- Real-time voice feedback and explanations
- Expertise in hydrological concepts and modeling decisions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DarriEy/MARTy.git
cd MARTy


2. Create and activate a virtual environment:
´´´
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
´´´

3. Install required packages:
´´´
pip install -r requirements.txt
´´´

4. Set up environment variables in a .env file:
´´´
ANTHROPIC_API_KEY=your_api_key
GOOGLE_CREDENTIALS_PATH=path_to_credentials.json
´´´

## Usage
Run MARTy:

´´´
python MARTy.py
´´´
MARTy will start in voice-interactive mode. You can:

Press Enter to use voice input
Type your message directly
Type 'exit' to end the session

## Key Commands

"new project" or "start project": Initialize a new CONFLUENCE project
"run model" or "execute model": Run an existing CONFLUENCE configuration
"analyze results": Examine model outputs
"explain [concept]": Get detailed explanations of hydrological concepts

## Requirements

Python 3.8+
Anthropic API key for Claude
Google Cloud credentials for text-to-speech (optional)
Working microphone for voice input

## License
MIT License - see LICENSE file for details.
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
