# MARTy.py

from typing import TYPE_CHECKING, Optional
from anthropic import Anthropic # type: ignore
from google.cloud import texttospeech # type: ignore
import pygame # type: ignore
import sounddevice as sd # type: ignore
import soundfile as sf # type: ignore
import numpy as np
import speech_recognition as sr # type: ignore
import time
import wave
import pygame.mixer # type: ignore
from tempfile import NamedTemporaryFile
import yaml
from pathlib import Path
import os
import sys
from gtts import gTTS # type: ignore


if TYPE_CHECKING:
    from INDRA import INDRA # type: ignore

class MARTyProjectManager:
    async def initiate_project(self):
        """Guide user through project initialization via voice interaction"""
        try:
            # Get watershed name
            self.marty.speak_response("What is the name of the watershed you want to model?")
            watershed_name = await self._get_validated_input("watershed name")

            # Initialize project through INDRA
            self.marty.speak_response(
                "Thank you. I'll now use INDRA to suggest an optimal configuration for "
                f"modeling the {watershed_name} watershed. This may take a moment..."
            )
            
            # Get initial config and justification from INDRA
            config, justification = self.indra.chairperson.expert_initiation(watershed_name)
            
            # Explain the suggested configuration
            explanation = (
                f"INDRA has suggested a configuration for the {watershed_name} watershed. "
                f"\n\nHere's a summary of the key settings:\n{self._summarize_config(config)}"
                f"\n\nThe reasoning behind these choices is:\n{justification}"
            )
            self.marty.speak_response(explanation)

            # Handle configuration modifications
            config = await self._handle_config_modifications(config)
            
            if config:
                # Save the configuration
                config_path = await self._save_configuration(config)
                self.current_config_path = config_path
                
                # Ask about running CONFLUENCE
                self.marty.speak_response(
                    "Configuration has been saved. Would you like to proceed with running "
                    "CONFLUENCE using this configuration?"
                )
                
                if await self._get_confirmation():
                    await self.run_confluence(config_path)
                
                return config_path
            
            return None

        except Exception as e:
            error_msg = f"An error occurred during project initialization: {str(e)}"
            self.marty.speak_response(error_msg)
            return None

    async def _handle_config_modifications(self, config: dict) -> Optional[dict]:
        """Handle user modifications to the suggested configuration"""
        self.marty.speak_response(
            "Would you like to modify any of these settings? I can help you understand "
            "and adjust each parameter."
        )

        if await self._get_confirmation():
            modified_config = config.copy()
            
            while True:
                # Ask which parameter to modify
                self.marty.speak_response(
                    "Which parameter would you like to modify? You can say the parameter "
                    "name, or say 'list parameters' to hear them again, or 'done' to finish."
                )
                
                response = await self._get_validated_input("parameter choice")
                
                if response.lower() == 'done':
                    break
                    
                if response.lower() == 'list parameters':
                    await self._list_parameters(modified_config)
                    continue
                
                # Find matching parameter
                param = self._find_matching_parameter(response, modified_config)
                
                if param:
                    # Explain current value and get new value
                    await self._modify_parameter(modified_config, param)
                else:
                    self.marty.speak_response(
                        "I couldn't find that parameter. Please try again or say 'list parameters' "
                        "to hear the available options."
                    )

            # Confirm final configuration
            self.marty.speak_response(
                "Here's the final configuration after your modifications. "
                f"\n{self._summarize_config(modified_config)}\n\nWould you like to proceed with this configuration?"
            )
            
            if await self._get_confirmation():
                return modified_config
            
            return None
        
        return config

    def _summarize_config(self, config: dict) -> str:
        """Create a user-friendly summary of the configuration"""
        key_params = [
            'HYDROLOGICAL_MODEL',
            'DOMAIN_DISCRETIZATION',
            'FORCING_DATASET',
            'ELEVATION_BAND_SIZE',
            'MIN_HRU_SIZE'
        ]
        
        summary = []
        for param in key_params:
            if param in config:
                summary.append(f"{param}: {config[param]}")
        
        return "\n".join(summary)

    async def _list_parameters(self, config: dict):
        """List available parameters with their current values"""
        self.marty.speak_response("Here are the current parameters and their values:")
        
        for param, value in config.items():
            self.marty.speak_response(f"{param}: {value}")
            
        # Brief pause between listings and next prompt
        await asyncio.sleep(1)

    def _find_matching_parameter(self, user_input: str, config: dict) -> Optional[str]:
        """Find the closest matching parameter name"""
        user_input = user_input.lower()
        
        # Direct match
        for param in config:
            if user_input == param.lower():
                return param
        
        # Partial match
        for param in config:
            if user_input in param.lower():
                return param
        
        return None

    async def _modify_parameter(self, config: dict, param: str):
        """Guide user through modifying a specific parameter"""
        current_value = config[param]
        
        # Get parameter explanation from INDRA
        explanation = self.indra.chairperson.get_parameter_explanation(param)
        
        self.marty.speak_response(
            f"The parameter {param} currently has value: {current_value}\n"
            f"{explanation}\n"
            "What would you like to change it to?"
        )
        
        new_value = await self._get_validated_input("new value")
        
        # Validate the new value
        validation_result = self.indra.chairperson.validate_parameter_value(param, new_value)
        
        if validation_result['valid']:
            config[param] = self._convert_value_type(new_value, type(current_value))
            self.marty.speak_response(f"Updated {param} to: {new_value}")
        else:
            self.marty.speak_response(
                f"That value appears to be invalid: {validation_result['reason']}\n"
                "Please try again."
            )
            await self._modify_parameter(config, param)

    def _convert_value_type(self, value: str, target_type: type):
        """Convert string value to appropriate type"""
        try:
            if target_type == bool:
                return value.lower() in ['true', 'yes', '1', 'on']
            return target_type(value)
        except ValueError:
            return value

class MARTy:
    def __init__(self, anthropic_api_key, google_credentials_path=None):
        """Initialize MARTy with required APIs and configurations"""
        # Initialize APIs
        self.anthropic = Anthropic(api_key=anthropic_api_key)
        
        # Initialize Google Cloud TTS
        if google_credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials_path
        self.tts_client = texttospeech.TextToSpeechClient()
        
        # Configure voice settings
        self.voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-GB-Standard-B",  # Deep, authoritative male voice
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )
        
        # Configure audio settings
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.9,  # Slightly slower for clarity
            pitch=0,  # Natural pitch
            volume_gain_db=0.0  # Normal volume
        )
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        
        # Audio recording settings
        self.sample_rate = 44100
        self.channels = 1
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        # Initialize project manager
        self.project_manager = None
        
       # Define enhanced system message
        self.system_message = """
        You are MARTy (Model Agnostic Research Tool for Hydrology), an advanced AI assistant 
        specializing in hydrological modeling and research. You can:

        1. Help users set up and run CONFLUENCE hydrological models
        2. Initialize new modeling projects through INDRA
        3. Explain hydrological concepts and modeling decisions
        4. Guide users through the modeling workflow
        5. Analyze and explain model results

        When discussing hydrological modeling:
        1. Use clear, accessible language while maintaining technical accuracy
        2. Ask clarifying questions when needed
        3. Provide context for technical decisions
        4. Explain the implications of different modeling choices

        Key commands you understand:
        - "new project" or "start project": Initialize a new CONFLUENCE project
        - "run model" or "execute model": Run an existing CONFLUENCE configuration
        - "analyze results": Examine model outputs
        - "explain [concept]": Provide detailed explanations of hydrological concepts

        Always maintain a helpful and educational tone while ensuring technical accuracy.
        """
        # Initialize INDRA if available
        try:
            sys.path.append(str(Path(__file__).resolve().parent))
            from INDRA import INDRA # type: ignore
            self.indra = INDRA(anthropic_api_key)
            self.has_indra = True
        except ImportError:
            self.has_indra = False
            print("Note: INDRA framework not available. Some modeling features will be limited.")

    def speak_response(self, text):
        """Convert text to speech using Google Cloud TTS"""
        try:
            # Create temporary file
            with NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_filename = fp.name
            
            # Synthesize speech
            synthesis_input = texttospeech.SynthesisInput(text=text)
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=self.voice,
                audio_config=self.audio_config
            )
            
            # Save the audio content
            with open(temp_filename, "wb") as out:
                out.write(response.audio_content)
            
            # Play the audio
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()
            
            # Wait for the audio to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Clean up
            pygame.mixer.music.unload()
            os.unlink(temp_filename)
            
        except Exception as e:
            print(f"Error in speech synthesis: {e}")
            # Fallback to basic gTTS
            try:
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(temp_filename)
                
                pygame.mixer.music.load(temp_filename)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                    
                pygame.mixer.music.unload()
                os.unlink(temp_filename)
                
            except Exception as e2:
                print(f"Error with fallback voice: {e2}")
                print(f"Response (text only): {text}")

    def change_voice(self, voice_name=None, speaking_rate=0.9, pitch=0):
        """Change voice parameters"""
        try:
            if voice_name:
                self.voice = texttospeech.VoiceSelectionParams(
                    language_code="en-US",
                    name=voice_name,
                    ssml_gender=texttospeech.SsmlVoiceGender.MALE
                )
            
            self.audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=speaking_rate,
                pitch=pitch,
                volume_gain_db=0.0
            )
            
            print(f"Voice updated: {voice_name if voice_name else self.voice.name}")
            return True
        except Exception as e:
            print(f"Error changing voice: {e}")
            return False

    def list_available_voices(self):
        """List all available voices"""
        try:
            voices = self.tts_client.list_voices().voices
            print("\nAvailable voices:")
            for voice in voices:
                if "en-US" in voice.language_codes:
                    print(f"Name: {voice.name}")
                    print(f"Gender: {voice.ssml_gender}")
                    print(f"Natural Sample Rate: {voice.natural_sample_rate_hertz}")
                    print("---")
            return voices
        except Exception as e:
            print(f"Error listing voices: {e}")
            return None

    def record_audio(self):
        """Record audio for a specified duration and convert to text"""
        print("\nPress Enter to start recording (recording duration: 5 seconds) or type your message...")
        user_input = input()
        
        # If user typed something, return it directly
        if user_input.strip():
            return user_input
            
        print("Recording... Speak clearly into your microphone.")
        duration = 5  # seconds
        recording = sd.rec(
            int(self.sample_rate * duration),
            samplerate=self.sample_rate,
            channels=self.channels
        )
        
        # Wait for the recording to complete
        sd.wait()
        
        print("Recording complete! Processing speech...")
        
        # Save as WAV file
        temp_file = "temp_input.wav"
        
        # Need to normalize the audio data to 16-bit integers
        recording = np.int16(recording * 32767)
        
        # Save as WAV with proper parameters
        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 2 bytes for 16-bit audio
            wf.setframerate(self.sample_rate)
            wf.writeframes(recording.tobytes())
        
        # Convert speech to text
        try:
            with sr.AudioFile(temp_file) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source)
                # Record the audio file
                audio = self.recognizer.record(source)
                # Use Google Speech Recognition
                text = self.recognizer.recognize_google(audio)
                print(f"\nRecognized text: {text}")
                
                # Clean up
                os.remove(temp_file)
                
                return text
                
        except sr.UnknownValueError:
            print("Could not understand the audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from speech recognition service; {e}")
            return None
        finally:
            # Ensure temp file is removed
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def get_claude_response(self, user_input):
        """Get response from Claude"""
        message = self.anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4096,
            system=self.system_message,
            messages=[{
                "role": "user",
                "content": user_input
            }]
        )
        # Extract just the text content from the response
        if hasattr(message.content, 'text'):
            return message.content.text
        elif isinstance(message.content, list):
            # Combine all text blocks
            return ' '.join([block.text for block in message.content if hasattr(block, 'text')])
        else:
            return str(message.content)

    def chunk_text(self, text, max_chunk_size=500):
        """Split text into chunks at sentence boundaries for better speech synthesis"""
        sentences = text.replace('.', '.|').replace('?', '?|').replace('!', '!|').split('|')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_words = len(sentence.split())
            
            if current_length + sentence_words > max_chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_words
            else:
                current_chunk.append(sentence)
                current_length += sentence_words
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    async def chat_loop(self, anthropic_api_key):
        """Main chat loop with voice interaction and project management"""
        try:
            # Import INDRA here to avoid circular imports
            from INDRA import INDRA # type: ignore
            # Initialize project manager if needed
            if self.project_manager is None:
                self.project_manager = MARTyProjectManager()

            # Start with voice greeting
            greeting = (
                "Hello! I'm MARTy, your Model Agnostic Research Tool for Hydrology. "
                "I can help you set up and run CONFLUENCE models, analyze results, "
                "and explain hydrological concepts. You can interact with me through "
                "voice by pressing Enter, or by typing your messages directly. "
                "Would you like to start a new project or discuss an existing one?"
            )
            
            print("\n" + "="*50)
            print("Welcome to MARTy - Hydrological Research Assistant")
            print("="*50)
            print("\nInitializing voice greeting...")
            
            self.speak_response(greeting)
            print("\nReady for interaction!")
            print("- Press Enter to use voice input")
            print("- Or type your message directly")
            print("- Type 'exit' or press Ctrl+C to end session")
            
            while True:
                try:
                    # Get user input (voice or text)
                    print("\nListening...")
                    user_input = self.record_audio()
                    
                    if user_input is None:
                        self.speak_response("I didn't catch that. Could you please repeat?")
                        continue
                    
                    # Check for exit command
                    if user_input.lower() in ['exit', 'quit', 'goodbye']:
                        self.speak_response("Goodbye! Feel free to return for more hydrological discussions!")
                        break
                    
                    # Process project-related commands
                    if any(phrase in user_input.lower() for phrase in 
                          ['new project', 'start project', 'initialize project', 'create project']):
                        config_path = await self.project_manager.initiate_project()
                        if config_path:
                            self.speak_response(
                                "Project initialization complete. Would you like to review the "
                                "configuration or proceed with running the model?"
                            )
                        continue
                    
                    elif any(phrase in user_input.lower() for phrase in 
                           ['run model', 'execute model', 'start model']):
                        if hasattr(self.project_manager, 'current_config_path'):
                            await self.project_manager.run_confluence(self.project_manager.current_config_path)
                        else:
                            self.speak_response(
                                "I don't have an active project configuration. "
                                "Would you like to start a new project?"
                            )
                        continue
                    
                    elif 'analyze results' in user_input.lower():
                        if hasattr(self.project_manager, 'current_results'):
                            analysis = self.project_manager.indra.analyze_confluence_results(
                                self.project_manager.current_results
                            )
                            explanation = await self.project_manager._generate_results_explanation(analysis)
                            self.speak_response(explanation)
                        else:
                            self.speak_response(
                                "I don't have any model results to analyze. "
                                "Would you like to run a model first?"
                            )
                        continue
                    
                    # Get Claude's response for other queries
                    print("\nThinking...")
                    response = self.get_claude_response(user_input)
                    
                    # Speak the response
                    print(f"\nMARTy: {response}")
                    
                    # Split response into chunks and speak each chunk
                    chunks = self.chunk_text(response)
                    for chunk in chunks:
                        self.speak_response(chunk)

                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}. Would you like to try again?"
                    print(f"\nError: {str(e)}")
                    self.speak_response(error_msg)
                    
        except KeyboardInterrupt:
            final_message = "Goodbye! Feel free to return for more hydrological discussions!"
            print(f"\n{final_message}")
            self.speak_response(final_message)
        finally:
            # Cleanup
            pygame.mixer.quit()
    
if __name__ == "__main__":
    # Load API keys from environment variables
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
    GOOGLE_CREDENTIALS_PATH = os.getenv('GOOGLE_CREDENTIALS_PATH')
    VOICE_SAMPLES_DIR = "/Users/darrieythorsson/Desktop/Marty_Voice"
    
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    
    async def main():
        marty = MARTy(
            anthropic_api_key=ANTHROPIC_API_KEY,
            google_credentials_path=GOOGLE_CREDENTIALS_PATH
        )
        
        await marty.chat_loop(anthropic_api_key=ANTHROPIC_API_KEY)

    # Run the async main function
    import asyncio
    asyncio.run(main())