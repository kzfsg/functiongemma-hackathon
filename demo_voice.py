"""
CANDOR Voice Demo - Minimal Python implementation
Demonstrates the three-signal routing with actual voice transcription
"""

import sys
sys.path.insert(0, "cactus/python/src")

import json
import os
from cactus import cactus_init, cactus_transcribe, cactus_destroy
from main import generate_hybrid
from candor_tools import CANDOR_TOOLS

# Session state
conversation_history = []
whisper_model = None

def initialize():
    """Initialize Whisper model for transcription"""
    global whisper_model
    print("Initializing Whisper model...")
    whisper_model = cactus_init("cactus/weights/whisper-small")
    print("Ready for voice input!")

def transcribe_audio(audio_path):
    """Transcribe audio file to text"""
    prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
    response_str = cactus_transcribe(whisper_model, audio_path, prompt=prompt)
    response = json.loads(response_str)
    return response.get("response", "")

def process_voice_input(audio_path):
    """Main CANDOR loop: transcribe -> route -> respond"""
    # Transcribe
    print(f"\nTranscribing {audio_path}...")
    transcribed_text = transcribe_audio(audio_path)
    print(f"You said: {transcribed_text}")

    # Add to conversation
    conversation_history.append({
        "role": "user",
        "content": transcribed_text
    })

    # Route and generate response
    print("\nRouting decision...")
    result = generate_hybrid(conversation_history, CANDOR_TOOLS)

    # Display routing info
    print(f"\n--- ROUTING INFO ---")
    print(f"Route: {result.get('_candor_route', 'unknown').upper()}")
    print(f"Reason: {result.get('_candor_reason', 'N/A')}")
    print(f"Stress: {result.get('_candor_stress', 'N/A')}")
    print(f"Source: {result.get('source', 'N/A')}")
    print(f"Time: {result.get('total_time_ms', 0):.2f}ms")

    # Display function calls
    if result.get("function_calls"):
        print(f"\n--- FUNCTION CALLS ---")
        for call in result["function_calls"]:
            print(f"Function: {call['name']}")
            print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")

    # Add assistant response to history
    if result.get("response"):
        conversation_history.append({
            "role": "assistant",
            "content": result["response"]
        })

    return result

def print_session_stats():
    """Print routing statistics"""
    if hasattr(generate_hybrid, '_candor_call_count'):
        total = generate_hybrid._candor_call_count
        local = generate_hybrid._candor_local_count
        cloud = generate_hybrid._candor_cloud_count
        local_pct = (local / total * 100) if total > 0 else 0

        print(f"\n{'='*50}")
        print(f"SESSION STATS")
        print(f"{'='*50}")
        print(f"Total calls: {total}")
        print(f"Local: {local} ({local_pct:.1f}%)")
        print(f"Cloud: {cloud} ({100-local_pct:.1f}%)")
        print(f"{'='*50}\n")

def cleanup():
    """Cleanup resources"""
    global whisper_model
    if whisper_model:
        cactus_destroy(whisper_model)
        print("Whisper model destroyed")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python demo_voice.py <audio.wav>")
        print("   or: python demo_voice.py <audio1.wav> <audio2.wav> ...")
        sys.exit(1)

    try:
        initialize()

        # Process each audio file
        for audio_file in sys.argv[1:]:
            if not os.path.exists(audio_file):
                print(f"Error: {audio_file} not found")
                continue

            process_voice_input(audio_file)
            print("\n" + "="*50 + "\n")

        # Print final stats
        print_session_stats()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        cleanup()
