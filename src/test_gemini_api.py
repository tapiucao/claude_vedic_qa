# src/test_gemini_api.py
"""
Test script for Gemini API to diagnose model availability issues.
Reads API Key from config.
"""
import os
import sys
import google.generativeai as genai

# Import API Key from config
# Assuming src is in the Python path or run from the parent directory
try:
    from src.config import GEMINI_API_KEY, MODEL_NAME
except ImportError:
    print("Error: Could not import configuration. Make sure src is in PYTHONPATH or run from the project root.")
    # Attempt to load directly if run from src directory
    try:
         from config import GEMINI_API_KEY, MODEL_NAME
    except ImportError:
        print("Error: config.py not found.")
        sys.exit(1)


def test_gemini_api():
    """Test the Gemini API and display available models."""
    print("Testing Gemini API connectivity...")

    # Check if API key is loaded
    if not GEMINI_API_KEY:
        print("\nError: GEMINI_API_KEY is not set in your environment variables or .env file.")
        print("Please set the API key and try again.")
        sys.exit(1) # Exit if no key

    try:
        # Configure the API
        genai.configure(api_key=GEMINI_API_KEY)

        # List available models
        print("Retrieving available models...")
        models = genai.list_models()

        # Filter for generative models (check for 'generateContent' method)
        generative_models = [
            model for model in models
            if 'generateContent' in model.supported_generation_methods
        ]

        if generative_models:
            print(f"\nFound {len(generative_models)} models supporting 'generateContent':")
            available_model_names = []
            for i, model in enumerate(generative_models, 1):
                # Model name often looks like "models/gemini-1.5-flash-latest"
                # We usually use the shorter name like "gemini-1.5-flash-latest"
                simplified_name = model.name.split('/')[-1]
                available_model_names.append(simplified_name)
                print(f"  {i}. {model.name} (Simplified: {simplified_name})")
                print(f"     - Display name: {model.display_name}")
                print(f"     - Supported generation methods: {', '.join(model.supported_generation_methods)}")


            # Check if the configured MODEL_NAME is available
            print(f"\nChecking availability of configured model: '{MODEL_NAME}'...")
            if MODEL_NAME in available_model_names:
                 print(f"Model '{MODEL_NAME}' is available.")
                 test_model_name = MODEL_NAME
            elif available_model_names:
                 test_model_name = available_model_names[0] # Fallback to first available
                 print(f"Warning: Configured model '{MODEL_NAME}' not found in list.")
                 print(f"Falling back to test with first available model: '{test_model_name}'")
            else:
                 print("Error: No suitable generative models found, cannot perform test generation.")
                 sys.exit(1)


            # Test a simple completion with the selected model
            print(f"\nTesting simple generation with model: '{test_model_name}'...")
            model = genai.GenerativeModel(test_model_name)
            response = model.generate_content("What is the meaning of the Sanskrit term 'citta'?")

            print("\nResponse from Gemini model:")
            print("-" * 50)
            # Accessing response content might vary slightly depending on version/model
            # Using response.text is common for simple text generation
            try:
                print(response.text)
            except ValueError as ve:
                print(f"Could not extract text directly. Might be blocked. Details: {ve}")
                print("Full response object:", response)
            except AttributeError:
                 print("Could not access response.text. Full response object:", response)

            print("-" * 50)

            print("\nAPI test completed.")
            print(f"Ensure MODEL_NAME in config.py is set to an available model (e.g., '{test_model_name}' or another from the list).")

        else:
            print("\nError: No generative models found for your API key.")
            print("Please check your API key permissions and ensure it has access to Gemini models.")

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("\nPossible solutions:")
        print("1. Verify your GEMINI_API_KEY is correct and active.")
        print("2. Check network connectivity and firewall settings.")
        print("3. Ensure Google Cloud Project associated with the key has 'Generative Language API' enabled.")
        print("4. Update the Google Generative AI library: pip install -U google-generativeai")
        print("5. If using a VPN, try temporarily disabling it.")

if __name__ == "__main__":
    test_gemini_api()