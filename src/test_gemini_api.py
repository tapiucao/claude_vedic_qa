# src/test_gemini_api.py
"""
Test script for Gemini API to diagnose model availability issues.
Reads API Key and Model Name from config.
"""
import os
import sys
import google.generativeai as genai
import warnings

# Import necessary config variables safely
try:
    # Assumes script is run from project root (e.g., python src/test_gemini_api.py)
    from src.config import GEMINI_API_KEY, MODEL_NAME
except ImportError:
    # Fallback if run directly from src directory
    try:
        from config import GEMINI_API_KEY, MODEL_NAME
    except ImportError:
        print("CRITICAL ERROR: Could not import configuration (config.py). "
              "Ensure the script is run correctly relative to the 'src' directory "
              "or that 'src' is in the PYTHONPATH.", file=sys.stderr)
        sys.exit(1)

# Suppress the specific UserWarning about convert_system_message_to_human if needed elsewhere
# warnings.filterwarnings("ignore", message="Convert_system_message_to_human will be deprecated!")

def test_gemini_api():
    """Tests Gemini API connectivity, lists models, and attempts generation."""
    print("--- Testing Gemini API Connectivity ---")

    if not GEMINI_API_KEY:
        print("\nERROR: GEMINI_API_KEY is not set.")
        print("Please set it in your environment variables or a .env file in the project root.")
        sys.exit(1)
    else:
         # Mask part of the key for display
         masked_key = GEMINI_API_KEY[:4] + "*****" + GEMINI_API_KEY[-4:]
         print(f"Using API Key: {masked_key}")


    try:
        genai.configure(api_key=GEMINI_API_KEY)

        print("\nRetrieving available models...")
        models = list(genai.list_models()) # Convert iterator to list

        # Filter for models supporting 'generateContent'
        generative_models = [
            model for model in models
            if 'generateContent' in model.supported_generation_methods
        ]

        if not generative_models:
            print("\nERROR: No generative models found for your API key.")
            print("Check API key permissions (ensure Generative Language API is enabled in Google Cloud project).")
            sys.exit(1)

        print(f"\nFound {len(generative_models)} models supporting 'generateContent':")
        available_model_names_simplified = []
        for i, model in enumerate(generative_models, 1):
            simplified_name = model.name.split('/')[-1]
            available_model_names_simplified.append(simplified_name)
            print(f"  {i}. {model.name} (Simplified: {simplified_name})")
            # print(f"     - Display name: {model.display_name}") # Optional detail

        # Check configured model
        print(f"\nConfigured MODEL_NAME: '{MODEL_NAME}'")
        if MODEL_NAME in available_model_names_simplified:
             print(f"Configured model '{MODEL_NAME}' IS available.")
             test_model_name = MODEL_NAME
        elif available_model_names_simplified:
             test_model_name = available_model_names_simplified[0] # Fallback to first available
             print(f"WARNING: Configured model '{MODEL_NAME}' not found in the available list.")
             print(f"--> Will attempt test using first available model: '{test_model_name}'")
        else:
             # Should not happen if generative_models list was not empty, but safety check
             print("ERROR: No suitable models found to perform test generation.")
             sys.exit(1)

        print(f"\nAttempting simple generation with model: '{test_model_name}'...")
        model_instance = genai.GenerativeModel(test_model_name)

        # Example prompt
        prompt = "Explain the concept of 'ahimsa' in Vedic thought briefly."
        print(f"Prompt: \"{prompt}\"")

        # Generate content
        response = model_instance.generate_content(prompt)

        print("\nResponse from Gemini model:")
        print("-" * 50)
        # Robustly access response text, handling potential blocking or errors
        try:
            # Access parts if available (safer for complex responses)
            if response.parts:
                print(response.parts[0].text)
            else:
                # Fallback to .text, might raise error if blocked
                print(response.text)
        except ValueError as ve:
            print(f"ERROR: Could not extract text. Response might be blocked due to safety settings or other issues.")
            print(f"Details: {ve}")
            print("\nPrompt Feedback:", response.prompt_feedback)
            # print("Full Response Object:", response) # For debugging if needed
        except AttributeError:
             print("ERROR: Could not access response text attribute.")
             # print("Full Response Object:", response) # For debugging if needed
        except Exception as e:
             print(f"ERROR: An unexpected error occurred accessing the response: {e}")
             # print("Full Response Object:", response) # For debugging if needed


        print("-" * 50)
        print("\nAPI test sequence completed.")
        print(f"Recommendation: Ensure MODEL_NAME in config.py (currently '{MODEL_NAME}') matches an available and suitable model from the list above.")

    except google.api_core.exceptions.PermissionDenied as e:
         print(f"\nERROR: Permission Denied accessing Google AI API: {e}")
         print("Check API key validity and ensure the associated Google Cloud project has the 'Generative Language API' enabled.")
    except Exception as e:
        print(f"\nAn unexpected ERROR occurred: {str(e)}")
        print("Troubleshooting suggestions:")
        print("- Verify your GEMINI_API_KEY.")
        print("- Check network connection and firewall.")
        print("- Update Google AI library: pip install -U google-generativeai")
        print("- Check Google Cloud project status and API enablement.")

if __name__ == "__main__":
    test_gemini_api()