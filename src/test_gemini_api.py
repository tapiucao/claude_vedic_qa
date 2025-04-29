"""
Test script for Gemini API to diagnose model availability issues.
"""
import os
import sys
import google.generativeai as genai

# API key (same as in config.py)
API_KEY = 'AIzaSyDuLhEqJMWWtTseYm7V5KouXJ-605afKxY'

def test_gemini_api():
    """Test the Gemini API and display available models."""
    print("Testing Gemini API connectivity...")
    
    try:
        # Configure the API
        genai.configure(api_key=API_KEY)
        
        # List available models
        print("Retrieving available models...")
        models = genai.list_models()
        
        # Filter for Gemini models
        gemini_models = [model for model in models if "gemini" in model.name.lower()]
        
        if gemini_models:
            print(f"\nFound {len(gemini_models)} Gemini models:")
            for i, model in enumerate(gemini_models, 1):
                print(f"  {i}. {model.name}")
                # Print some model details
                print(f"     - Supported generation methods: {', '.join(model.supported_generation_methods)}")
                print(f"     - Display name: {model.display_name}")
                
            # Test a simple completion with the first available model
            print("\nTesting a simple completion with the first available model...")
            model_name = gemini_models[0].name
            simplified_name = model_name.split('/')[-1]
            
            model = genai.GenerativeModel(simplified_name)
            response = model.generate_content("What is the meaning of the Sanskrit term 'citta'?")
            
            print("\nResponse from Gemini model:")
            print("-" * 50)
            print(response.text)
            print("-" * 50)
            
            print("\nAPI test completed successfully!")
            print(f"In your config.py, use MODEL_NAME = '{simplified_name}'")
            
        else:
            print("No Gemini models found. Make sure your API key has access to Gemini models.")
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("\nPossible solutions:")
        print("1. Check if your API key is valid")
        print("2. Ensure you have access to Google Gemini API")
        print("3. Check if the Google Generative AI library is up to date:")
        print("   pip install -U google-generativeai")
        print("4. Check your internet connection")
        print("5. If using a VPN, try disabling it")

if __name__ == "__main__":
    test_gemini_api()