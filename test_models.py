#!/usr/bin/env python3
"""Test which Gemini models are available."""

import os
import sys
import tomllib

# Load API key from .streamlit/secrets.toml
secrets_path = ".streamlit/secrets.toml"
api_key = None

if os.path.exists(secrets_path):
    try:
        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)
            api_key = secrets.get("GEMINI_API_KEY")
    except Exception as e:
        print(f"❌ Failed to read secrets: {e}")

# Fallback to environment variable
if not api_key:
    api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found in .streamlit/secrets.toml or environment")
    sys.exit(1)

print(f"✓ API key loaded (last 8 chars: ...{api_key[-8:]})")
print()

try:
    import google.genai as genai
except ImportError:
    print("❌ google-genai not installed")
    sys.exit(1)

# Initialize client
client = genai.Client(api_key=api_key)

# Models to test (ordered by preference)
models_to_test = [
    "gemini-2.0-flash-001",
    "gemini-1.5-flash-001", 
    "gemini-1.5-pro-001",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

print("=" * 70)
print("Testing available Gemini models (google-genai SDK)")
print("=" * 70)
print()

for model_name in models_to_test:
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[{"role": "user", "parts": [{"text": "test"}]}],
        )
        print(f"✓ {model_name:<30} - AVAILABLE")
    except Exception as e:
        error = str(e)
        if "404" in error or "not found" in error.lower():
            print(f"✗ {model_name:<30} - NOT FOUND (404)")
        elif "quota" in error.lower() or "429" in error:
            print(f"⚠ {model_name:<30} - QUOTA EXCEEDED")
        else:
            print(f"✗ {model_name:<30} - ERROR: {error[:50]}")

print()
print("=" * 70)
