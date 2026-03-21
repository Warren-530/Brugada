#!/usr/bin/env python3
"""Quick test of the API call format."""

import os
import tomllib

# Load API key
secrets_path = ".streamlit/secrets.toml"
if os.path.exists(secrets_path):
    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)
        api_key = secrets.get("GEMINI_API_KEY")
else:
    api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ No API key found")
    exit(1)

import google.genai as genai

client = genai.Client(api_key=api_key)

model = "gemini-2.0-flash"
print(f"Testing API with model: {model}")
print("-" * 60)

try:
    # Test 1: Simple text message (new format)
    print("\n1️⃣ Testing simple text message...")
    response = client.models.generate_content(
        model=model,
        contents="Hello, what is 2+2?",
    )
    print(f"✓ Response: {response.text[:100]}")

    # Test 2: With system instruction prepended
    print("\n2️⃣ Testing with system instruction prepended to message...")
    system = "You are a medical expert. Be brief."
    message = "What is Brugada syndrome?"
    full_msg = f"{system}\n\n{message}"
    response = client.models.generate_content(
        model=model,
        contents=full_msg,
    )
    print(f"✓ Response: {response.text[:100]}")

    print("\n✅ API calls work correctly!")
    print("The issue was with message formatting in the code.")

except Exception as e:
    print(f"❌ API Error: {str(e)[:200]}")
