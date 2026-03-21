#!/usr/bin/env python
"""Diagnostic script to test Gemini API key and identify issues."""

import os
import sys
import time

# Test 1: Check if API key exists
print("=" * 60)
print("🔍 TEST 1: Checking API Key Configuration")
print("=" * 60)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY not found in environment")
    sys.exit(1)

print(f"✅ API Key found: {api_key[:20]}...")

# Test 2: Try importing google.genai
print("\n" + "=" * 60)
print("🔍 TEST 2: Importing google.genai SDK")
print("=" * 60)

try:
    import google.genai as genai
    print("✅ google.genai imported successfully")
except ImportError as e:
    print(f"❌ Failed to import google.genai: {e}")
    sys.exit(1)

# Test 3: Initialize client
print("\n" + "=" * 60)
print("🔍 TEST 3: Initializing Gemini Client")
print("=" * 60)

try:
    client = genai.Client(api_key=api_key)
    print("✅ Client initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize client: {e}")
    sys.exit(1)

# Test 4: List available models
print("\n" + "=" * 60)
print("🔍 TEST 4: Listing Available Models")
print("=" * 60)

try:
    models = client.models.list()
    print("✅ Successfully queried models")
    for model in models:
        if "gemini" in model.name.lower():
            print(f"  - {model.name}")
except Exception as e:
    print(f"❌ Failed to list models: {e}")
    print("   This might indicate: billing not set up, or API not enabled")

# Test 5: Test actual API call with retry
print("\n" + "=" * 60)
print("🔍 TEST 5: Testing Actual API Call")
print("=" * 60)

models_to_try = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-1.5-flash",
]

success = False
for model_name in models_to_try:
    try:
        print(f"\n⏳ Trying model: {model_name}")
        response = client.models.generate_content(
            model=model_name,
            contents="Say 'Hello, API works!' in short form.",
        )
        print(f"✅ SUCCESS with {model_name}")
        print(f"Response: {response.text[:100]}")
        success = True
        break
    except Exception as e:
        error_str = str(e).lower()
        if "404" in error_str or "not found" in error_str:
            print(f"❌ Model not found: {model_name}")
        elif "quota" in error_str or "429" in error_str:
            print(f"⚠️  Quota/Rate limit error for {model_name}")
            print(f"   Full error: {str(e)[:100]}")
        elif "billing" in error_str or "account" in error_str:
            print(f"❌ Billing/Account error for {model_name}")
            print(f"   Full error: {str(e)[:100]}")
        else:
            print(f"❌ Error with {model_name}: {str(e)[:100]}")

if not success:
    print("\n" + "=" * 60)
    print("❌ DIAGNOSIS: Possible Issues")
    print("=" * 60)
    print("""
1. ⚠️  Billing not properly set up
   → Fix: Go to console.cloud.google.com → Billing → Add payment method
   
2. ⚠️  API not enabled
   → Fix: Search 'Generative Language API' in console and enable it
   
3. ⚠️  API key permissions wrong
   → Fix: Create a new API key with proper permissions
   
4. ⚠️  Quota exceeded
   → Fix: Wait a few minutes and try again
   """)
else:
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED! API is working correctly")
    print("=" * 60)
