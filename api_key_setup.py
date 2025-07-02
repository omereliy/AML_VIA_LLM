#!/usr/bin/env python3
"""
API Key Setup Helper Script
Run this to check which API keys you have configured
"""

import os
from dotenv import load_dotenv

def check_api_keys():
    """Check which API keys are configured"""
    
    load_dotenv()
    
    keys_to_check = {
        'ANTHROPIC_API_KEY': 'Anthropic (Claude)',
        'OPENAI_API_KEY': 'OpenAI (ChatGPT/GPT-4)',
        'GOOGLE_API_KEY': 'Google (Gemini)',
        'COHERE_API_KEY': 'Cohere',
    }
    
    print("🔑 API Key Configuration Status\n")
    print("-" * 50)
    
    configured_keys = []
    missing_keys = []
    
    for env_var, service_name in keys_to_check.items():
        key_value = os.getenv(env_var)
        
        if key_value and key_value != 'your_api_key_here':
            print(f"✅ {service_name}: Configured")
            configured_keys.append(service_name)
        else:
            print(f"❌ {service_name}: Not configured")
            missing_keys.append(service_name)
    
    print(f"\n📊 Summary:")
    print(f"   Configured: {len(configured_keys)}/{len(keys_to_check)}")
    print(f"   Missing: {len(missing_keys)}")
    
    if missing_keys:
        print(f"\n🚨 Missing API keys for: {', '.join(missing_keys)}")
        print("💡 See the setup guide above to get these keys")
    
    if configured_keys:
        print(f"\n🎉 You can use: {', '.join(configured_keys)}")
    
    return len(configured_keys) > 0

def create_env_template():
    """Create a .env template file"""
    
    template = """# LLM API Keys
# Get these from the respective provider websites

# Anthropic Claude - https://console.anthropic.com
ANTHROPIC_API_KEY=your_api_key_here

# OpenAI ChatGPT/GPT-4 - https://platform.openai.com
OPENAI_API_KEY=your_api_key_here

# Google Gemini - https://aistudio.google.com
GOOGLE_API_KEY=your_api_key_here

# Cohere - https://dashboard.cohere.com
COHERE_API_KEY=your_api_key_here
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(template)
        print("✅ Created .env template file")
        print("📝 Edit .env file with your actual API keys")
    else:
        print("⚠️  .env file already exists")

def verify_connections():
    """Test connections to available APIs"""
    
    load_dotenv()
    
    print("\n🔗 Testing API Connections\n")
    print("-" * 30)
    
    # Test Anthropic
    if os.getenv('ANTHROPIC_API_KEY') and os.getenv('ANTHROPIC_API_KEY') != 'your_api_key_here':
        try:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                model="claude-3-haiku-20240307",  # Cheapest model for testing
                anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
            )
            response = llm.invoke("Hello! Just testing the connection.")
            print("✅ Anthropic (Claude): Connected")
        except Exception as e:
            print(f"❌ Anthropic (Claude): Error - {str(e)[:50]}...")
    
    # Test OpenAI
    if os.getenv('OPENAI_API_KEY') and os.getenv('OPENAI_API_KEY') != 'your_api_key_here':
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",  # Cheapest model for testing
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            response = llm.invoke("Hello! Just testing the connection.")
            print("✅ OpenAI (ChatGPT): Connected")
        except Exception as e:
            print(f"❌ OpenAI (ChatGPT): Error - {str(e)}")
    
    # Test Google
    if os.getenv('GOOGLE_API_KEY') and os.getenv('GOOGLE_API_KEY') != 'your_api_key_here':
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=os.getenv('GOOGLE_API_KEY')
            )
            response = llm.invoke("Hello! Just testing the connection.")
            print("✅ Google (Gemini): Connected")
        except Exception as e:
            print(f"❌ Google (Gemini): Error - {str(e)[:50]}...")
    
    # Test Cohere
    if os.getenv('COHERE_API_KEY') and os.getenv('COHERE_API_KEY') != 'your_api_key_here':
        try:
            from langchain_cohere import ChatCohere
            llm = ChatCohere(
                model="command-light",  # Cheaper model for testing
                cohere_api_key=os.getenv('COHERE_API_KEY')
            )
            response = llm.invoke("Hello! Just testing the connection.")
            print("✅ Cohere: Connected")
        except Exception as e:
            print(f"❌ Cohere: Error - {str(e)[:50]}...")

if __name__ == "__main__":
    print("🚀 LLM API Key Setup Helper\n")
    
    # Create .env template if it doesn't exist
    create_env_template()
    
    # Check current configuration
    has_keys = check_api_keys()
    
    if has_keys:
        # Test connections if we have some keys
        try:
            verify_connections()
        except ImportError as e:
            print(f"\n⚠️  Install required packages first: pip install -r requirements.txt")
    
    print(f"\n💡 Next steps:")
    print("1. Get API keys from the provider websites")
    print("2. Add them to your .env file")
    print("3. Run this script again to test connections")
    print("4. Start building your LLM agents!")
