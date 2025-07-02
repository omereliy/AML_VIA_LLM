#!/usr/bin/env python3
"""
API Key Validation Script
Validates the configuration and basic connectivity of LLM provider API keys
"""

import os
import requests
from dotenv import load_dotenv
from typing import Dict, Tuple


def check_api_key_format(key: str, provider: str) -> bool:
    """Basic format validation for API keys"""
    if not key or key == 'your_api_key_here':
        return False
    
    # Basic format checks
    format_checks = {
        'anthropic': key.startswith('sk-ant-'),
        'openai': key.startswith('sk-'),
        'google': len(key) > 20,  # Google keys are typically longer
        'cohere': len(key) > 20   # Cohere keys are typically longer
    }
    
    return format_checks.get(provider.lower(), len(key) > 10)


def validate_anthropic_key(api_key: str) -> Tuple[bool, str]:
    """Validate Anthropic API key with a simple API call"""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        # Simple test call
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )
        return True, "Connected successfully"
    except ImportError:
        return False, "anthropic package not installed"
    except Exception as e:
        return False, f"Connection failed: {str(e)[:50]}..."


def validate_openai_key(api_key: str) -> Tuple[bool, str]:
    """Validate OpenAI API key with a simple API call"""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        # Simple test call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        return True, "Connected successfully"
    except ImportError:
        return False, "openai package not installed"
    except Exception as e:
        return False, f"Connection failed: {str(e)[:50]}..."


def validate_google_key(api_key: str) -> Tuple[bool, str]:
    """Validate Google API key with a simple API call"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        # Simple test call
        response = model.generate_content("Hi")
        return True, "Connected successfully"
    except ImportError:
        return False, "google-generativeai package not installed"
    except Exception as e:
        return False, f"Connection failed: {str(e)[:50]}..."


def validate_cohere_key(api_key: str) -> Tuple[bool, str]:
    """Validate Cohere API key with a simple API call"""
    try:
        import cohere
        client = cohere.Client(api_key=api_key)
        # Simple test call
        response = client.generate(
            model='command',
            prompt='Hi',
            max_tokens=5
        )
        return True, "Connected successfully"
    except ImportError:
        return False, "cohere package not installed"
    except Exception as e:
        return False, f"Connection failed: {str(e)[:50]}..."


def validate_api_keys(test_connections: bool = True) -> Dict[str, Dict]:
    """
    Validate all configured API keys
    
    Args:
        test_connections: Whether to test actual API connections (requires packages)
    
    Returns:
        Dictionary with validation results for each provider
    """
    load_dotenv()
    
    providers = {
        'ANTHROPIC_API_KEY': {
            'name': 'Anthropic (Claude)',
            'validator': validate_anthropic_key,
            'provider': 'anthropic'
        },
        'OPENAI_API_KEY': {
            'name': 'OpenAI (ChatGPT/GPT-4)',
            'validator': validate_openai_key,
            'provider': 'openai'
        },
        'GOOGLE_API_KEY': {
            'name': 'Google (Gemini)',
            'validator': validate_google_key,
            'provider': 'google'
        },
        'COHERE_API_KEY': {
            'name': 'Cohere',
            'validator': validate_cohere_key,
            'provider': 'cohere'
        }
    }
    
    results = {}
    
    for env_var, config in providers.items():
        api_key = os.getenv(env_var)
        provider_name = config['name']
        
        result = {
            'configured': False,
            'format_valid': False,
            'connection_valid': False,
            'message': 'Not configured'
        }
        
        if api_key and api_key != 'your_api_key_here':
            result['configured'] = True
            result['format_valid'] = check_api_key_format(api_key, config['provider'])
            
            if not result['format_valid']:
                result['message'] = 'Invalid key format'
            elif test_connections:
                is_valid, message = config['validator'](api_key)
                result['connection_valid'] = is_valid
                result['message'] = message
            else:
                result['message'] = 'Format valid (connection not tested)'
        
        results[provider_name] = result
    
    return results


def create_env_template():
    """Create a .env template file if it doesn't exist"""
    template = '''# LLM API Keys
# Get these from the respective provider websites

# Anthropic Claude - https://console.anthropic.com
ANTHROPIC_API_KEY=your_api_key_here

# OpenAI ChatGPT/GPT-4 - https://platform.openai.com
OPENAI_API_KEY=your_api_key_here

# Google Gemini - https://aistudio.google.com
GOOGLE_API_KEY=your_api_key_here

# Cohere - https://dashboard.cohere.com
COHERE_API_KEY=your_api_key_here
'''
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(template)
        print("✅ Created .env template file")
        print("📝 Edit .env file with your actual API keys")
        return True
    else:
        print("⚠️  .env file already exists")
        return False


def print_validation_results(results: Dict[str, Dict]):
    """Print formatted validation results"""
    print("🔑 API Key Validation Results\n")
    print("-" * 50)
    
    configured_count = 0
    valid_count = 0
    
    for provider, result in results.items():
        if result['configured']:
            configured_count += 1
            if result['connection_valid']:
                valid_count += 1
                status = "✅"
            elif result['format_valid']:
                status = "⚠️ "
            else:
                status = "❌"
        else:
            status = "❌"
        
        print(f"{status} {provider}: {result['message']}")
    
    print(f"\n📊 Summary:")
    print(f"   Configured: {configured_count}/{len(results)}")
    print(f"   Valid & Connected: {valid_count}/{len(results)}")
    
    if valid_count > 0:
        valid_providers = [name for name, result in results.items() 
                          if result['connection_valid']]
        print(f"🎉 Ready to use: {', '.join(valid_providers)}")
    
    if configured_count == 0:
        print("\n💡 No API keys configured. Run with --setup to create .env template")


def main():
    """Main script function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate LLM API keys')
    parser.add_argument('--setup', action='store_true', 
                       help='Create .env template file')
    parser.add_argument('--no-test', action='store_true',
                       help='Skip connection testing (format validation only)')
    
    args = parser.parse_args()
    
    print("🔍 LLM API Key Validator\n")
    
    if args.setup:
        create_env_template()
        print()
    
    # Validate keys
    test_connections = not args.no_test
    results = validate_api_keys(test_connections=test_connections)
    print_validation_results(results)
    
    if not test_connections:
        print("\n💡 Run without --no-test flag to test actual connections")
    
    print(f"\n📝 Next steps:")
    print("1. Get API keys from provider websites")
    print("2. Add them to your .env file")
    print("3. Run this script again to validate")


if __name__ == "__main__":
    main()