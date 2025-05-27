from langchain import MultiLLMAgent
# Quick start
agent = MultiLLMAgent()
agent.add_llm_provider("claude", "anthropic")
agent.add_llm_provider("gpt4", "openai")
agent.create_domain_analysis_tools()

# Process your domain
results = agent.process_domain_description("Your domain description here")