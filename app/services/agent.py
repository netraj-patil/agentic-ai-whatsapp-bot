"""
AI Agent Service Module

This module provides a simple interface to the LangGraph-based AI assistant
with calendar integration and web search capabilities.

Author: Netraj Patil
"""

import logging
import traceback
from typing import Optional, Dict, Any
from langchain.schema import HumanMessage, BaseMessage

# Import the graph components from your existing modules
from .graph_nodes import create_assistant_graph, AIAssistantConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIAgent:
    """
    AI Agent service class that provides a simple interface to the conversation graph.
    
    This class encapsulates the LangGraph functionality and provides easy-to-use
    methods for generating responses from user input.
    """
    
    def __init__(self, config: Optional[AIAssistantConfig] = None):
        """
        Initialize the AI Agent.
        
        Args:
            config: Optional configuration for the AI assistant.
                   If None, uses default configuration.
        """
        self.config = config or AIAssistantConfig()
        self._graph = None
        self._initialized = False
        
        logger.info("AI Agent initialized with configuration")
    
    def _initialize_graph(self) -> None:
        """
        Lazy initialization of the conversation graph.
        
        This method creates and compiles the graph only when needed,
        which helps with startup time and error handling.
        
        Raises:
            RuntimeError: If graph initialization fails
        """
        if self._initialized:
            return
            
        try:
            logger.info("Initializing conversation graph...")
            self._graph = create_assistant_graph(self.config)
            self._initialized = True
            logger.info("Conversation graph initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize conversation graph: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Agent initialization failed: {str(e)}")
    
    def generate_response(self, text: str) -> str:
        """
        Generate a response to the given text input.
        
        This is the main interface method that takes user text input,
        processes it through the conversation graph, and returns the
        AI assistant's response.
        
        Args:
            text: User input text to process
            
        Returns:
            AI assistant's response as a string
            
        Raises:
            ValueError: If input text is empty or invalid
            RuntimeError: If response generation fails
            
        Example:
            >>> agent = AIAgent()
            >>> response = agent.generate_response("What's the weather today?")
            >>> print(response)
        """
        # Validate input
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
        
        if not text.strip():
            raise ValueError("Input text cannot be empty or whitespace only")
        
        # Initialize graph if needed
        if not self._initialized:
            self._initialize_graph()
        
        try:
            logger.info(f"Processing user input: {text[:100]}...")  # Log first 100 chars
            
            # Create human message from input text
            user_message = HumanMessage(content=text.strip())
            
            # Invoke the conversation graph
            result = self._graph.invoke({"messages": [user_message]})
            
            # Extract the response from the result
            response_text = self._extract_response(result)
            
            logger.info("Response generated successfully")
            return response_text
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Response generation failed: {str(e)}")
    
    def _extract_response(self, result: Dict[str, Any]) -> str:
        """
        Extract the final response text from the graph result.
        
        Args:
            result: Result dictionary from graph execution
            
        Returns:
            Final response text as string
            
        Raises:
            ValueError: If no valid response found in result
        """
        try:
            messages = result.get("messages", [])
            
            if not messages:
                raise ValueError("No messages found in graph result")
            
            # Get the last message (should be the final response)
            last_message = messages[-1]
            
            # Extract content based on message type
            if hasattr(last_message, 'content'):
                response_content = last_message.content
            elif isinstance(last_message, dict):
                response_content = last_message.get('content', '')
            else:
                response_content = str(last_message)
            
            if not response_content or not str(response_content).strip():
                raise ValueError("Empty response content")
            
            return str(response_content).strip()
            
        except Exception as e:
            logger.error(f"Failed to extract response: {e}")
            return "I apologize, but I encountered an issue generating a response. Please try again."
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the agent.
        
        Returns:
            Dictionary containing health status information
        """
        try:
            if not self._initialized:
                self._initialize_graph()
            
            # Test with a simple message
            test_response = self.generate_response("Hello")
            
            return {
                "status": "healthy",
                "initialized": self._initialized,
                "model": self.config.model_name,
                "timezone": self.config.timezone,
                "test_response_length": len(test_response)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "initialized": self._initialized
            }


# Global agent instance for easy access
_agent_instance: Optional[AIAgent] = None


def get_agent(config: Optional[AIAssistantConfig] = None) -> AIAgent:
    """
    Get or create a global AI agent instance.
    
    This function implements a singleton pattern for the agent,
    ensuring only one instance is created and reused.
    
    Args:
        config: Optional configuration. Only used if no instance exists yet.
        
    Returns:
        AI Agent instance
    """
    global _agent_instance
    
    if _agent_instance is None:
        logger.info("Creating new AI agent instance")
        _agent_instance = AIAgent(config)
    
    return _agent_instance


def generate_response(text: str, config: Optional[AIAssistantConfig] = None) -> str:
    """
    Convenience function to generate a response using the global agent instance.
    
    This is the main function that external services should use to interact
    with the AI assistant.
    
    Args:
        text: User input text
        config: Optional configuration (only used for first call)
        
    Returns:
        AI assistant's response as string
        
    Example:
        >>> from services.agent import generate_response
        >>> response = generate_response("Schedule a meeting for tomorrow at 2 PM")
        >>> print(response)
    """
    agent = get_agent(config)
    return agent.generate_response(text)


# Backwards compatibility and convenience functions
def create_agent_with_custom_config(
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timezone: str = "Asia/Kolkata"
) -> AIAgent:
    """
    Create an AI agent with custom configuration.
    
    Args:
        model_name: LLM model to use
        temperature: Temperature for response generation
        max_tokens: Maximum tokens in response
        timezone: Timezone for calendar operations
        
    Returns:
        Configured AI Agent instance
    """
    config = AIAssistantConfig(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        timezone=timezone
    )
    return AIAgent(config)


if __name__ == "__main__":
    """
    Test the agent functionality when run directly.
    """
    try:
        print("Testing AI Agent...")
        
        # Test basic functionality
        response = generate_response("What's the current time?")
        print(f"Test response: {response}")
        
        # Test health check
        agent = get_agent()
        health = agent.health_check()
        print(f"Health check: {health}")
        
        print("AI Agent test completed successfully!")
        
    except Exception as e:
        print(f"AI Agent test failed: {e}")
        traceback.print_exc()