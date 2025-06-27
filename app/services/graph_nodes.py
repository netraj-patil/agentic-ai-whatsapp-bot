"""
Graph Nodes Module for AI Assistant with Calendar Integration

This module implements a LangGraph-based AI assistant that can handle calendar operations
and web searches. It uses LangChain components for LLM integration and tool management.

Author: Netraj Patil
"""

import logging
from datetime import datetime
from typing import Annotated, Dict, List, Optional
from zoneinfo import ZoneInfo

from langchain.schema import HumanMessage
from langchain_core.messages import ToolMessage, BaseMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from .calendar_tools import get_all_calendar_tools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEZONE = 'Asia/Kolkata'
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024
MAX_SEARCH_RESULTS = 2


class GraphState(TypedDict):
    """
    State definition for the conversation graph.
    
    Attributes:
        messages: List of conversation messages with automatic aggregation
    """
    messages: Annotated[List[BaseMessage], add_messages]


class AIAssistantConfig:
    """Configuration class for the AI Assistant."""
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timezone: str = DEFAULT_TIMEZONE
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timezone = timezone
        self.current_time = self._get_current_time()
    
    def _get_current_time(self) -> datetime:
        """Get current time in the configured timezone."""
        try:
            zone = ZoneInfo(self.timezone)
            return datetime.now(zone)
        except Exception as e:
            logger.error(f"Failed to get timezone {self.timezone}: {e}")
            # Fallback to UTC
            return datetime.now(ZoneInfo('UTC'))


class PromptManager:
    """Manages system prompts for different nodes in the graph."""
    
    def __init__(self, config: AIAssistantConfig):
        self.config = config
        self.current_time_str = config.current_time.strftime("%d-%m-%Y %H:%M:%S %Z%z")
    
    @property
    def system_prompt(self) -> str:
        """System prompt for the main chatbot node."""
        return f"""You are a helpful AI assistant with access to a calendar and web search.
        
            Current time: {self.current_time_str}

            Instructions:
            - Always resolve relative time references (e.g., 'tomorrow', 'next Monday') into absolute datetime strings
            - Ensure all tool arguments are correct before making calls
            - Only call each tool once per interaction
            - Keep responses concise and helpful
            - Handle errors gracefully and inform the user of any issues
        """
    
    @property
    def response_prompt(self) -> str:
        """System prompt for the response generation node."""
        return f"""You are a helpful AI assistant focused on providing clear, concise responses.
        
            Current time: {self.current_time_str}

            Instructions:
            - Provide helpful and accurate information
            - Keep responses concise but informative
            - Acknowledge any limitations or errors that occurred
            - Be friendly and professional in tone
        """


class LLMManager:
    """Manages LLM initialization and configuration."""
    
    def __init__(self, config: AIAssistantConfig):
        self.config = config
        self._llm = None
        self._llm_with_tools = None
    
    @property
    def llm(self) -> ChatGroq:
        """Get or create the base LLM instance."""
        if self._llm is None:
            try:
                self._llm = ChatGroq(
                    model_name=self.config.model_name,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                logger.info(f"Initialized LLM with model: {self.config.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}")
                raise
        return self._llm
    
    def get_llm_with_tools(self, tools: List) -> ChatGroq:
        """Get LLM instance bound with tools."""
        if self._llm_with_tools is None:
            try:
                self._llm_with_tools = self.llm.bind_tools(tools)
                logger.info(f"Bound {len(tools)} tools to LLM")
            except Exception as e:
                logger.error(f"Failed to bind tools to LLM: {e}")
                raise
        return self._llm_with_tools


class ToolManager:
    """Manages tool initialization and execution."""
    
    def __init__(self):
        self._tools = None
    
    def get_tools(self) -> List:
        """
        Initialize and return all available tools.
        
        Returns:
            List of initialized tools including search and calendar tools
        """
        if self._tools is None:
            try:
                # Initialize search tool
                search_tool = TavilySearch(max_results=MAX_SEARCH_RESULTS)
                
                # Get calendar tools
                calendar_tools = get_all_calendar_tools()
                
                # Combine all tools
                self._tools = [search_tool] + list(calendar_tools)
                
                logger.info(f"Initialized {len(self._tools)} tools")
                
            except Exception as e:
                logger.error(f"Failed to initialize tools: {e}")
                raise
                
        return self._tools


class ImprovedToolNode:
    """
    Enhanced tool execution node with better error handling and logging.
    
    This node handles tool calls from the LLM, executes them safely,
    and returns properly formatted tool messages.
    """
    
    def __init__(self, tools: List) -> None:
        """
        Initialize the tool node.
        
        Args:
            tools: List of tools to be made available for execution
        """
        self.tools_by_name = {tool.name: tool for tool in tools}
        logger.info(f"ToolNode initialized with tools: {list(self.tools_by_name.keys())}")
    
    def __call__(self, inputs: Dict) -> Dict[str, List[ToolMessage]]:
        """
        Execute tool calls from the latest message.
        
        Args:
            inputs: Dictionary containing conversation state
            
        Returns:
            Dictionary with executed tool results as ToolMessage objects
            
        Raises:
            ValueError: If no messages found in input
        """
        try:
            messages = inputs.get("messages", [])
            if not messages:
                raise ValueError("No messages found in input")
            
            message = messages[-1]
            
            if not hasattr(message, 'tool_calls') or not message.tool_calls:
                logger.warning("No tool calls found in message")
                return {"messages": []}
            
            outputs = []
            
            for tool_call in message.tool_calls:
                result = self._execute_tool_call(tool_call)
                outputs.append(result)
            
            logger.info(f"Executed {len(outputs)} tool calls successfully")
            return {"messages": outputs}
            
        except Exception as e:
            logger.error(f"Error in tool node execution: {e}")
            error_message = ToolMessage(
                content=f"Tool execution error: {str(e)}",
                name="error",
                tool_call_id="error"
            )
            return {"messages": [error_message]}
    
    def _execute_tool_call(self, tool_call: Dict) -> ToolMessage:
        """
        Execute a single tool call safely.
        
        Args:
            tool_call: Dictionary containing tool call information
            
        Returns:
            ToolMessage with the result or error information
        """
        tool_name = tool_call.get("name", "unknown")
        tool_id = tool_call.get("id", "unknown")
        
        try:
            if tool_name not in self.tools_by_name:
                raise ValueError(f"Tool '{tool_name}' not found")
            
            tool = self.tools_by_name[tool_name]
            args = tool_call.get("args", {})
            
            logger.info(f"Executing tool '{tool_name}' with args: {args}")
            
            result = tool.invoke(args)
            
            return ToolMessage(
                content=str(result),
                name=tool_name,
                tool_call_id=tool_id,
            )
            
        except Exception as e:
            logger.error(f"Tool '{tool_name}' execution failed: {e}")
            return ToolMessage(
                content=f"Tool '{tool_name}' error: {str(e)}",
                name=tool_name,
                tool_call_id=tool_id,
            )


class GraphNodes:
    """Contains all node functions for the conversation graph."""
    
    def __init__(self, config: AIAssistantConfig):
        self.config = config
        self.prompt_manager = PromptManager(config)
        self.llm_manager = LLMManager(config)
        self.tool_manager = ToolManager()
    
    def chatbot_node(self, state: GraphState) -> Dict[str, List[BaseMessage]]:
        """
        Main chatbot node that processes user input and decides on tool usage.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with LLM response
        """
        try:
            messages = state["messages"]
            
            # Add system prompt for first user message
            if len(messages) == 1 and isinstance(messages[0], HumanMessage):
                system_msg = {
                    "role": "system", 
                    "content": self.prompt_manager.system_prompt
                }
                messages = [system_msg, messages[0]]
            
            # Get LLM with tools and invoke
            tools = self.tool_manager.get_tools()
            llm_with_tools = self.llm_manager.get_llm_with_tools(tools)
            
            response = llm_with_tools.invoke(messages)
            
            logger.info("Chatbot node executed successfully")
            return {"messages": [response]}
            
        except Exception as e:
            logger.error(f"Error in chatbot node: {e}")
            # Return error message to continue conversation
            error_response = HumanMessage(
                content=f"I apologize, but I encountered an error: {str(e)}"
            )
            return {"messages": [error_response]}
    
    def response_node(self, state: GraphState) -> Dict[str, List[BaseMessage]]:
        """
        Response generation node that creates final user-facing responses.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with final response
        """
        try:
            messages = state["messages"]
            
            # Add response system prompt
            system_msg = {
                "role": "system", 
                "content": self.prompt_manager.response_prompt
            }
            messages = [system_msg] + messages
            
            response = self.llm_manager.llm.invoke(messages)
            
            logger.info("Response node executed successfully")
            return {"messages": [response]}
            
        except Exception as e:
            logger.error(f"Error in response node: {e}")
            error_response = HumanMessage(
                content="I apologize, but I'm having trouble generating a response right now."
            )
            return {"messages": [error_response]}
    
    def route_tools(self, state: GraphState) -> str:
        """
        Routing function to determine if tools should be called.
        
        Args:
            state: Current conversation state
            
        Returns:
            Next node name ("tools" or END)
        """
        try:
            if isinstance(state, list):
                ai_message = state[-1]
            elif messages := state.get("messages", []):
                ai_message = messages[-1]
            else:
                logger.error(f"No messages found in state: {state}")
                return END
            
            # Check if tools need to be called
            if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
                logger.info(f"Routing to tools: {len(ai_message.tool_calls)} tool calls")
                return "tools"
            
            logger.info("No tools needed, ending conversation")
            return END
            
        except Exception as e:
            logger.error(f"Error in routing: {e}")
            return END


class GraphBuilder:
    """Builds and compiles the conversation graph."""
    
    def __init__(self, config: Optional[AIAssistantConfig] = None):
        """
        Initialize the graph builder.
        
        Args:
            config: Configuration for the AI assistant. If None, uses defaults.
        """
        self.config = config or AIAssistantConfig()
        self.nodes = GraphNodes(self.config)
        self.tool_manager = ToolManager()
    
    def build_graph(self) -> StateGraph:
        """
        Build and compile the conversation graph.
        
        Returns:
            Compiled StateGraph ready for execution
            
        Raises:
            Exception: If graph compilation fails
        """
        try:
            logger.info("Building conversation graph...")
            
            # Initialize graph builder
            graph_builder = StateGraph(GraphState)
            
            # Get tools for tool node
            tools = self.tool_manager.get_tools()
            tool_node = ImprovedToolNode(tools=tools)
            
            # Add nodes to graph
            graph_builder.add_node("chatbot", self.nodes.chatbot_node)
            graph_builder.add_node("tools", tool_node)
            graph_builder.add_node("response", self.nodes.response_node)
            
            # Add edges
            graph_builder.add_conditional_edges(
                "chatbot",
                self.nodes.route_tools,
                {"tools": "tools", END: END},
            )
            graph_builder.add_edge("tools", "response")
            graph_builder.add_edge("response", END)
            graph_builder.add_edge(START, "chatbot")
            
            # Compile the graph
            graph = graph_builder.compile()
            
            logger.info("Graph compiled successfully")
            return graph
            
        except Exception as e:
            logger.error(f"Failed to build graph: {e}")
            raise


# Factory function for easy graph creation
def create_assistant_graph(config: Optional[AIAssistantConfig] = None) -> StateGraph:
    """
    Factory function to create a conversation graph with default or custom configuration.
    
    Args:
        config: Optional configuration. If None, uses defaults.
        
    Returns:
        Compiled conversation graph
        
    Example:
        >>> graph = create_assistant_graph()
        >>> result = graph.invoke({"messages": [HumanMessage("Hello")]})
    """
    builder = GraphBuilder(config)
    return builder.build_graph()


# Backwards compatibility
def build_graph() -> StateGraph:
    """
    Legacy function for backwards compatibility.
    
    Returns:
        Compiled conversation graph with default configuration
    """
    logger.warning("build_graph() is deprecated. Use create_assistant_graph() instead.")
    return create_assistant_graph()


if __name__ == "__main__":
    # Example usage
    try:
        config = AIAssistantConfig(
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=512
        )
        
        graph = create_assistant_graph(config)
        
        # Test the graph
        test_message = HumanMessage("What's the current time?")
        result = graph.invoke({"messages": [test_message]})
        
        print("Graph test successful!")
        print(f"Response: {result['messages'][-1].content}")
        
    except Exception as e:
        logger.error(f"Graph test failed: {e}")