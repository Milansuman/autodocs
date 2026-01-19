from typing import TypedDict, Annotated, Any
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import operator
import os
from dotenv import load_dotenv
from .tools import (
    find_project_sources,
    get_file_symbols,
    get_module_docstring,
    get_module_variables,
    get_module_functions,
    get_module_classes,
    get_specific_function,
    get_specific_class,
    get_specific_variable,
    save_documentation
)

class AgentContext(TypedDict):
    project_path: str
    file_paths: list[str]
    current_file_index: int
    file_symbols: dict[str, str]
    file_docstrings: dict[str, str]  # Module-level docstrings
    file_variables: dict[str, list]  # Module-level variables
    file_functions: dict[str, list]  # Module-level functions
    file_classes: dict[str, list]  # Module-level classes
    documentation: dict[str, str]
    file_topics: dict[str, str]  # Maps file path to topic
    messages: Annotated[list, operator.add]
    codebase_extensions: list[str]
    output_dir: str
    topics: list[str]  # Optional list of topics to organize by

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_STRING = os.getenv("GROQ_MODEL_STRING")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")

if not GROQ_MODEL_STRING:
    GROQ_MODEL_STRING = "openai/gpt-oss-120b"

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL_STRING
)

def find_sources_node(state: AgentContext) -> AgentContext:
    """Find all source files in the project."""
    project_path = state.get("project_path", ".")
    extensions = state.get("codebase_extensions", [".py"])
    
    file_paths = find_project_sources(extensions, project_path)
    
    return {
        "file_paths": file_paths,
        "current_file_index": 0,
        "file_symbols": {},
        "documentation": {},
        "messages": [HumanMessage(content=f"Found {len(file_paths)} source files to document.")]
    }

def analyze_file_node(state: AgentContext) -> AgentContext:
    """Analyze the current file and extract symbols."""
    file_paths = state["file_paths"]
    current_index = state["current_file_index"]
    file_symbols = state.get("file_symbols", {})
    file_docstrings = state.get("file_docstrings", {})
    file_variables = state.get("file_variables", {})
    file_functions = state.get("file_functions", {})
    file_classes = state.get("file_classes", {})
    
    if current_index >= len(file_paths):
        return {"messages": [HumanMessage(content="All files analyzed.")]}
    
    current_file = file_paths[current_index]
    
    # Determine language based on extension
    lang = "python" if current_file.endswith(".py") else "unknown"
    
    if lang == "unknown":
        return {
            "current_file_index": current_index + 1,
            "messages": [HumanMessage(content=f"Skipping unsupported file: {current_file}")]
        }
    
    # Get summary of symbols
    symbols = get_file_symbols(current_file, lang)
    file_symbols[current_file] = symbols
    
    # Get module docstring
    docstring = get_module_docstring(current_file, lang)
    file_docstrings[current_file] = docstring
    
    # First, get all symbols to see what's available (exclude private symbols)
    all_variables = get_module_variables(current_file, lang, include_private=False)
    all_functions = get_module_functions(current_file, lang, include_private=False)
    all_classes = get_module_classes(current_file, lang, include_private=False)
    
    # Now use specific tools to selectively fetch important symbols
    # For now, we'll fetch all public symbols, but this structure allows
    # for more selective fetching in the future (e.g., only exported symbols)
    selected_variables = []
    for var in all_variables:
        # Use specific tool to get detailed info
        specific_var = get_specific_variable(current_file, lang, var['name'])
        if specific_var:
            selected_variables.append(specific_var)
    
    selected_functions = []
    for func in all_functions:
        # Use specific tool to get detailed info
        specific_func = get_specific_function(current_file, lang, func['name'])
        if specific_func:
            selected_functions.append(specific_func)
    
    selected_classes = []
    for cls in all_classes:
        # Use specific tool to get detailed info
        specific_cls = get_specific_class(current_file, lang, cls['name'])
        if specific_cls:
            selected_classes.append(specific_cls)
    
    file_variables[current_file] = selected_variables
    file_functions[current_file] = selected_functions
    file_classes[current_file] = selected_classes
    
    return {
        "file_symbols": file_symbols,
        "file_docstrings": file_docstrings,
        "file_variables": file_variables,
        "file_functions": file_functions,
        "file_classes": file_classes,
        "messages": [HumanMessage(content=f"Analyzed file: {current_file} using specific symbol tools")]
    }

def categorize_file_node(state: AgentContext) -> AgentContext:
    """Categorize the current file into a topic."""
    file_paths = state["file_paths"]
    current_index = state["current_file_index"]
    file_symbols = state["file_symbols"]
    file_topics = state.get("file_topics", {})
    topics = state.get("topics", [])
    
    if current_index >= len(file_paths):
        return {"messages": [HumanMessage(content="All files categorized.")]}
    
    current_file = file_paths[current_index]
    symbols = file_symbols.get(current_file, "")
    
    if not symbols:
        file_topics[current_file] = "general"
        return {
            "file_topics": file_topics,
            "messages": [HumanMessage(content=f"Categorized {current_file} as: general")]
        }
    
    # If topics are provided, categorize the file
    if topics:
        categorization_prompt = f"""You are categorizing code files into topics.

Available topics: {', '.join(topics)}

File symbols:
{symbols}

File path: {current_file}

Based on the symbols and file path, choose the MOST appropriate topic from the list above. Respond with ONLY the topic name, nothing else."""
        
        messages = [HumanMessage(content=categorization_prompt)]
        response = llm.invoke(messages)
        topic = response.content.strip()
        
        # Validate topic is in the list
        if topic not in topics:
            # Use first topic as fallback
            topic = topics[0] if topics else "general"
    else:
        # If no topics specified, use filename or a default category
        topic = "general"
    
    file_topics[current_file] = topic
    
    return {
        "file_topics": file_topics,
        "messages": [HumanMessage(content=f"Categorized {current_file} as: {topic}")]
    }

def generate_documentation_node(state: AgentContext) -> AgentContext:
    """Generate documentation for the current file using LLM."""
    file_paths = state["file_paths"]
    current_index = state["current_file_index"]
    file_symbols = state["file_symbols"]
    file_docstrings = state.get("file_docstrings", {})
    file_variables = state.get("file_variables", {})
    file_functions = state.get("file_functions", {})
    file_classes = state.get("file_classes", {})
    documentation = state.get("documentation", {})
    file_topics = state.get("file_topics", {})
    
    if current_index >= len(file_paths):
        return {"messages": [HumanMessage(content="All documentation generated.")]}
    
    current_file = file_paths[current_index]
    symbols = file_symbols.get(current_file, "")
    topic = file_topics.get(current_file, "general")
    docstring = file_docstrings.get(current_file, "")
    variables = file_variables.get(current_file, [])
    functions = file_functions.get(current_file, [])
    classes = file_classes.get(current_file, [])
    
    # Skip if no public symbols found
    if not symbols or (not variables and not functions and not classes):
        return {
            "current_file_index": current_index + 1,
            "messages": [HumanMessage(content=f"No public symbols found for: {current_file}")]
        }
    
    # Build detailed context for the LLM
    detailed_context = f"""File path: {current_file}

## Module Docstring:
{docstring if docstring else 'No module docstring'}

## Module-level Variables:
"""
    
    if variables:
        for var in variables:
            detailed_context += f"\n- **{var['name']}**: {var['value']}"
    else:
        detailed_context += "\nNo module-level variables."
    
    detailed_context += "\n\n## Functions:\n"
    if functions:
        for func in functions:
            detailed_context += f"\n### {func['name']}({', '.join(func['parameters'])})\n"
            if func.get('docstring'):
                detailed_context += f"Docstring: {func['docstring']}\n"
            if func.get('decorators'):
                detailed_context += f"Decorators: {', '.join(func['decorators'])}\n"
    else:
        detailed_context += "No functions defined.\n"
    
    detailed_context += "\n## Classes:\n"
    if classes:
        for cls in classes:
            detailed_context += f"\n### {cls['name']}"
            if cls.get('bases'):
                detailed_context += f" (inherits from: {', '.join(cls['bases'])})"
            detailed_context += "\n"
            if cls.get('docstring'):
                detailed_context += f"Docstring: {cls['docstring']}\n"
            if cls.get('methods'):
                detailed_context += "Methods:\n"
                for method in cls['methods']:
                    detailed_context += f"  - {method['name']}({', '.join(method['parameters'])})\n"
                    if method.get('docstring'):
                        detailed_context += f"    Docstring: {method['docstring']}\n"
    else:
        detailed_context += "No classes defined.\n"
    
    # Generate documentation using LLM
    system_prompt = """You are a technical documentation writer creating quickstart guides. Generate practical, example-focused documentation in Markdown format.

Your documentation should:
1. Start with a brief title and one-sentence overview
2. Focus on HOW TO USE the module, not detailed API documentation
3. Provide practical, working code examples that show common use cases
4. Be concise - quickstart guides should be easy to scan and follow
5. Don't document individual classes or functions in detail - instead show how to accomplish tasks
6. Use real, executable code examples
7. Only document what actually exists in the module - don't invent features

Format:
- Brief introduction (1-2 sentences)
- Quick installation/import instructions if relevant
- 2-3 practical code examples showing common tasks
- Keep it short and actionable

DO NOT write detailed API documentation. Focus on practical usage."""

    user_prompt = f"""Create a quickstart guide for this Python module. Focus on practical examples, not detailed API documentation.

{detailed_context}

Topic: {topic}

IMPORTANT:
- Only document features that actually exist (shown above)
- Focus on practical examples showing HOW TO USE the module
- Keep it concise - this is a quickstart guide, not comprehensive API docs
- Show 2-3 common use cases with working code examples
- Don't document each class/function individually - show how to accomplish tasks"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    doc_content = response.content
    
    documentation[current_file] = doc_content
    
    return {
        "documentation": documentation,
        "messages": [HumanMessage(content=f"Generated documentation for: {current_file}")]
    }

def save_documentation_node(state: AgentContext) -> AgentContext:
    """Save the documentation for the current file."""
    file_paths = state["file_paths"]
    current_index = state["current_file_index"]
    documentation = state["documentation"]
    file_topics = state.get("file_topics", {})
    output_dir = state.get("output_dir", "docs")
    
    if current_index >= len(file_paths):
        return {"messages": [HumanMessage(content="All documentation saved.")]}
    
    current_file = file_paths[current_index]
    doc_content = documentation.get(current_file, "")
    topic = file_topics.get(current_file, "general")
    
    if not doc_content:
        return {
            "current_file_index": current_index + 1,
            "messages": [HumanMessage(content=f"No documentation to save for: {current_file}")]
        }
    
    # Generate output path organized by topic
    filename = os.path.basename(current_file)
    doc_filename = os.path.splitext(filename)[0] + ".md"
    output_path = os.path.join(output_dir, topic, doc_filename)
    
    save_documentation(output_path, doc_content)
    
    return {
        "current_file_index": current_index + 1,
        "messages": [HumanMessage(content=f"Saved documentation to: {output_path}")]
    }

def should_continue(state: AgentContext) -> str:
    """Decide whether to continue processing files or end."""
    file_paths = state["file_paths"]
    current_index = state["current_file_index"]
    
    if current_index >= len(file_paths):
        return "end"
    return "continue"

# Build the agent graph
agent_builder = StateGraph(AgentContext)

# Add nodes
agent_builder.add_node("find_sources", find_sources_node)
agent_builder.add_node("analyze_file", analyze_file_node)
agent_builder.add_node("categorize_file", categorize_file_node)
agent_builder.add_node("generate_docs", generate_documentation_node)
agent_builder.add_node("save_docs", save_documentation_node)

# Add edges
agent_builder.add_edge(START, "find_sources")
agent_builder.add_edge("find_sources", "analyze_file")
agent_builder.add_edge("analyze_file", "categorize_file")
agent_builder.add_edge("categorize_file", "generate_docs")
agent_builder.add_edge("generate_docs", "save_docs")

# Add conditional edge to loop through files
agent_builder.add_conditional_edges(
    "save_docs",
    should_continue,
    {
        "continue": "analyze_file",
        "end": END
    }
)

agent_graph = agent_builder.compile()

def generate_documentation(
    project_path: str = ".",
    codebase_extensions: list[str] = None,
    output_dir: str = "docs",
    topics: list[str] = None
) -> dict:
    """Run the documentation generation agent.
    
    Args:
        project_path: Path to the project to document
        codebase_extensions: List of file extensions to process (default: [".py"])
        output_dir: Directory to save documentation (default: "docs")
        topics: Optional list of topics to organize documentation by
    
    Returns:
        Final state of the agent after processing all files
    """
    if codebase_extensions is None:
        codebase_extensions = [".py"]
    
    if topics is None:
        topics = []
    
    initial_state = {
        "project_path": project_path,
        "codebase_extensions": codebase_extensions,
        "output_dir": output_dir,
        "topics": topics,
        "file_paths": [],
        "current_file_index": 0,
        "file_symbols": {},
        "file_docstrings": {},
        "file_variables": {},
        "file_functions": {},
        "file_classes": {},
        "documentation": {},
        "file_topics": {},
        "messages": []
    }
    
    result = agent_graph.invoke(initial_state)
    return result