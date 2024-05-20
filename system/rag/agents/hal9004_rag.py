from rag.abs import RagEdge, RagNode, NodeContext
from rag.nodes import ParamExtractorNode, CasualResponseNode, ToolSelectorNode, ToolCasualEndNode, WebSearchLookup, WebSearchResponse
from rag.abs import RagGraph

nodes = [
    RagNode("StartNode", start_node=True),
    CasualResponseNode(
        "CasualResponse",
        system_prompt="You are an Higly intelligent and carismatic AI, you should respond presicely but still casual to the users prompt."
    ),
    ParamExtractorNode(
        "WebExtract",
        schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            }
        },
        tool_description="""The web search function will search the web for the given query, this can be especially useful when the user want to search for current information.""",
        schema_example="""{
        "query": "the users search query"
        }""",
        tool_name="web_search"
    ),
    ParamExtractorNode(
        "MemoryLookup",
        schema={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "A clear meomory lookup description"
                }
            }
        },
        tool_description="""The memory lookup function will search the bots memory for the given description, this can be especially useful when the user want to recall a previous conversation or information.""",
        schema_example="""{
        "description": "The users memory lookup description"
        }""",
        tool_name="memory_lookup"
    ),
    ParamExtractorNode(
        "ToolUsageCategorizer",
        schema={
            "type": "object",
            "properties": {
                "intends": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["web_search", "memory_lookup", "casual"]
                    },
                    "description": "List of user intends"
                }
            }
        },
        schema_example="""{
        "intends": "List of user intends"
        }""",
        tool_name="intend_categorizer",
        tool_description="""
The tool usage categorizer function should list all intends the user has, here are the descriptions of the indends:

- web_search: The web search function will search the web for the given query, this can be especially useful when the user want to search for current information.
- memory_lookup: The memory lookup function will search the bots memory for the given description, this can be especially useful when the user want to recall a previous conversation or information.
- casual: The casual function is a casual conversation with the bot, this can be especially useful when the user want to have a casual conversation with the bot.
The casual function can only be used without the other functions intends.
""",
    ),
    ToolSelectorNode("ToolSelector"),
    WebSearchLookup("WebSearchLookup"),
    WebSearchResponse("WebSearchResponse", end_node=True),
    ToolCasualEndNode("EndNode", end_node=True)
]

def end_casual_check(edge, context: NodeContext):
    selected_tools = context.all_results["ToolUsageCategorizer"].response["intends"]
    if "casual" in selected_tools and len(selected_tools) == 1:
        edge.disabed = False
    else:
        edge.disabled = True
    print(f"*** End casual edge disabled: {edge.disabled}")
    
def use_webseach_check(edge, context: NodeContext):
    selected_tools = context.all_results["ToolUsageCategorizer"].response["intends"]
    if "web_search" in selected_tools:
        edge.disabled = False
    else:
        edge.disabled = True
    print(f"*** Web search edge disabled: {edge.disabled}")

edges = [
    # Inital stage
    RagEdge(
        start="StartNode",
        end="WebExtract"
    ),
    RagEdge(
        start="StartNode",
        end="CasualResponse"
    ),
    RagEdge(
        start="StartNode",
        end="MemoryLookup"
    ),
    RagEdge(
        start="StartNode",
        end="ToolUsageCategorizer"
    ),
    # Process first stage results
    RagEdge(
        start="WebExtract",
        end="ToolSelector"
    ),
    RagEdge(
        start="ToolUsageCategorizer",
        end="ToolSelector"
    ),
    RagEdge(
        start="MemoryLookup",
        end="ToolSelector"
    ),
    RagEdge(
        start="CasualResponse",
        end="ToolSelector"
    ),
    
    RagEdge(
        start="ToolSelector",
        end="WebSearchLookup",
        update_overwrite=use_webseach_check
    ),
    
    RagEdge(
        start="WebSearchLookup",
        end="WebSearchResponse"
    ),

    RagEdge( # Happy-Path to casual response
        start="ToolSelector",
        end="EndNode",
        update_overwrite=end_casual_check
    )
]


def get_graph() -> RagGraph:
    graph = RagGraph(
        nodes=nodes,
        edges=edges
    )
    return graph
    


    
