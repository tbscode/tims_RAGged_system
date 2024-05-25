from rag.abs import RagNode, NodeContext, RagNodeResult, YieldMessage


class ToolCasualEndNode(RagNode):
    
    def run(self, context: NodeContext):
        print(f"End node {self.name} reached")
        casual_response = context.all_results["CasualResponse"].response
        print("Casual response:", casual_response)
        return RagNodeResult(
            node_name=self.name,
            response=casual_response,
            forward=False
        )

class ToolSelectorNode(RagNode):
    # a node that evaluates context results
    def run(
            self,
            context: NodeContext
        ):
        results = context.parent_results
        assert all([results[res].forward for res in results]), "Not all results are valid"
        
        parsed = {results[res].node_name: results[res].response for res in results}
        tool_map = {
            "web_search": "WebExtract",
            "memory_lookup": "MemoryLookup",
            "casual": "CasualResponse"
        }
        selected_tools = parsed["ToolUsageCategorizer"]["intends"]
        
        tool_params = {}
        for tool in selected_tools:
            tool_params[tool_map[tool]] = parsed[tool_map[tool]]
        
        return RagNodeResult(
            node_name=self.name,
            forward=True,
            response={
                "selected_tools": selected_tools,
                "tool_params": tool_params
            },
            yield_messages=[
                YieldMessage("info", f"Selected tools: {selected_tools}"),
                YieldMessage("info", f"Tool parameters: {tool_params}")
            ],
            meta={
                "parsed": parsed
            }
        )