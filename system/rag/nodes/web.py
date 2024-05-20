from rag.abs import RagNode, NodeContext, RagNodeResult

class WebSearchLookup(RagNode):

    def run(self, context: NodeContext):
        
        print(f"*** Running {self.name}, context: {context.parent_results}")
        
        selected_tools = context.parent_results["ToolSelector"].response["selected_tools"]
        assert "web_search" in selected_tools, "Web search not selected"
        tool_params = context.parent_results["ToolSelector"].response["tool_params"]
        search_query = tool_params["WebExtract"]["query"]
        
        print(f"*** Web search query: {search_query}")
        
        return RagNodeResult(
            node_name=self.name,
            forward=True,
            response={
                "search_query": search_query
            }
        )