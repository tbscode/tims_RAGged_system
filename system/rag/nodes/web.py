from rag.abs import RagNode, NodeContext, RagNodeResult, DEFAULT_MODEL
from rag.models import get_model, get_client_for_model
from serpapi import GoogleSearch
import os

class WebSearchLookup(RagNode):

    def run(self, context: NodeContext):
        
        print(f"*** Running {self.name}, context: {context.parent_results}")
        
        selected_tools = context.parent_results["ToolSelector"].response["selected_tools"]
        assert "web_search" in selected_tools, "Web search not selected"
        tool_params = context.parent_results["ToolSelector"].response["tool_params"]
        search_query = tool_params["WebExtract"]["query"]
        
        print(f"*** Web search query: {search_query}")
        
        
        params = {
          "engine": "google",
          "q": search_query,
          "api_key": os.getenv("SERPAPI_API_KEY")
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        
        return RagNodeResult(
            node_name=self.name,
            forward=True,
            response={
                "search_query": search_query,
                "search_results": results
            }
        )
        

class WebSearchResponse(RagNode):
    base_prompt = """
Based on the users query, you used the 'web_search' tool to search the web for the query: {search_query}.

Here are the search results:
{web_search_results}

Use them to answer the users query.
Respond with a consize answer to the users query.
"""

    system_prompt = ""
    
    model_name = DEFAULT_MODEL
    
    def create(self, **kwargs):
        self.model_name = kwargs.get("model", self.model_name)
        self.model = get_model(self.model_name)


    def run(self, context: NodeContext):
        
        print(f"*** Running {self.name}, context: {context.parent_results}")
        
        search_results = context.parent_results["WebSearchLookup"].response
        
        self.system_prompt = self.base_prompt.format(
            search_query=search_results["search_query"],
            web_search_results=search_results["search_results"]
        )

        messages = [{
            "role": "system",
            "content": self.base_prompt
        }, {
            "role": "user",
            "content": context.prompt
        }]
        
        print("Running ParamExtractorNode", messages)
        
        completion_params = {
            "model": self.model.model,
            "messages": messages,
            "max_tokens": 400,
            "temperature": 0.0,
        }

        client = get_client_for_model(self.model.model)
        response = client.chat.completions.create(
            **completion_params
        )

        res = response.choices[0].message.content
        
        return RagNodeResult(
            node_name=self.name,
            forward=True,
            response=res        
        )