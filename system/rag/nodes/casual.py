from rag.abs import RagNode, RagNodeResult, NodeContext, DEFAULT_MODEL
from rag.models import get_model, get_client_for_model

class CasualResponseNode(RagNode):
    
    system_prompt = """You are an Higly intelligent and carismatic AI, you should respond presicely but still casual to the users prompt."""
    model_name = DEFAULT_MODEL

    
    def create(self, system_prompt=None, **kwargs):
        self.system_prompt = system_prompt
        self.model_name = kwargs.get("model", self.model_name)
        self.model = get_model(self.model_name)

    
    def run(
            self,
            context: NodeContext
        ):

        messages = [{
            "role": "system",
            "content": self.system_prompt
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

        client = get_client_for_model(self.model.model) # TODO effienctly lookup
        response = client.chat.completions.create(
            **completion_params
        )
        print("Running CasualResponseNode")
        
        res = response.choices[0].message.content
        
        return RagNodeResult(
            node_name=self.name,
            forward=True,
            response=res
        )