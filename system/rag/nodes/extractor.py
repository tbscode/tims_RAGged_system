import json
from rag.abs import RagNode, DEFAULT_MODEL, NodeContext, RagNodeResult
from jsonschema import validate
from rag.models import get_model, get_client_for_model

class ParamExtractorNode(RagNode):
    base_prompt: str = """
You are a function parameter generating AI.
The User Intend AI has already identified the user intend as "{tool_name}".

So the user want to call the "{tool_name}" function, that that perform the following:
{tool_description}

The "{tool_name}" function has the following input schema:

{schema}

You should respond with a json object that looks e.g.: like this.

{schema_example}
  
Based on the field descriptions of the schema analyze the user input and generate the parameters.
""" 

    system_prompt = None
    schema = None
    schema_example = None
    tool_name = None
    tool_description = None
    model_name = DEFAULT_MODEL
    
    def create(
            self,
            schema: dict,
            schema_example: dict,
            tool_name: str,
            tool_description: str,
            **kwargs
        ):
        
        self.schema = schema
        self.schema_example = schema_example
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.model_name = kwargs.get("model", self.model_name)
        self.model = get_model(self.model_name)

        
    def run(
            self,
            context: NodeContext
        ):

        self.system_prompt = self.base_prompt.format(
            tool_name=context.prompt,
            schema=self.schema,
            tool_description=self.tool_description,
            schema_example=self.schema_example
        )
        
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

        parsable = False
        parsed = None
        res = response.choices[0].message.content
        try:
            parsed = json.loads(res)
            parsable = True
        except Exception as e:
            print("ERROR:" + str(e), self.model.model)
            
        valid = False
        if parsable:
            try:
                validate(parsed, self.schema)
                valid = True
            except Exception as e:
                print("ERROR:" + str(e), self.model.model)

        print("RES", res) 
        
        return RagNodeResult(
            node_name=self.name,
            forward=valid,
            response=parsed,
            meta={
                "valid": valid,
                "parsable": parsable,
                "parsed": parsed,
            },
        )