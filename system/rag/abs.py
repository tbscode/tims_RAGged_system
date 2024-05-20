import concurrent.futures
from typing import List
from dataclasses import dataclass, field
import time

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-70B-Instruct"

def timed(func):
    def _w(*a, **k):
        then = time.time()
        res = func(*a, **k)
        elapsed = time.time() - then
        return elapsed, res
    return _w

class RagNode:
    # some init params & a self.run(prompt, context) method
    name: str = None
    start_node: bool = False
    end_node: bool = False

    def __init__(
            self,
            name: str,
            start_node: bool = False,
            end_node: bool = False,
            **kwargs
        ):
        self.name = name
        self.start_node = start_node
        self.end_node = end_node
        self.create(**kwargs)
        
    def __repr__(self) -> str:
        return f"RagNode({self.name})" + ("[start]" if self.start_node else "[end]" if self.end_node else "")
        
    def create(self, *args, **kwargs):
        pass
    
@dataclass
class YieldMessage:
    kind: str
    content: str

class NodeContext:
    message_history: List[dict] = []
    parent_results: dict = {}
    all_results: dict = {}
    prompt: str = ""
    
    def update_all_results(self, res=None):
        if self.all_results is None:
            self.all_results = self.parent_results
        else:
            self.all_results = {
                **self.all_results,
                **self.parent_results
            }
            
        if res is not None:
            self.parent_results = res
            print("*** New parent results:", self.parent_results)
    
    def __init__(self, message_history, prompt):
        self.message_history = message_history
        self.prompt = prompt
        
    def to_dict(self):
        return {
            "message_history": self.message_history,
            "parent_results": self.parent_results,
            "prompt": self.prompt
        }
        
    def copy(self):
        return NodeContext(**self.to_dict())

class RagEdge:
    # connect two RagNodes
    start: str = None
    end: str = None
    disabled: bool = False
    update_overwrite: callable = None
    
    def __init__(
            self, 
            start: RagNode, 
            end: RagNode,
            **kwargs
        ):
        self.start = start
        self.end = end
        self.update_overwrite = kwargs.get("update_overwrite", None)
        
    def __repr__(self) -> str:
        return f"RagEdge({self.start} -> {self.end})" + ("[disabled]" if self.disabled else "")
        
    def update_state(self, context):
        print(f"*** Updating edge {self}")
        if self.update_overwrite is not None:
            self.update_overwrite(self, context)

class RagGraph:
    
    yield_messages: List[YieldMessage] = []
    
    def __init__(
            self, 
            nodes: List[RagNode],
            edges: List[RagEdge]
        ):
        self.nodes = nodes
        self.edges = edges
        
    def get_start_node(self):
        for node in self.nodes:
            if node.start_node:
                return node
        return None
    
    def get_node(self, name):
        for node in self.nodes:
            if node.name == name:
                return node
        return None
    
    def get_sibling_nodes(self, node):
        siblings = []
        for edge in self.edges:
            if (not edge.disabled) and (edge.start == node.name):
                siblings.append(self.get_node(edge.end))
        return siblings
    
    def update_edges(self, start_node, context):
        for edge in self.edges:
            if edge.start == start_node.name:
                edge.update_state(context)
    
    def run_nodes(self, nodes, context):
        node_res = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as executor:
            futures = [executor.submit(timed(node.run), context) for node in nodes]

            results = [future.result() for future in futures]
            
            for result in results:
                elapsed, res = result
                format_time = "{:.2f}".format(elapsed)
                print("Elapsed:", format_time, "Result:", res.response)
                node_res[res.node_name] = res
        return node_res
    
    def check_for_end_node(self, nodes):
        for node in nodes:
            if node.end_node:
                return node
        return None
    
    def run_subgraph2(self, 
            nodes: List[RagNode],
            context: NodeContext
        ):
        print("*** Running subgraph ***")
        siblings = []
        for node in nodes:
            self.update_edges(node, context)
            sibls = self.get_sibling_nodes(node)
            for s in sibls:
                if s not in siblings:
                    siblings.append(s)
                    
        print("*** Siblings:", siblings)
        
        if len(siblings) == 0:
            print("No futher siblings to traverse")
            return context
        
        res = self.run_nodes(siblings, context)
        context.update_all_results(res)
        
        next_nodes = []
        for node_name, node_res in res.items():
            print(f"===> Node: {node_name}, response: {node_res.response}")
            if len(node_res.yield_messages) > 0:
                for msg in node_res.yield_messages:
                    print(f"=====> Yielded message: {msg.content}")
                self.yield_messages.extend(node_res.yield_messages)
            if node_res.forward:
                node = self.get_node(node_name)
                
                if node not in next_nodes:
                    next_nodes.append(node)
        
        print("*** Next nodes:", next_nodes)
        end_node = self.check_for_end_node(next_nodes)
        if end_node is not None:
            print("End node found:", end_node)
            return context
        return self.run_subgraph2(next_nodes, context)
    
    def get_final_result(self, context):
        end_node_names = list(context.parent_results.keys())
        assert len(end_node_names) == 1, "Multiple end nodes found."
        end_node_name = end_node_names[0]
        return context.parent_results[end_node_name].response

    def run(
            self,
            context: NodeContext,
        ):
        start_node = self.get_start_node()
        if start_node is None:
            raise Exception("No start node found.")
        
        print("Start node:", start_node)
        context = self.run_subgraph2([start_node], context)
        
        end_result = self.get_final_result(context)
        print("Yield messages:", self.yield_messages)
        print("Final results:", end_result)
        
@dataclass
class RagNodeResult:
    node_name: str
    meta: dict = field(default_factory=dict)
    yield_messages: List[YieldMessage] = field(default_factory=list)
    response: dict = None
    forward: bool = True
    
    def to_dict(self):
        return {
            "node_name": self.node_name,
            "forward": self.forward,
            "response": self.response,
            "meta": self.meta
        }