import argparse
from rag.agents.hal9004_rag import get_graph
from rag.abs import NodeContext

graph_by_name = {
    "hal9004_rag": get_graph
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run the RAGged system')
    parser.add_argument("-p", type=str, help='The user prompt')
    parser.add_argument("-a", type=str, help='The agent to use (e.g. hal9004_rag)', default="hal9004_rag")
    args = parser.parse_args()
    

    graph = graph_by_name[args.a]()
    
    graph.run(
        context=NodeContext(
            message_history=[],
            prompt=args.p
        )
    )