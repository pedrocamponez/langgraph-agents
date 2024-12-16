from typing import List, Sequence
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
# MessageGraph is a StateGraph where everynode receives a list of messages as input
# and returns one or more messages as output.
from chains import generate_chain, reflect_chain

REFLECT = "reflect"
GENERATE = "generate"


# The state is simply a sequence of messages, a list of messages, and the Node is going to run the
# generation chain invoking all the state that we already have (wether its the first, second or third run, e.g)
def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})


# Very similarly with the generation node, but the response we get back from the LLM, which would be role to AI,
# we now change it to be Human content instead of AI. We do that because we want to trick the LLM to think
# that the Human is sending this message, so it keep talking to itself. It's a very important technique
# and its basically mimicking a conversation between a Human and the LLM.
def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]


builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())


if __name__ == '__main__':
    print("Hello LangGraph!")
    inputs = HumanMessage(content="""Make this tweet better:"
                                    @LangChainAI
                                    - newly Tool Calling feature is seriously underrated.
                                    After a long wait, it's here - making the implementation of agents across different
                                    models with function calling.
                                    Made a video covering their newest blog post.
                """)
    response = graph.invoke(inputs)