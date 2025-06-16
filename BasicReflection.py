from  langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage
from langgraph.graph import END,MessageGraph
from dotenv import load_dotenv
load_dotenv()




generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)



model=ChatGoogleGenerativeAI(model="gemini-2.0-flash")

generate_chain=generation_prompt|model
reflection_chain=reflection_prompt|model



graph=MessageGraph()


GENERATE="generate_node"
REFLECTION="reflect_node"

def generatenode(state):
    return  generate_chain.invoke({
        "messages":state
    })

def reflectionnode(messages):
    response =reflection_chain.invoke({"messages":messages})
    return [HumanMessage(content=response.content)]

def condition_node(state):
    if(len(state)>4):
        return END
    else:
        return REFLECTION
    




graph.add_node(GENERATE,generatenode)
graph.add_node(REFLECTION,reflectionnode)

graph.set_entry_point(GENERATE)
graph.add_conditional_edges(GENERATE,condition_node)


graph.add_edge(REFLECTION,GENERATE)



app=graph.compile()

res=app.invoke(HumanMessage(content="AI Agents taking over content creation"))
