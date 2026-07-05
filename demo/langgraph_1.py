import io

from typing_extensions import TypedDict

class JobApplicationState(TypedDict):
    job_description: str
    is_suitable: bool
    application: str

from langgraph.graph import StateGraph, START, END

def analyze_job_description(state):
    print("Analyzing job description...")
    return {"is_suitable" : len(state["job_description"]) > 100}

def generate_application(state):
    print("Generating application...")
    return {"application" : "some fake application"}
#
# builder = StateGraph(JobApplicationState)
# builder.add_node("analyze_job_description", analyze_job_description)
# builder.add_node("generate_application", generate_application)
#
# builder.add_edge(START, "analyze_job_description")
# builder.add_edge("analyze_job_description", "generate_application")
# builder.add_edge("generate_application", END)
#
# graph = builder.compile()
#
# from IPython.display import Image, display
# from PIL import Image
#
#
# png = graph.get_graph().draw_mermaid_png()
# image = Image.open(io.BytesIO(png))
# # image.show()
#
# res = graph.invoke({"job_description":"some fake jd"})
# print(res)

from typing import Literal

def is_suitable_condition(state) -> Literal["generate_application", END]:
    if state["is_suitable"]:
        return "generate_application"
    return END

builder = StateGraph(JobApplicationState)
builder.add_node("analyze_job_description", analyze_job_description)
builder.add_node("generate_application", generate_application)

builder.add_edge(START, "analyze_job_description")
builder.add_conditional_edges("analyze_job_description", is_suitable_condition)
builder.add_edge("generate_application", END)

graph = builder.compile()

from IPython.display import Image, display
from PIL import Image


png = graph.get_graph().draw_mermaid_png()
image = Image.open(io.BytesIO(png))
image.show()

res = graph.invoke({"job_description":"some fake jd"})
print(res)




