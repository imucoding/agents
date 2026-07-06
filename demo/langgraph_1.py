import io

from langchain_core.prompts import PromptTemplate
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

# from typing import Literal
#
# def is_suitable_condition(state) -> Literal["generate_application", END]:
#     if state["is_suitable"]:
#         return "generate_application"
#     return END
#
# builder = StateGraph(JobApplicationState)
# builder.add_node("analyze_job_description", analyze_job_description)
# builder.add_node("generate_application", generate_application)
#
# builder.add_edge(START, "analyze_job_description")
# builder.add_conditional_edges("analyze_job_description", is_suitable_condition)
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
# image.show()
#
# res = graph.invoke({"job_description":"some fake jd"})
# print(res)
#
#


from enum import Enum
from langchain_classic.output_parsers import EnumOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class IsSuitableJobEnum(Enum):
   YES = "YES"
   NO = "NO"

parser = EnumOutputParser(enum=IsSuitableJobEnum)

# assert parser.invoke("YES") == IsSuitableJobEnum.YES
# assert parser.invoke("NO") == IsSuitableJobEnum.NO
# assert parser.invoke("YES\n") == IsSuitableJobEnum.YES
# assert parser.invoke(HumanMessage(content="YES")) == IsSuitableJobEnum.YES

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gemma4:e4b",
    temperature=0
)

job_description: str = """
백엔드 개발 경력 7년 이상의  자바 소프트웨어 개발자 모집합니다.
"""

prompt_template = PromptTemplate.from_template(
    "Given a job description. decide whether it suits a junior java developer.\n"
    "Job Description : {job_description}\n"
    "Answer only YES or NO."
)

# chain = prompt_template | llm | parser
# result = chain.invoke({"job_description" : job_description})
# print(result)

class JobApplicationState(TypedDict):
    job_description: str
    is_suitable: IsSuitableJobEnum
    application: str

analyze_chain = llm | parser

def analyze_job_description(state):
    prompt = prompt_template.format(job_description=state["job_description"])
    result = analyze_chain.invoke(prompt)
    return {"is_suitable" : result}

def is_suitable_condition(state):
    return state["is_suitable"] == IsSuitableJobEnum.YES

def generate_application(state):
    print("Generating application...")
    return {"application" : "some fake application", "actions":["action2"]}

builder = StateGraph(JobApplicationState)
builder.add_node("analyze_job_description", analyze_job_description)
builder.add_node("generate_application", generate_application)

builder.add_edge(START, "analyze_job_description")
builder.add_conditional_edges("analyze_job_description", is_suitable_condition, {True : "generate_application", False : END})
builder.add_edge("generate_application", END)
graph = builder.compile()


from IPython.display import display, Image
from PIL import Image


png = graph.get_graph().draw_mermaid_png()
image = Image.open(io.BytesIO(png))
image.show()
