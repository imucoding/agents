from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOllama(
    model="gemma4:e4b",
    temperature=0
)

prompt_template = (
    "Given a job description. decide whether it suits a junior java developer.\n"
    "Job Description : {job_description}\n"
    "Answer only YES or NO."
)

job_description: str = """
백엔드 개발 경력 7년 이상의  자바 소프트웨어 개발자 모집합니다.
"""

msg_template = HumanMessagePromptTemplate.from_template(prompt_template)
msg_example = msg_template.format(job_description="fake_jd")

chat_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant."),
    msg_template
])
chain = chat_prompt_template | llm | StrOutputParser()
# res = chain.invoke({"job_description": job_description})
# print(res)

# 플레이스홀더

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
placeholder_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant."),
        ("placeholder", "{history}"),
        ("human",prompt_template)
    ]
)

print(len(placeholder_prompt_template.invoke({
    "job_description": job_description,
    "history": [("human", "hi"), ("ai", "hi")]
}).to_messages()))

# 프롬프트 연결
system_template = PromptTemplate.from_template("a : {a} b : {b}")
system_template_part = system_template.partial(a="a")
print(system_template_part.invoke({"b":"b"}).text)

system_template_part_1 = PromptTemplate.from_template("a : {a}")
system_template_part_2 = PromptTemplate.from_template("b : {b}")
system_template_merged = system_template_part_1 + " " + system_template_part_2
print(system_template_merged.invoke({"a":"a", "b":"b"}).text)