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

# CoT 프롬프트
from langsmith import Client
from langchain_core.output_parsers import StrOutputParser

client = Client()
math_cot_prompt = client.pull_prompt("arietem/math_cot", dangerously_pull_public_prompt=True,)
cot_chain = math_cot_prompt | llm | StrOutputParser()

# print(math_cot_prompt)
# print(cot_chain.invoke({"question" : "Solve equation 2*x+5=15"}))


# CoT를 통해 긴 사고 과정으로 답을 만들게 하고 그 이후 단계의 프롬프팅을 통해 답만 추출
from operator import itemgetter

parse_prompt = (
    "Given the initial question and a full answer,"
    "extract the concise answer. Do not assume anything end "
    "only use a provided full answer. \n\n Question : \n {question}\n"
    "FULL ANSWER : \n{full_answer}\n\n CONCISE ANSWER:\n"
)

parse_prompt_template = PromptTemplate.from_template(parse_prompt)

final_chain = (
    {"full_answer" : itemgetter("question") | cot_chain,
     "question" : itemgetter("question")}
    | parse_prompt_template
    | llm
    | StrOutputParser()
)

# print(final_chain.invoke(({"question" : "Solve equation 2*x+5=15"})))


# 자기 일관성
'''
온도를 높여서 더 다양한 출력이 나오게하고 여러번 수행하여 그중 가장 빈번한것을 선택
'''
generations = []
for _ in range(20):
    generations.append(final_chain.invoke({"question" : "Solve equation 2*x+5=15"}, temperature=2.0).strip())

from collections import Counter
print(Counter(generations).most_common(1)[0][0])

