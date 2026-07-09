from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph

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

# 맵-리듀스
from langgraph.constants import Send
import operator
from typing_extensions import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END

reduce_prompt = ChatPromptTemplate.from_template("""
다음은 영상을 일정 시간 단위로 요약한 내용입니다.

{summaries}

이 내용을 하나의 자연스러운 전체 영상 요약으로 만들어주세요.
""")


def _merge_summaries( summaries: list[str], interval_secs: int,):
    merged = []

    for index, summary in enumerate(summaries):
        start = index * interval_secs
        end = (index + 1) * interval_secs

        merged.append(
            f"""
            [{start}초 ~ {end}초]
            
            {summary}
            """
        )

    return "\n".join(merged)

class AgentState(TypedDict):
    video_url: str
    chunks: int
    interval_secs: int
    summaries: Annotated[list, operator.add] # 타입은 list이고 다른 값이 들어오면 덮어쓰지않고 add함
    final_summary: str

class _ChunkState(TypedDict):
    video_url: str
    start_offset: int
    interval_secs: int

human_part = {"type" : "text", "text" : "Provide a summary of the video."}

async def _summarize_video_chunk(state: _ChunkState):
    start_offset = state["start_offset"]
    interval_secs = state["interval_secs"]
    video_part = {
        "type" : "media", "file_uri" : state["video_url"], "mime_type" : "video/mp4",
        "video_metadata" : {
            "start_offset" : start_offset * interval_secs,
            "end_offset" : (start_offset+1) * interval_secs,
        }
    }
    response = await llm.ainvoke(
        [HumanMessage(content=[human_part, video_part])]
    )
    return {"summaries" : [response.content]}

async def _generate_final_summary(state: AgentState):
    summary = _merge_summaries(
        summaries=state["summaries"],
        interval_secs=state["interval_secs"]
    )
    final_summary = await (reduce_prompt | llm | StrOutputParser()).ainvoke({"summaries" : summary})
    return {"final_summary" : final_summary}

def _map_summaries(state: AgentState):
    chunks = state["chunks"]
    payloads = [{
        "video_url" : state["video_url"],
        "interval_secs": state["interval_secs"],
        "start_offset" : i
    } for i in range(state["chunks"])]
    return [Send("summarize_video_chunk", payload) for payload in payloads]
# Send는 노드를 호출하는 langgraph API이고 langgraph에서는 직접 노드를 호출해서는 안됨
# State 관리, 체크포인트 등을 모두 langgraph가 관리하기 때문에 Send를 통해서 해야함

graph = StateGraph(AgentState)
graph.add_node("summarize_video_chunk", _summarize_video_chunk)
graph.add_node("generate_final_summary", _generate_final_summary)

graph.add_conditional_edges(START, _map_summaries, ["summarize_video_chunk"])
'''
add_conditional_edges(start, condition, [path_map]) 형태
_map_summaries는 Send를 생성하는 노드이고 video_chunk를 알기 전까지 몇개를 만들 지(몇개의 노드 연결이 생길지) 정해지지 않음
일반 add_edge는 정적으로 고정해버리는 것이므로 사용할 수 없음
만약 _map_summaries에서 Send하는 대상이 _generate_final_summary도 있었다면 path_map에 같이 추가되어야함
'''
graph.add_edge("summarize_video_chunk", "generate_final_summary") # _generate_final_summary는 한번만 실행
graph.add_edge("generate_final_summary", END)
app = graph.compile()

result = await app.ainvoke(
    {
        "video_uri" : video_url,
        "chunks" : 5,
        "interval_secs" : 600
    },
    {"max_concurrency": 3}
)["final_summary"]
