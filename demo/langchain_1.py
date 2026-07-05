from langchain_classic.chains.question_answering.map_reduce_prompt import messages
from langchain_core.language_models import FakeListLLM
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from config.config import Config

config = Config()

from langchain_openai import OpenAI
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI

# gemini_pro = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
# messages = [
#     SystemMessage(content="You;re helpful programming assistant."),
#     HumanMessage(content="Write python code to calculate factorial"),]
#
# response = gemini_pro.invoke(messages)
#
# print(response.content)

# fake_llm = FakeListLLM(responses=["hello"])
# result = fake_llm.invoke("Any input will return Hello")
# print(result)

# from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
#
# template = ChatPromptTemplate.from_messages(
#     [
#         ("system","You are experienced programmer and mathematical analyst."),
#         ("user","{problem}")
#     ]
# )
#
# chat = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash-lite",
#     max_tokens=2048,
#     temperature=0.8,
#     top_p=0.95,
#     top_k=10,
#     thinking_budget=512
# )
#
# problem = """
# 정렬되지 않은 배열에서 k번째로 큰 원소를 찾는 알고리즘을 설계하라.
# 가능한 최적의 시간 복잡도를 갖도록하며, 알고리즘의 시간 및 공간 복잡도를 분석하고 왜 그것이 최적인지 설명하라.
# """
#
# chain = template | chat
# response = chain.invoke({"problem":problem})
# print(response.content)

#
# question_template = PromptTemplate.from_template("Answer this question : {question}")
# prompt_text = question_template.format(question="What is the capital of france?")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    max_tokens=2048,
    temperature=0.8,
    top_p=0.95,
    top_k=10,
    thinking_budget=512
)

story_prompt = PromptTemplate.from_template("Write a short story about {topic}")
story_chain = story_prompt | llm | StrOutputParser()

analysis_prompt = PromptTemplate.from_template("Analyze the following story's mood in korean:\n{story}")
analysis_chain = analysis_prompt | llm | StrOutputParser()


# 이렇게 하면 최종 분석 결과만 나오고 원본 스토리 사라짐
story_with_analysis = story_chain | analysis_chain
# story_analysis = story_with_analysis.invoke({"topic" : "a rainy day"})
# print(story_analysis)

from langchain_core.runnables import RunnablePassthrough
'''
RunnablePassthrough를 통해 그대로 다음으로 넘어간 topic은 assign을 통해 story_chain을 거치고나서 
그 결과가 story키에 저장된 dict 구조가 됨 즉 {"topic","story"} 이렇게 두 개의 키를 갖게됨
그리고 analysis_chain에 그 중 story의 값만 빼서 invoke 시키면 analysis에 그 결과가 저장됨
'''

enhanced_chain = (RunnablePassthrough
                  .assign(story=story_chain)
                  .assign(analysis=lambda x: analysis_chain.invoke(x["story"])))
result = enhanced_chain.invoke({"topic":"a rainy day"})
print(result.content)

from operator import itemgetter

'''
기본적으로 딕셔너리는 병렬처리로 {"topic" : "rain"} 이라는 데이터가 chain에 inovke되면
RunnablePassthrough()는 그대로 뒤로 흘려버리고 {"story", "topic"} 두 갈래길로 복사해서 같이 흘러감
story_chain은 topic을 쓰는 부분이 있으므로 딕셔너리의 topic 값이 거기에 채워지고, itemgetter로 추출된 topic은
"topic"에 저장됨, (itemgetter 대신 람다를 써도 동일 lambda x : x["topic"]) 

'''

manual_chain = (
    RunnablePassthrough() |
    {
        "story": story_chain,
        "topic": itemgetter("topic"),
    }
    | RunnablePassthrough.assign(
        analysis = analysis_chain
    )
)

result = manual_chain.inovke({"topic":"a rainy day"})


simple_dict_chain = story_chain | {
    "story" : RunnablePassthrough(),
    "analysis" : analysis_chain
}

result = simple_dict_chain.inovke({"topic":"a rainy day"})



