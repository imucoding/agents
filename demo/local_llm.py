from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# local_llm = ChatOllama(
#     model = "gemma4:e4b",
#     temperature=0,
#     max_tokens=512
# )
#
# prompt = PromptTemplate.from_template(
#     "explain {concept} in simple terms"
# )
#
# chain = prompt | local_llm | StrOutputParser()
#
# result = chain.invoke({"concept":"quantum computing"})
# print(result)

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

'''
기존 허깅페이스의 transformers 라이브러리에서 제공되던 pipeline을 그대로 가져옴
기존 라이브러리는 대화(chat) 개념이 없으므로 ChatHuggingFace와 분리함

'''
# llm = HuggingFacePipeline.from_model_id(
#     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#     pipeline_kwargs=dict(
#         max_new_tokens=512,
#         do_sample=False,
#         repetition_penalty=1.03
#     )
# )
#
# chat_model = ChatHuggingFace(llm=llm)
# messages = [
#     SystemMessage(content="you are helpful assistant"),
#     HumanMessage(content="Explain the concept of machine learning")
# ]
#
# result = chat_model.invoke(messages)
# print(result)

#
# local_llm = ChatOllama(
#     model = "gemma4:e4b",
#     temperature=0,
#     top_p=0.95
# )
#
# prompt = PromptTemplate.from_template(
#     "explain {concept} in simple terms"
# )
#
# chain = prompt | local_llm | StrOutputParser()
#
# for chunk in chain.stream({"concept":"quantum computing"}):
#     print(chunk, end="", flush=True)

# import time
#
# def safe_model_call(llm, prompt, max_retries=2):
#     retries = 0
#     while retries < max_retries:
#         try:
#             return llm.invoke(prompt)
#         except RuntimeError as e:
#             if "CUDA out of memory" in str(e):
#                 time.sleep(1)
#                 retries += 1
#             else:
#                 return "Error!"
#     return "Model not unavailable!"
#
# from langchain_core.runnables import RunnableLambda
# safe_llm = RunnableLambda(lambda x : safe_model_call(local_llm, x))
# safe_chain = prompt | safe_llm


import base64

import base64

# 1. 로컬 이미지를 base64 인코딩 문자열로 변환하는 헬퍼 함수
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 2. 로컬 Ollama의 Gemma 멀티모달 모델 초기화
# (세부 파라미터는 안정적으로 model_kwargs에 전달)
local_llm = ChatOllama(
    model="gemma4:e4b",
    temperature=0
)

# 3. 이미지 준비 및 인코딩 (같은 디렉토리에 샘플 이미지 파일이 있어야 합니다)
image_path = "moon.jpg"
base64_str = encode_image_to_base64(image_path)

# 4. LangChain 표준 규격에 맞춘 멀티모달 메시지 생성
# 객체 내부의 'type', 'text', 'image_url' 같은 핵심 키 규격을 맞춰줍니다.
multimodal_message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "이 이미지에 무엇이 담겨 있는지 한글로 설명해줘."
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpg;base64,{base64_str}"
            }
        }
    ]
)

print("🔮 Gemma가 이미지를 분석하는 중입니다 (Streaming)...\n")

# 5. 스트리밍으로 결과 출력
for chunk in local_llm.stream([multimodal_message]):
    print(chunk.content, end="", flush=True)
print("\n\n✅ 분석 완료.")