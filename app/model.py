# app/model.py
import os
from dotenv import load_dotenv
load_dotenv()

import json
import operator
from typing import TypedDict, Annotated, List, Optional

# --- LangChain / LangGraph ---
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool

# --- Weaviate (v4) ---
import weaviate
from weaviate.classes.init import Auth

# --- 내부 스키마 ---
from .schema import AgentResponse, PlaceDetail, ScheduleItem

# =========================
# 1. 환경 설정
# =========================================
TMAP_APP_KEY = os.getenv("TMAP_API_KEY", "")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# =========================
# 2. Weaviate 연결
# =========================
weaviate_client = None
jeju_collection = None

def init_weaviate_connection():
    global weaviate_client, jeju_collection
    try:
        weaviate_client = weaviate.connect_to_wcs(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
            headers={"X-OpenAI-Api-Key": OPENAI_API_KEY}
        )
        jeju_collection = weaviate_client.collections.get("JejuSpot")
        print("✅ Weaviate 'JejuSpot' Collection 연결 성공.")
    except Exception as e:
        print(f"❌ Weaviate 연결 실패: {e}")

init_weaviate_connection()

# =========================
# 3. 도구 (Tools) 정의
# =========================

@tool
def search_places(query: str) -> str:
    """
    Weaviate DB에서 장소를 검색합니다.
    쿼리 예시: '아이랑 가기 좋은 카페', '성산일출봉 근처 맛집', '비오는 날 갈만한 곳'
    """
    if not jeju_collection:
        return "DB 연결 실패"
    try:
        response = jeju_collection.query.near_text(query=query, limit=5)
        results = []
        for obj in response.objects:
            p = obj.properties
            results.append({
                "name": p.get("name"),
                "category": p.get("category"),
                "address": p.get("address"),
                "description": p.get("themeTags"),
                "latitude": p.get("yCoord"),
                "longitude": p.get("xCoord")
            })
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"검색 오류: {e}"

@tool
def get_web_info(query: str) -> str:
    """최신 정보(날씨, 휴무일 등)가 필요할 때 웹 검색을 사용합니다."""
    tool = TavilySearchResults(k=2, tavily_api_key=TAVILY_API_KEY)
    return tool.invoke(query)

# ⭐️ [핵심 변경] AgentResponse를 위한 '진짜 도구' 생성
# 이 함수가 실행되어야 히스토리에 "완료됨" 도장이 찍힙니다.
@tool(args_schema=AgentResponse)
def submit_final_response(**kwargs) -> str:
    """
    최종 답변을 사용자에게 전달할 때 사용하는 도구입니다.
    이 도구를 호출하면 답변 생성 과정이 완료됩니다.
    """
    return "답변이 성공적으로 생성되어 사용자에게 전달되었습니다."

# tools 리스트에 추가
tools = [search_places, get_web_info, submit_final_response]
tool_node = ToolNode(tools)

# =========================
# 4. 시스템 프롬프트
# =========================
SYSTEM_PROMPT = """
당신은 제주도 여행 전문가 AI입니다.
사용자의 요청을 분석하여, 반드시 **`submit_final_response` 도구**를 호출하여 응답해야 합니다.

[역할]
사용자가 원하는 여행 스타일에 맞춰 일정을 계획하고, DB(`search_places`)에서 정확한 장소 정보를 찾아 제공합니다.

**[날씨 반영 규칙]**
1. 일정을 계획하기 전에 `get_weather_forecast` 도구를 호출해보세요.
2. **비 예보가 있는 경우**: 박물관, 미술관, 예쁜 카페 등 **실내 관광지** 위주로 추천하세요.
3. **맑은 예보인 경우**: 오름, 해변, 테마파크 등 **실외 관광지**를 적극 추천하세요.
4. **예보 정보가 없는 경우(너무 먼 미래 등)**: 날씨 제약 없이 **사용자의 여행 스타일(힐링, 액티비티 등)에 가장 잘 맞는 최적의 장소**를 추천하세요. (맑음으로 가정)

[사용 가능한 Action 및 규칙]
(기존 규칙과 동일)
1. `create_schedule`
2. `suggest_alternative`
3. `update_schedule`
4. `remove_place`
5. `chat`

[필수 지침]
- 모든 장소 데이터(좌표 포함)는 추측하지 말고 `search_places` 도구를 사용해 DB에서 가져오세요.
- **최종 응답은 반드시 `submit_final_response` 도구를 호출하며 끝내야 합니다.** 텍스트로만 답하지 마세요.
"""

# =========================
# 5. LangGraph 정의
# =========================

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

def call_model(state: AgentState):
    messages = state["messages"]
    full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    # bind_tools에 우리가 만든 tools 리스트를 그대로 넘깁니다.
    model = llm.bind_tools(tools)
    response = model.invoke(full_messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    
    if last_message.tool_calls:
        # ⭐️ [핵심 변경] AgentResponse(submit_final_response)여도 'continue'를 반환하여
        # ToolNode가 실행되게 만듭니다. 그래야 히스토리가 완성됩니다.
        return "continue"
        
    return "end"

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
workflow.add_edge("action", "agent")

memory = MemorySaver()
agent_app = workflow.compile(checkpointer=memory)
print("\n✅ 제주도 챗봇 에이전트(Fixed Tool Loop) 컴파일 완료!")