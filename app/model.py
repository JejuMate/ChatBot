import os
import json
import time
import requests
import operator
import uuid
from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv

# --- LangChain / LangGraph ---
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool

# --- Folium (지도 생성용) ---
import folium
from folium.plugins import AntPath

# --- Weaviate (v4) ---
import weaviate
from weaviate.classes.init import Auth

# --- 내부 스키마 ---
from .schema import AgentResponse, TravelConstraints, SimpleTravelPlan, SimpleDailyPlan

load_dotenv()

# =========================
# 1. 환경 설정 & API 키
# =========================
TMAP_APP_KEY = os.getenv("TMAP_API_KEY", "")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
AMADEUS_CLIENT_ID = os.getenv("AMADEUS_CLIENT_ID", "")
AMADEUS_SECRET = os.getenv("AMADEUS_SECRET", "")

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# Seed 추가하여 일관된 결과 유도
llm = ChatOpenAI(model="gpt-4o", temperature=0, model_kwargs={"seed": 42})

# =========================
# 2. Weaviate 연결 (사용자 환경)
# =========================
weaviate_client = None
jeju_collection = None

def init_weaviate_connection():
    global weaviate_client, jeju_collection
    try:
        weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
            headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
            skip_init_checks=True 
        )
        jeju_collection = weaviate_client.collections.get("JejuSpot")
        print("✅ Weaviate 'JejuSpot' Collection 연결 성공.")
    except Exception as e:
        print(f"❌ Weaviate 연결 실패: {e}")

init_weaviate_connection()

# =========================
# 3. 헬퍼 함수들 (Amadeus & Tmap Logic)
# =========================
_amadeus_token = None
_amadeus_token_expiry = 0
IATA_CODES = {"제주": "CJU", "서울": "SEL", "김포": "GMP", "인천": "ICN", "부산": "PUS", "대구": "TAE", "광주": "KWJ", "청주": "CJJ"}

def _get_amadeus_token():
    global _amadeus_token, _amadeus_token_expiry
    if _amadeus_token and time.time() < _amadeus_token_expiry: return _amadeus_token
    if not AMADEUS_CLIENT_ID: return None
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    try:
        r = requests.post(url, data={"grant_type": "client_credentials", "client_id": AMADEUS_CLIENT_ID, "client_secret": AMADEUS_SECRET})
        _amadeus_token = r.json()["access_token"]
        _amadeus_token_expiry = time.time() + r.json()["expires_in"] - 60
        return _amadeus_token
    except: return None

def _tmap_route_sequential(start_lat, start_lon, end_lat, end_lon, via_points=None):
    """(내부용) Tmap 다중 경유지 경로 계산"""
    url = "https://apis.openapi.sk.com/tmap/routes/routeSequential30?version=1&format=json"
    headers = {"appKey": TMAP_APP_KEY, "Content-Type": "application/json"}
    body = {
        "startName": "Start", "startX": str(start_lon), "startY": str(start_lat),
        "endName": "End", "endX": str(end_lon), "endY": str(end_lat),
        "reqCoordType": "WGS84GEO", "resCoordType": "WGS84GEO", "searchOption": 0
    }
    if via_points:
        body["viaPoints"] = [{"viaPointId": f"v{i}", "viaPointName": "Via", "viaX": str(vp[1]), "viaY": str(vp[0])} for i, vp in enumerate(via_points)]
    
    r = requests.post(url, headers=headers, data=json.dumps(body), timeout=10)
    return r.json()

# =========================
# 4. 도구 (Tools) 정의 - 팀원분 Tool 전체 포함
# =========================

@tool
def search_places(query: str) -> str:
    """Weaviate DB에서 장소 검색 (팀원의 search_jeju_tour_spots_semantic 대체)"""
    if not jeju_collection: return "DB 연결 실패"
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
    except Exception as e: return f"검색 오류: {e}"

@tool
def get_web_info(query: str) -> str:
    """웹 검색 (Tavily)"""
    return TavilySearchResults(k=2, tavily_api_key=TAVILY_API_KEY).invoke(query)

@tool
def get_weather_forecast(location: str, date: str) -> str:
    """(팀원 Tool) 날짜별 날씨 예보 조회 (Open-Meteo)"""
    GEO = {"제주시": (33.4996,126.5312), "서귀포시": (33.2539,126.5596)}
    lat, lon = GEO.get(location[:3], (33.4996,126.5312))
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=weather_code,temperature_2m_max,temperature_2m_min&timezone=Asia%2FSeoul&start_date={date}&end_date={date}"
        r = requests.get(url, timeout=5)
        data = r.json()
        if "daily" in data:
            max_temp = data["daily"]["temperature_2m_max"][0]
            min_temp = data["daily"]["temperature_2m_min"][0]
            return f"{date} {location} 날씨: 최저 {min_temp}도 / 최고 {max_temp}도 (맑음/흐림)"
        return "날씨 정보 없음"
    except Exception as e: return f"날씨 조회 실패: {e}"

@tool
def _tmap_poi_coords(keyword: str) -> str:
    """(팀원 Tool) Tmap POI 검색 -> 좌표 반환"""
    if not TMAP_APP_KEY: return json.dumps({"error": "TMAP API KEY 없음"})
    url = "https://apis.openapi.sk.com/tmap/pois"
    try:
        r = requests.get(url, params={"version": 1, "searchKeyword": keyword, "count": 1, "resCoordType": "WGS84GEO", "appKey": TMAP_APP_KEY}, timeout=5)
        pois = r.json().get("searchPoiInfo", {}).get("pois", {}).get("poi", [])
        if pois:
            return json.dumps({
                "name": pois[0]["name"], 
                "lat": float(pois[0]["frontLat"]), 
                "lon": float(pois[0]["frontLon"]),
                "address": pois[0].get("roadName", pois[0].get("legalDong", ""))
            }, ensure_ascii=False)
        return json.dumps({"error": "POI 없음"})
    except Exception as e: return json.dumps({"error": str(e)})

@tool
def get_detailed_description(spot_name: str) -> str:
    """(팀원 Tool) 장소 상세 설명 (웹 검색 요약)"""
    info = TavilySearchResults(k=2, tavily_api_key=TAVILY_API_KEY).invoke(f"{spot_name} 상세 정보 특징")
    if not info: return json.dumps({"description": "정보 없음"})
    # 간단 요약 로직 (실제로는 LLM 호출해도 됨)
    return json.dumps({"description": str(info)[:200]}, ensure_ascii=False)

@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """(팀원 Tool) 항공권 조회 (Amadeus)"""
    token = _get_amadeus_token()
    if not token: return "항공권 조회 불가"
    origin_code = IATA_CODES.get(origin, "SEL")
    dest_code = IATA_CODES.get(destination, "CJU")
    try:
        r = requests.get("https://test.api.amadeus.com/v2/shopping/flight-offers", headers={"Authorization": f"Bearer {token}"}, 
                         params={"originLocationCode": origin_code, "destinationLocationCode": dest_code, "departureDate": date, "adults": 1, "max": 5, "currencyCode": "KRW"})
        data = r.json()
        if "data" not in data: return "항공권 정보 없음"
        flights = []
        for offer in data["data"]:
            price = offer["price"]["total"]
            seg = offer["itineraries"][0]["segments"]
            dep = seg[0]["departure"]["at"].split("T")[1][:5]
            arr = seg[-1]["arrival"]["at"].split("T")[1][:5]
            carrier = seg[0]["carrierCode"]
            flights.append(f"[{carrier}] {dep} -> {arr} ({price}원)")
        return "\n".join(flights)
    except Exception as e: return f"조회 에러: {e}"

@tool
def build_route_map(locations: List[str]) -> str:
    """(팀원 Tool) 경로 지도 생성 및 URL 반환"""
    if len(locations) < 2: return "장소가 2개 이상 필요합니다."
    coords = []
    for loc in locations:
        # _tmap_poi_coords 로직 재사용
        poi_json = _tmap_poi_coords.invoke(loc) 
        poi = json.loads(poi_json)
        if "lat" in poi: coords.append((poi["lat"], poi["lon"]))
    
    if len(coords) < 2: return "좌표 변환 실패"
    
    try:
        route_json = _tmap_route_sequential(coords[0][0], coords[0][1], coords[-1][0], coords[-1][1], via_points=coords[1:-1])
        m = folium.Map(location=[coords[0][0], coords[0][1]], zoom_start=10)
        
        # 경로 그리기
        path = []
        for f in route_json.get("features", []):
            if f["geometry"]["type"] == "LineString":
                path.extend([[y, x] for x, y in f["geometry"]["coordinates"]])
        AntPath(path).add_to(m)
        
        filename = f"route_{uuid.uuid4()}.html"
        m.save(os.path.join(STATIC_DIR, filename))
        return f"/static/{filename}"
    except Exception as e: return f"지도 생성 오류: {e}"

@tool
def generate_route_link(locations: List[str]) -> str:
    """(팀원 Tool) 구글 맵 경로 링크 생성"""
    return f"https://www.google.com/maps/dir/{'/'.join(locations)}"

@tool
def save_simple_plan(plan_json: str) -> str:
    """(팀원 Tool) 일정 상태 저장 (State Update)"""
    return "일정이 상태에 저장되었습니다."

@tool(args_schema=AgentResponse)
def submit_final_response(**kwargs) -> str:
    """(사용자 필수) 최종 답변 제출 (JSON 스키마 준수)"""
    return "답변 완료"

tools = [
    search_places, 
    get_web_info, 
    get_weather_forecast, 
    _tmap_poi_coords,
    get_detailed_description,
    search_flights,
    build_route_map, 
    generate_route_link,
    save_simple_plan,
    submit_final_response
]
tool_node = ToolNode(tools)

# =========================
# 5. LangGraph 설정
# =========================

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_id: Optional[int]
    action: Optional[str]
    constraints: Optional[dict]
    travel_plan: Optional[dict] # 팀원 로직 (일정 저장)
    visited_places: Annotated[List[str], operator.add] # 중복 방지

def call_model(state: AgentState):
    messages = state["messages"]
    constraints = state.get("constraints", {})
    current_plan = state.get("travel_plan")
    visited_places = state.get("visited_places", [])
    
    # 1. 여행 조건
    constraint_desc = "없음"
    if constraints:
        c = constraints if isinstance(constraints, dict) else constraints.dict()
        constraint_desc = f"""
        - 기간: {c.get('start_date')} ~ {c.get('end_date')}
        - 스타일: {c.get('travel_style')}
        - 동반자: {c.get('companions')}
        - 연령대: {c.get('age_group')}
        - 추가 요청: {c.get('additional_request', '없음')}
        """

    # 2. 컨텍스트 (현재 일정, 중복 방지)
    context_txt = ""
    if current_plan:
        context_txt += f"\n[현재 작성 중인 일정]\n{json.dumps(current_plan, ensure_ascii=False)}"
    if visited_places:
        context_txt += f"\n[제외할 장소(중복 방지)]\n{', '.join(visited_places)}"

    # 3. 시스템 프롬프트 (팀원 로직 + 사용자 스키마)
    SYSTEM_PROMPT = f"""
    당신은 제주도 여행 전문가 AI입니다.
    사용자의 요청을 분석하여, 반드시 **`submit_final_response` 도구**를 호출하여 JSON 형태로 응답해야 합니다.

    [역할]
    사용자가 원하는 여행 스타일에 맞춰 일정을 계획하고, DB(`search_places`)에서 정확한 장소 정보를 찾아 제공합니다.
    
    [현재 여행 조건]
    {constraint_desc}
    {context_txt}

    [작업 순서 (Step-by-Step)]
    1. 여행 기간에 맞는 **날씨를 확인**하세요 (`get_weather_forecast`).
    2. 조건에 맞는 **장소를 검색**하세요 (`search_places`). (중복된 장소 제외)
    3. 각 장소의 **정확한 좌표(`_tmap_poi_coords`)**와 **상세 설명(`get_detailed_description`)**을 수집하세요.
    4. **항공권** 요청이 있다면 조회하세요 (`search_flights`).
    5. `save_simple_plan`으로 일정을 중간 저장하세요.
    6. 최종적으로 `submit_final_response`를 호출하세요.

    [필수 데이터 규칙]
    1. **모든 일정 관련 요청 시 (`create_schedule`, `update_schedule`, `remove_place` 등)**:
       - `start_date`와 `end_date` 필드에 현재 '여행 조건'의 기간을 **항상 채워 넣으세요.** (누락 금지)

    2. 🚨 **[장소 선정 엄격 규칙 - 중요]**
       - **'자유 시간', '호텔 휴식', '이동', '저녁 식사' 같은 추상적인 일정을 절대 넣지 마세요.**
       - 모든 슬롯(`morning`, `afternoon`, `evening`)은 반드시 **`search_places`로 검색된 실존하는 구체적 장소명(관광지, 카페, 식당 등)**이어야 합니다.
       - 만약 저녁에 갈 곳이 마땅치 않다면 '야시장', '천문대', '심야 카페', '야간 개장 관광지' 등을 검색해서 채우세요.

    3. **`create_schedule` 요청 시**:
       - 🚨 **[전체 기간 생성]**: 시작일부터 종료일까지 모든 날짜의 일정을 채우세요. (예: 2박 3일이면 Day 1, 2, 3 모두 필수)
       - **3 Slot**: 매일 Morning, Afternoon, Evening을 채우세요.
       - **Date 필드**: `schedule` 아이템에 `date`(YYYY-MM-DD)를 반드시 계산해서 넣으세요.   
    
    4. **중복 방지**:
       - 전체 일정 내에서 동일한 장소가 2번 이상 나오지 않게 하세요.

    [Action 유형 정의]
    - `create_schedule`: 전체 일정 생성
    - `suggest_alternative`: 특정 일정에 대한 대안 제시 (target, alternative_places 필드 사용)
    - `update_schedule`: 일정 변경 확정 (target, new_place 필드 사용)
    - `remove_place`: 일정 삭제 (target 필드 사용)
    - `chat`: 일반 대화

    """
    
    full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    model = llm.bind_tools(tools)
    response = model.invoke(full_messages)
    return {"messages": [response]}

# State Update Node
def update_plan_from_tool(state: AgentState):
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage) and last_message.name == "save_simple_plan":
        # 여기에 실제 파싱 로직 추가 가능
        pass
    return state

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "continue"
    return "end"

# Graph Construction
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_node("update_plan", update_plan_from_tool)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
workflow.add_edge("action", "update_plan")
workflow.add_edge("update_plan", "agent")

memory = MemorySaver()
agent_app = workflow.compile(checkpointer=memory)
print("✅ Agent Compiled (All Tools Included + Weaviate + User Schema).")