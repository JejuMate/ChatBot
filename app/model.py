import os
# ⬇️⬇️⬇️ [추가] .env 파일을 로드하기 위한 코드 ⬇️⬇️⬇️
from dotenv import load_dotenv
load_dotenv() # 이 함수가 .env 파일에서 환경변수를 읽어옵니다.

import json
import math
import operator
import requests
import pandas as pd
from urllib.parse import quote
from typing import TypedDict, Annotated, List, Optional
import uuid

# --- Colab '보안 비밀' ---
try:
    from google.colab import userdata
except Exception:
    userdata = None

# --- LangChain / LangGraph ---
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- Folium ---
import folium
from folium.plugins import AntPath

# --- 내부 스키마 임포트 ---
from .schema import (
    AgentResponse, PlaceDetail, TimeSlotScheduleItem, TargetPlace, # LLM이 출력할 JSON 스키마
    SimpleTravelPlan, SimpleDailyPlan # 백엔드 상태 저장용 스키마
)

# [추가] DB 폴더 경로 상수 (create_db.py와 동일해야 함)
DB_PERSIST_DIRECTORY = 'chroma_db_persistent'

# =========================
# 0) 상수 및 API 키 로딩
# =========================
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

TMAP_APP_KEY = ""
TAVILY_API_KEY = "" # ⬇️ [수정] TAVILY 키를 저장할 변수 추가
try:
    if userdata:
        os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
        os.environ["TAVILY_API_KEY"] = userdata.get('TAVILY_API_KEY')
        TMAP_APP_KEY = userdata.get('TMAP_API_KEY') or ""
    else:
        # .env 파일에서 로드되므로, os.getenv만 남깁니다.
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY', "")
        os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY', "") 
        TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', "") # ⬇️ [수정] 변수에 키 저장
        TMAP_APP_KEY = os.getenv("TMAP_API_KEY", "")
    
    # ⬇️ [수정] TAVILY_API_KEY 변수를 직접 체크하도록 수정
    if not os.getenv('OPENAI_API_KEY') or not TAVILY_API_KEY or not TMAP_APP_KEY:
        print("경고: .env 파일에서 API 키를 찾을 수 없습니다. 키가 로드되지 않았습니다.")
    else:
        print("API 키 로딩 완료 (OpenAI, Tavily, Tmap).")
        
except Exception as e:
    print(f"API 키 로딩 중 오류 발생: {e}")

llm = ChatOpenAI(model="gpt-4o", temperature=0)


# =========================
# 2) RAG용 Chroma 로드
# =========================
def load_persistent_db():
    """ (수정) 매번 임베딩하는 대신, 이미 구축된 DB를 로드합니다. """
    
    # project/chroma_db_persistent 경로를 찾기 위해 (main.py 기준)
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', DB_PERSIST_DIRECTORY)

    if not os.path.exists(db_path):
        print(f"---" * 10)
        print(f"오류: '{db_path}' 폴더를 찾을 수 없습니다.")
        print("DB를 먼저 구축해야 합니다. (Colab으로 생성 후 'chroma_db_persistent' 폴더에 압축 해제)")
        print(f"---" * 10)
        return None, pd.DataFrame() # DB 로드 실패

    try:
        print("임베딩 모델 로드 중... (DB 로드에 필요)")
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        print(f"'{db_path}'에서 기존 DB 로드 중... (매우 빠름)")
        
        # [핵심] Chroma.from_texts(...) 대신, Chroma() 생성자를 사용해 로드
        vector_db = Chroma(
            persist_directory=db_path,
            embedding_function=embedding_function
        )
        print("ChromaDB 로드 완료.")

        # RAG의 df_jeju도 필요하므로, CSV를 다시 로드 (임베딩보다는 훨씬 가벼움)
        print("RAG용 DataFrame 로드 중...")
        
        # ⬇️⬇️⬇️ [확인] 사용자님 CSV 경로 ⬇️⬇️⬇️
        df_category = pd.read_csv("C:/Users/Hoon/Desktop/캡스톤 디자인_인공지능/Jeju_Chatbot/data/V3_관광지카테고리_End.csv")
        df_visit    = pd.read_csv("C:/Users/Hoon/Desktop/캡스톤 디자인_인공지능/Jeju_Chatbot/data/visit_jeju.csv")
        
        df_merged = pd.merge(df_category, df_visit, left_on='VISIT_AREA_NM', right_on='이름', how='left')
        df_merged = df_merged.drop_duplicates(subset=['VISIT_AREA_NM']).reset_index(drop=True)
        df_merged['테마']         = df_merged['테마'].fillna('')
        df_merged['ROAD_NM_ADDR'] = df_merged['ROAD_NM_ADDR'].fillna('주소 정보 없음')
        df_merged['소분류']       = df_merged['소분류'].fillna('기타')
        print("DataFrame 로드 완료.")

        return vector_db, df_merged

    except Exception as e:
        print(f"DB 로드 중 오류 발생: {e}")
        return None, pd.DataFrame()

# [핵심] DB 로드 함수 호출
vector_db, df_jeju = load_persistent_db()


# =========================
# 3) Tmap 헬퍼 (POI/경로/지도)
# =========================
@tool
def _tmap_poi_coords(keyword: str, count: int = 1) -> str:
    """
    (도구) 장소 이름(keyword)으로 Tmap POI를 검색하여 좌표(lat, lon), 주소, 장소 이름을 JSON 문자열로 반환합니다.
    """
    if not TMAP_APP_KEY:
        return json.dumps({"error": "TMAP_APP_KEY가 설정되어 있지 않습니다."})
    url = "https://apis.openapi.sk.com/tmap/pois"
    params = {
        "version": 1,
        "searchKeyword": keyword,
        "count": count,
        "resCoordType": "WGS84GEO",
        "reqCoordType": "WGS84GEO",
        "appKey": TMAP_APP_KEY
    }
    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        pois = r.json().get("searchPoiInfo", {}).get("pois", {}).get("poi", [])
        if not pois:
            return json.dumps({"error": f"'{keyword}'에 대한 POI를 찾을 수 없습니다."})
        
        found = None
        for p in pois:
            addr = p.get("roadName") or p.get("legalDong") or p.get("upperAddrName")
            if addr:
                found = {
                    "name": p["name"],
                    "latitude": float(p["frontLat"]),
                    "longitude": float(p["frontLon"]),
                    "address": f"{addr} {p.get('detailAddrName', '')}".strip()
                }
                break
        
        if found:
            return json.dumps(found, ensure_ascii=False)
        else:
            p = pois[0]
            return json.dumps({
                "name": p["name"],
                "latitude": float(p["frontLat"]),
                "longitude": float(p["frontLon"]),
                "address": "주소 정보 없음"
            }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Tmap POI 검색 중 오류 발생: {e}"})

def _tmap_route_sequential(start_lat, start_lon, end_lat, end_lon, via_points=None, search_option=0, res_coord="WGS84GEO"):
    """(헬퍼) 다중 경유지 경로 계산"""
    url = "https://apis.openapi.sk.com/tmap/routes/routeSequential30?version=1&format=json"
    headers = {"appKey": TMAP_APP_KEY, "Content-Type": "application/json"}
    body = {
        "startName": "출발지", "startX": str(start_lon), "startY": str(start_lat),
        "endName": "도착지",   "endX": str(end_lon),     "endY": str(end_lat),
        "reqCoordType": "WGS84GEO", "resCoordType": res_coord,
        "searchOption": search_option
    }
    if via_points:
        body["viaPoints"] = [
            {"viaPointId": f"via{i+1}", "viaPointName": vp[2], "viaX": str(vp[1]), "viaY": str(vp[0])}
            for i, vp in enumerate(via_points)
        ]
    r = requests.post(url, headers=headers, data=json.dumps(body), timeout=15)
    r.raise_for_status()
    return r.json()

def _draw_route_on_map(route_json, res_coord="WGS84GEO", zoom_start=11):
    """(헬퍼) Tmap 경로 JSON → Folium 지도"""
    props = route_json.get("properties", {})
    feats = route_json.get("features", [])
    print(f"지도 생성: 총 거리 {props.get('totalDistance')}m, 총 시간 {props.get('totalTime')}초")

    lines, markers = [], []
    for f in feats:
        g = f.get("geometry", {})
        p = f.get("properties", {})
        if g.get("type") == "LineString":
            latlngs = [[(y, x) if res_coord == "WGS84GEO" else (x, y) for x, y in g.get("coordinates", [])]]
            lines.extend(latlngs)
        else:
            coord = g.get("coordinates", [])
            if coord:
                lat, lon = (coord[1], coord[0]) if res_coord == "WGS84GEO" else (coord[0], coord[1])
                markers.append((lat, lon, p.get("description", p.get("pointType", ""))))

    center = lines[0][0] if lines else (markers[0][:2] if markers else [33.4996,126.5312])
    m = folium.Map(location=center, zoom_start=zoom_start)
    for latlngs in lines:
        AntPath(latlngs, reversed=True).add_to(m)
    for (lat, lon, desc) in markers:
        folium.Marker([lat, lon], tooltip=desc or "point").add_to(m)
    return m


# =========================
# 4) 도구들
# =========================
@tool
def search_jeju_tour_spots_semantic(query: str) -> str:
    """
    (도구) 감성/주관 키워드(예: '분위기 좋은', '아이랑', '힐링')로 유사 관광지 4곳을 반환합니다.
    """
    if vector_db is None or df_jeju.empty:
        return json.dumps({"error": "검색 DB가 준비되지 않았습니다."})
    similar_docs = vector_db.similarity_search(query, k=4)
    if not similar_docs:
        return json.dumps({"error": "관련 정보를 찾지 못했습니다."})
    
    results = []
    for doc in similar_docs:
        meta = doc.metadata
        results.append({
            "name": meta.get("name"),
            "address": meta.get("addr"),
            "category": meta.get("cat", "기타"), # 'cat'이 '소분류'
            "theme": meta.get("theme")
        })
    return json.dumps(results, ensure_ascii=False)

@tool
def get_detailed_description(spot_name: str) -> str:
    """
    (도구) 웹 검색 결과를 요약(운영시간/입장료/특징/리뷰)을 간결히 반환합니다.
    """
    # ⬇️⬇️ [수정] TavilySearchResults에 키를 명시적으로 전달합니다.
    info = TavilySearchResults(k=2, tavily_api_key=TAVILY_API_KEY).invoke(f"{spot_name} 상세 정보, 운영 시간, 입장료, 특징, 최근 리뷰")
    if not info:
        return json.dumps({"description": f"'{spot_name}'에 대한 상세 정보를 웹에서 찾을 수 없습니다."})
    prompt = ChatPromptTemplate.from_template(
        "다음은 '{spot}'에 대한 검색 결과입니다. 방문객에게 유용한 핵심(특징, 분위기, 주요 활동)을 50자 이내로 간결히 요약하세요.\n"
        "--- 검색 결과 ---\n{info}\n--- 요약 ---"
    )
    summary = (prompt | llm).invoke({"spot": spot_name, "info": json.dumps(info, ensure_ascii=False)}).content
    return json.dumps({"description": summary}, ensure_ascii=False)

@tool
def get_weather_forecast(location: str, date: str) -> str:
    """ (도구) Open-Meteo로 실시간 예보 요약 """
    GEO = {"제주시": (33.4996,126.5312), "서귀포시": (33.2539,126.5596)}
    try:
        from dateutil import parser as dateparser
        target = dateparser.parse(date).date().isoformat()
    except Exception:
        return "날짜는 YYYY-MM-DD 형식으로 입력해주세요."
    lat, lon = GEO.get(location, (33.4996,126.5312))
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           "&hourly=temperature_2m,precipitation&timezone=Asia%2FSeoul")
    r = requests.get(url, timeout=8)
    if r.status_code != 200:
        return f"날씨 조회 실패: HTTP {r.status_code}"
    h = r.json().get("hourly", {})
    rows = [(t,temp,pr) for t,temp,pr in zip(h.get("time",[]), h.get("temperature_2m",[]), h.get("precipitation",[])) if t.startswith(target)]
    if not rows:
        return f"{location}의 {target} 예보가 없습니다."
    preview = "\n".join([f"- {t.split('T')[1]}: {temp:.1f}℃, 강수 {pr}mm" for t,temp,pr in rows[:4]])
    return f"[{location} {target} 날씨]\n{preview}\n※ 비 예보 시 실내 코스를 제안합니다."

# ⬇️⬇️ [수정] 오류가 발생한 지점. web_search_tool에도 키를 명시적으로 전달합니다.
web_search_tool = TavilySearchResults(k=3, name="general_web_search", tavily_api_key=TAVILY_API_KEY)

@tool
def build_route_map(locations: List[str]) -> str:
    """ (도구) 장소 리스트로 Tmap 경로를 계산해 Folium HTML 지도를 저장하고 *웹 URL*을 반환합니다. """
    if not TMAP_APP_KEY:
        return "오류: TMAP_APP_KEY가 설정되어 있지 않습니다."
    coords=[]
    for name in locations:
        poi_str = _tmap_poi_coords("제주 " + name)
        try:
            poi_data = json.loads(poi_str)
            if "error" not in poi_data:
                coords.append((poi_data["name"], poi_data["latitude"], poi_data["longitude"]))
        except Exception:
            pass 

    if len(coords) < 2:
        return "좌표를 2곳 이상 찾지 못해 경로를 만들 수 없습니다."
    
    start, end = coords[0], coords[-1]
    vias = [(lat, lon, nm) for (nm, lat, lon) in coords[1:-1]]
    
    try:
        route_json = _tmap_route_sequential(start[1], start[2], end[1], end[2], via_points=vias, res_coord="WGS84GEO")
        m = _draw_route_on_map(route_json, res_coord="WGS84GEO")
        
        filename = f"route_map_{uuid.uuid4()}.html"
        out_path = os.path.join(STATIC_DIR, filename)
        m.save(out_path)
        
        web_url = f"/{STATIC_DIR}/{filename}"
        return f"경로 지도를 저장했습니다: {web_url}"
    
    except Exception as e:
        print(f"경로 지도 생성 중 오류: {e}")
        return f"경로 지도 생성에 실패했습니다: {e}"

@tool
def generate_route_link(locations: List[str]) -> str:
    """ (도구) Google Maps Directions 링크 """
    pts=[]
    for name in locations:
        poi_str = _tmap_poi_coords("제주 " + name)
        try:
            poi_data = json.loads(poi_str)
            if "error" not in poi_data:
                pts.append((poi_data["latitude"], poi_data["longitude"]))
        except Exception:
            pass

    if len(pts) < 2:
        return "좌표를 2곳 이상 찾지 못해 링크를 만들 수 없습니다."
    origin = f"{pts[0][0]},{pts[0][1]}"
    destination = f"{pts[-1][0]},{pts[-1][1]}"
    
    base = "http://googleusercontent.com/maps/google.com/1"
    url_parts = [base, quote(origin)]
    
    if len(pts) > 2:
        waypoints_str = "/".join([f"{lat},{lon}" for (lat,lon) in pts[1:-1]])
        url_parts.append(waypoints_str)
        
    url_parts.append(quote(destination))
    url = "/".join(url_parts)
        
    return f"웹에서 경로 보기: {url}"

@tool
def save_simple_plan(title: str, plan_details: List[dict]) -> str:
    """
    (도구) 에이전트가 생성한 간단한 일정(장소 이름 목록)을 백엔드 상태(state['travel_plan'])에 저장하도록 요청합니다.
    """
    try:
        plan_obj = SimpleTravelPlan(
            title=title,
            plan=[SimpleDailyPlan(**day_plan) for day_plan in plan_details]
        )
        return plan_obj.model_dump_json(indent=2)
    except Exception as e:
        return json.dumps({"error": f"일정 저장 형식 오류: {e}"})


# =========================
# 5) 에이전트 그래프
# =========================
tools = [
    search_jeju_tour_spots_semantic,
    get_detailed_description,
    get_weather_forecast,
    _tmap_poi_coords, # 도구로 추가
    build_route_map,
    generate_route_link,
    web_search_tool,
    save_simple_plan # 도구로 추가
]

SYSTEM_PROMPT = """
당신은 친절하고 유능한 제주도 전문 여행 가이드입니다.
당신은 사용자에게 응답할 때, **반드시 `AgentResponse` JSON 스키마를 '도구 호출' 형식으로만 반환**해야 합니다. 일반 텍스트로 응답해서는 안 됩니다.

[규칙]
1.  **최종 응답은 `AgentResponse` 스키마로만**:
    * 사용자에게 인사, 질문, 확인 등 모든 대화는 `AgentResponse`를 호출하고 `action`을 "chat"으로 설정하세요.
    * 예: `AgentResponse(response_text="안녕하세요! 며칠 여행하시나요?", action="chat")`
2.  **`state['travel_plan']`**:
    * 현재 사용자와 공유 중인 "간소화된 일정" (장소 이름 목록)이 `state['travel_plan']`에 dict로 저장되어 있습니다.
    * 일정이 변경될 때마다 `save_simple_plan` 도구를 호출하여 `state['travel_plan']`을 업데이트해야 합니다.
3.  **일정 생성 (`action='create_schedule'`)**:
    * 사용자에게 기간, 테마, 동반자를 질문합니다 (`action='chat'`).
    * 정보가 모이면, `search_jeju_tour_spots_semantic`을 사용해 각 날짜/시간대별 장소를 찾습니다.
    * **필수**: 찾은 *각 장소*에 대해 `_tmap_poi_coords` (lat/lon/address)와 `get_detailed_description` (description)을 **반드시 호출**하여 모든 상세 정보를 수집합니다.
    * `search_jeju_tour_spots_semantic` 결과에서 'category' 정보를 가져옵니다.
    * `time_slot` ('morning', 'afternoon', 'evening')을 적절히 배정합니다.
    * (중요) `save_simple_plan`을 먼저 호출하여 간소화된 일정을 `state`에 저장합니다.
    * 마지막으로, 모든 상세 정보(PlaceDetail)가 포함된 `schedule` 목록을 만들어 `AgentResponse(action='create_schedule', ...)`를 호출합니다.
4.  **일정 수정/대안 (`action='suggest_alternative' / 'update_schedule'`)**:
    * `state['travel_plan']`을 참조하여 `target` 객체를 구성합니다.
    * 대안 장소를 `search_...`로 찾고, `_tmap_poi_coords`, `get_detailed_description`로 상세 정보를 수집합니다.
    * `save_simple_plan`을 호출하여 `state`를 업데이트합니다.
    * `AgentResponse(action='suggest_alternative', ...)` 또는 `AgentResponse(action='update_schedule', ...)`를 호출합니다.
5.  **도구 사용**:
    * `AgentResponse`를 호출하기 전에 필요한 모든 정보(좌표, 설명 등)를 다른 도구(search, _tmap, get_desc)를 통해 수집해야 합니다.
    * `AgentResponse`는 항상 대화 턴의 *마지막*에 호출되어야 합니다.
"""

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    travel_plan: dict | None # SimpleTravelPlan의 dict 버전

tool_node = ToolNode(tools)

def call_model(state: AgentState):
    print("---CALLING MODEL---")
    messages = state["messages"]
    current_plan = state.get("travel_plan")
    
    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    if current_plan:
        plan_json = json.dumps(current_plan, indent=2, ensure_ascii=False)
        msgs.append(SystemMessage(content=(
            "현재 공유 중인 간소화된 여행 계획(JSON)이 있습니다. "
            "일정 생성/수정 시 이 `state`를 참조하고, 변경 시 `save_simple_plan`을 호출하세요.\n"
            f"```json\n{plan_json}\n```"
        )))
    
    # LLM이 AgentResponse 스키마도 도구처럼 호출하도록 바인딩
    model_with_tools = llm.bind_tools(tools + [AgentResponse]) 
    response = model_with_tools.invoke(msgs)
    
    return {"messages": [response]}

def update_plan_from_tool(state: AgentState):
    """ save_simple_plan 도구 호출 시 state['travel_plan'] 업데이트 """
    print("---UPDATING STATE FROM TOOL---")
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage):
        if last_message.name == "save_simple_plan":
            try:
                plan_dict = json.loads(last_message.content)
                if "error" not in plan_dict:
                    state["travel_plan"] = plan_dict 
                    print("✅ Travel plan updated in state.")
                else:
                    print(f"❗️ Failed to save plan: {plan_dict['error']}")
            except Exception as e:
                print(f"❗️ Failed to parse save_simple_plan output: {e}")
    return state


def should_continue(state: AgentState):
    """ 마지막 메시지가 AgentResponse 호출이면 'end' """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        # AgentResponse 호출은 대화 턴의 끝을 의미
        if last.tool_calls[0]["name"] == "AgentResponse":
            return "end"
        # 다른 도구 호출은 계속
        return "continue"
    # LLM이 프롬프트를 무시하고 일반 텍스트 응답 시 'end'
    return "end"

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_node("update_plan", update_plan_from_tool)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
workflow.add_edge("action", "update_plan")
workflow.add_edge("update_plan", "agent")

agent_app = workflow.compile()
print("\n✅ LangGraph Agent 컴파일 완료!")