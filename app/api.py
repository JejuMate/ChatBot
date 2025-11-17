import json
from fastapi import FastAPI, APIRouter
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# [필수] CORS 임포트
from fastapi.middleware.cors import CORSMiddleware

# --- 내부 모듈 임포트 ---
from .schema import (
    ChatTurnRequest, ChatTurnResponse, ChatMessage, 
    AgentResponse, SimpleTravelPlan
)
from .model import agent_app, STATIC_DIR

# --- FastAPI 앱 및 라우터 초기화 ---
app_fastapi = FastAPI(
    title="제주 여행 챗봇 API",
    description="LangGraph와 FastAPI를 이용한 제주 여행 일정 플래너",
    version="1.1.0"
)

# [필수] CORS 미들웨어 설정
app_fastapi.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

api_router = APIRouter()

# --- API 엔드포인트 ---
@api_router.post("/chat", response_model=ChatTurnResponse)
def chat_endpoint(request: ChatTurnRequest):
    """
    사용자 메시지를 받아 챗봇의 다음 응답을 반환합니다.
    (상태는 클라이언트가 매번 전송)
    """
    
    # 1. [API 스키마] -> [LangChain 메시지]
    lc_messages = []
    for msg in request.messages:
        if msg.role == 'user':
            lc_messages.append(HumanMessage(content=msg.content))
        
        # ⬇️⬇️⬇️ [핵심 수정] ⬇️⬇️⬇️
        elif msg.role == 'ai':
            # AI의 응답은 'tool_call'이 아니라 'content'로만 기록합니다.
            # AI(LLM)는 'AgentResponse'의 JSON 구조가 아니라, 
            # 자기가 했던 말('response_text')만 기억하면 됩니다.
            try:
                # msg.content는 AgentResponse의 JSON 문자열입니다.
                ai_json = json.loads(msg.content) 
                if isinstance(ai_json, dict) and "response_text" in ai_json:
                    # 'response_text'만 뽑아서 content에 넣습니다.
                    lc_messages.append(AIMessage(content=ai_json['response_text']))
                else:
                    # JSON이 아니거나 형식이 다른 경우 (예: 이전 버전의 오류 메시지)
                    lc_messages.append(AIMessage(content=msg.content))
            except json.JSONDecodeError:
                # JSON이 아닌 일반 텍스트인 경우
                lc_messages.append(AIMessage(content=msg.content))
        # ⬆️⬆️⬆️ [핵심 수정 완료] ⬆️⬆️⬆️

    # 2. [API 스키마] -> [AgentState (dict)]
    travel_plan_dict = request.travel_plan

    # 3. 현재 상태(State) 구성
    input_state = {
        "messages": lc_messages,
        "travel_plan": travel_plan_dict
    }
    
    # 4. LangGraph 실행
    try:
        final_state = agent_app.invoke(input_state)
        
        # 5. [최종 AI 응답] -> [API 스키마 (AgentResponse)]
        ai_response_msg = final_state["messages"][-1]
        
        response_model: AgentResponse

        if isinstance(ai_response_msg, AIMessage) and ai_response_msg.tool_calls:
            tool_call = ai_response_msg.tool_calls[0]
            if tool_call["name"] == "AgentResponse":
                response_model = AgentResponse(**tool_call["args"])
            else:
                response_model = AgentResponse(
                    response_text=f"오류: 에이전트가 응답 대신 도구({tool_call['name']})를 호출하며 종료했습니다.",
                    action="chat"
                )
        else:
            response_model = AgentResponse(
                response_text=ai_response_msg.content or "...",
                action="chat"
            )

        # 6. [AgentState (dict)] -> [API 스키마]
        updated_plan_dict = final_state.get("travel_plan")

        return ChatTurnResponse(
            response=response_model,
            travel_plan=updated_plan_dict
        )
        
    except Exception as e:
        print(f"[API Error] /chat : {e}")
        return ChatTurnResponse(
            response=AgentResponse(
                response_text=f"죄송합니다. 서버 처리 중 오류가 발생했습니다: {e}",
                action="chat"
            ),
            travel_plan=request.travel_plan
        )

@app_fastapi.get("/")
def read_root():
    return {"message": "제주 여행 챗봇 API입니다. /docs 로 이동하여 API 문서를 확인하세요."}

# --- 정적 파일 (지도) 서빙 및 라우터 포함 ---
app_fastapi.mount(f"/{STATIC_DIR}", StaticFiles(directory=STATIC_DIR), name=STATIC_DIR)
app_fastapi.include_router(api_router, prefix="/api") # 예: /api/chat