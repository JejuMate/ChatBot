# main.py
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

# 모듈 가져오기
from app.model import agent_app
from app.schema import ChatRequest
# app/tmap_service.py가 있어야 합니다.
from app.tmap_service import generate_route_html 
from typing import List
import uvicorn

app_fastapi = FastAPI(
    title="Jeju Travel AI Backend",
    description="제주도 여행 일정 생성 및 지도 서비스 API",
    version="1.0"
)

# 정적 파일(지도 HTML 등) 서빙 설정
app_fastapi.mount("/static", StaticFiles(directory="static"), name="static")

@app_fastapi.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    AI 에이전트와 대화하거나 일정을 생성하는 엔드포인트
    """
    # LangGraph 설정 (Thread ID로 대화 맥락 유지)
    config = {"configurable": {"thread_id": str(request.user_id)}}
    
    user_input = ""
    
    # 1. 사용자 입력 텍스트 구성 (로그용 및 HumanMessage용)
    if request.action == "create_schedule" and request.constraints:
        c = request.constraints
        user_input = (
            f"다음 조건으로 제주도 여행 일정을 생성해줘.\n"
            f"- 기간: {c.start_date} ~ {c.end_date}\n"
            f"- 스타일: {c.travel_style}\n"
            f"- 동반자: {c.companions}\n"
            f"- 연령대: {c.age_group}\n"
            f"- 추가요청: {c.additional_request or '없음'}"
        )
        print(f"🆕 [User {request.user_id}] 일정 생성 요청: {c.start_date}~{c.end_date}")

    elif request.message:
        user_input = request.message
        print(f"💬 [User {request.user_id}] 메시지: {user_input}")
    
    else:
        raise HTTPException(status_code=400, detail="요청 형식이 올바르지 않습니다. (constraints 또는 message 필수)")

    try:
        # ⭐️ [핵심 수정] 상태(State)에 constraints와 action을 함께 넘겨줘야 model.py가 인식합니다!
        inputs = {
            "messages": [HumanMessage(content=user_input)],
            "user_id": request.user_id,
            "action": request.action,
            "constraints": request.constraints.dict() if request.constraints else {}
        }

        # 에이전트 실행
        result = agent_app.invoke(inputs, config=config)
        
        # 3. 결과 추출 로직 (submit_final_response 도구의 출력을 찾음)
        # 역순으로 탐색하여 가장 최근의 AI 응답(도구 호출)을 찾습니다.
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call["name"] == "submit_final_response":
                        print("✅ 최종 응답 데이터 추출 성공")
                        return tool_call["args"] # 여기가 최종 JSON (AgentResponse)

        # 예외: 도구 호출 없이 텍스트로만 끝난 경우 (에러 상황 등)
        last_message = result["messages"][-1]
        return {
            "action": "chat",
            "response_text": last_message.content if hasattr(last_message, 'content') else "죄송합니다. 응답을 생성하는 데 실패했습니다.",
            "schedule": None
        }

    except Exception as e:
        print(f"❌ 서버 에러 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 지도 생성용 요청 스키마
class MapRequest(BaseModel):
    places: List[str] 

@app_fastapi.post("/map/create")
async def create_map_endpoint(request: MapRequest):
    """
    [지도 서비스] 장소 이름 리스트 -> 경로 지도 HTML URL 반환
    """
    print(f"🗺️ 지도 생성 요청: {request.places}")
    
    try:
        map_url = generate_route_html(request.places)
        if map_url:
            return {"status": "success", "map_url": map_url}
        else:
            raise HTTPException(status_code=500, detail="지도 생성 실패 (좌표 검색 불가 등)")
    except Exception as e:
        print(f"❌ 지도 생성 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # main.py 직접 실행 시 8001 포트로 실행
    uvicorn.run(app_fastapi, host="0.0.0.0", port=8001)