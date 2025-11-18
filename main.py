# main.py
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from app.model import agent_app
from app.schema import ChatRequest
from typing import List
from app.tmap_service import generate_route_html

app_fastapi = FastAPI()

app_fastapi.mount("/static", StaticFiles(directory="static"), name="static")

@app_fastapi.post("/chat")
async def chat_endpoint(request: ChatRequest):
    
    config = {"configurable": {"thread_id": str(request.user_id)}}
    user_input = ""

    if request.action == "create_schedule" and request.constraints:
        c = request.constraints
        user_input = (
            f"다음 조건으로 제주도 여행 일정을 생성해줘(create_schedule).\n"
            f"- 기간: {c.start_date} ~ {c.end_date}\n"
            f"- 스타일: {c.travel_style}\n"
            f"- 동반자: {c.companions}\n"
            f"- 연령대: {c.age_group}\n"
            f"- 추가요청: {c.additional_request or '없음'}"
        )
        print(f"🆕 [User {request.user_id}] 일정 생성 요청")

    elif request.message:
        user_input = request.message
        print(f"💬 [User {request.user_id}] 메시지: {user_input}")
    
    else:
        raise HTTPException(status_code=400, detail="요청 형식이 올바르지 않습니다.")

    try:
        # invoke 실행
        inputs = {"messages": [HumanMessage(content=user_input)]}
        result = agent_app.invoke(inputs, config=config)
        
        # ⭐️ [데이터 추출 로직 수정]
        # 그래프가 Tool 실행까지 마치고 돌기 때문에, 역순으로 탐색해서 
        # 'submit_final_response'를 호출한 AI 메시지를 찾습니다.
        
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call["name"] == "submit_final_response":
                        # 우리가 원하던 JSON 데이터는 여기 arguments에 있습니다.
                        print("✅ 최종 응답 데이터 추출 성공")
                        return tool_call["args"]

        # 예외: 도구 호출 없이 텍스트로 끝난 경우
        last_message = result["messages"][-1]
        return {
            "action": "chat",
            "response_text": last_message.content if hasattr(last_message, 'content') else "응답을 생성할 수 없습니다.",
            "schedule": None
        }

    except Exception as e:
        print(f"❌ 서버 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class MapRequest(BaseModel):
    places: List[str] # 예: ["제주공항", "애월카페", "협재해수욕장"]

@app_fastapi.post("/map/create")
async def create_map_endpoint(request: MapRequest):
    """
    [시연용] 장소 이름 리스트를 받아서 실제 이동 경로가 그려진 HTML 지도를 생성합니다.
    반환값: 지도 HTML 파일의 URL
    """
    print(f"🗺️ 지도 생성 요청: {request.places}")
    
    map_url = generate_route_html(request.places)
    
    if map_url:
        return {"status": "success", "map_url": map_url}
    else:
        raise HTTPException(status_code=500, detail="지도 생성에 실패했습니다. (좌표 부족 또는 API 오류)")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app_fastapi, host="0.0.0.0", port=8001)


