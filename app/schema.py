from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# =========================
# 1) 상세 일정용 스키마 (JSON 응답용)
# =========================

class PlaceDetail(BaseModel):
    """(JSON 출력용) 장소 상세 정보 스키마"""
    name: str = Field(description="장소 이름")
    category: str = Field(description="장소 카테고리 (예: 관광지, 식당, 카페)")
    latitude: float = Field(description="위도 (WGS84)")
    longitude: float = Field(description="경도 (WGS84)")
    address: str = Field(description="도로명 주소")
    description: Optional[str] = Field(None, description="장소에 대한 간략한 설명이나 특징")

class TimeSlotScheduleItem(BaseModel):
    """(JSON 출력용) 시간대별 일정 아이템"""
    day: int = Field(description="여행 일차 (예: 1)")
    time_slot: Literal["morning", "afternoon", "evening"] = Field(description="시간대 (오전/오후/저녁)")
    place: PlaceDetail = Field(description="해당 시간대에 방문할 장소의 상세 정보")

class TargetPlace(BaseModel):
    """(JSON 출력용) 수정/삭제/대안 제시의 대상이 되는 장소"""
    day: int = Field(description="대상 여행 일차")
    time_slot: Literal["morning", "afternoon", "evening"] = Field(description="대상 시간대")
    place: str = Field(description="대상이 되는 기존 장소의 이름")


# =========================
# 2) AgentResponse (LLM의 최종 응답 스키마)
# =========================

class AgentResponse(BaseModel):
    """
    LLM 에이전트의 모든 최종 응답을 위한 단일 JSON 스키마.
    LLM은 반드시 이 스키마를 '도구 호출' 형식으로 반환해야 합니다.
    """
    response_text: str = Field(..., description="사용자에게 보여줄 친절한 챗봇 메시지")
    action: Literal["chat", "create_schedule", "suggest_alternative", "update_schedule", "remove_place"] = Field(
        ..., 
        description="응답의 의도를 나타내는 액션 태그"
    )
    
    # create_schedule 용
    schedule: Optional[List[TimeSlotScheduleItem]] = Field(
        None, 
        description="[action='create_schedule'] 일 때 사용. 상세한 전체 일정 목록."
    )
    
    # suggest_alternative, update_schedule, remove_place 용
    target: Optional[TargetPlace] = Field(
        None, 
        description="[action='suggest_alternative', 'update_schedule', 'remove_place'] 일 때 사용. 대상이 되는 일정."
    )
    
    # suggest_alternative 용
    alternative_places: Optional[List[PlaceDetail]] = Field(
        None, 
        description="[action='suggest_alternative'] 일 때 사용. 제안하는 대안 장소 목록."
    )
    
    # update_schedule 용
    new_place: Optional[PlaceDetail] = Field(
        None, 
        description="[action='update_schedule'] 일 때 사용. 새로 변경되는 장소 상세 정보."
    )


# =========================
# 3) 백엔드 상태 저장용 스키마 (기존 TravelPlan)
# =========================

class SimpleDailyPlan(BaseModel):
    """(백엔드 상태 저장용) 장소 이름 목록만 가진 간단한 일차별 계획"""
    day: int
    locations: List[str]
    description: str

class SimpleTravelPlan(BaseModel):
    """(백엔드 상태 저장용) 에이전트가 '기억'할 간단한 여행 계획"""
    title: str
    plan: List[SimpleDailyPlan]


# =========================
# 4) API 입출력 스키마
# =========================

class ChatMessage(BaseModel):
    """ API 통신을 위한 메시지 단일 객체 """
    role: str = Field(..., description="메시지 역할", examples=["user", "ai"])
    content: str = Field(..., description="메시지 내용 (AI 응답의 경우 AgentResponse의 JSON 문자열일 수 있음)")

class ChatTurnRequest(BaseModel):
    """ /chat 엔드포인트 요청 본문 """
    messages: List[ChatMessage] = Field(..., description="현재까지의 전체 대화 내역")
    # 백엔드가 상태 유지를 위해 사용하는 간단한 dict (SimpleTravelPlan)
    travel_plan: Optional[dict] = Field(None, description="현재까지 확정된 간소화된 여행 일정 (JSON dict)")

class ChatTurnResponse(BaseModel):
    """ /chat 엔드포인트 응답 본문 """
    # LLM이 생성한 상세 JSON 응답
    response: AgentResponse = Field(..., description="LLM이 생성한 상세 JSON 응답 (AgentResponse 스키마)")
    # 백엔드가 상태 유지를 위해 클라이언트에 다시 돌려주는 간단한 dict
    travel_plan: Optional[dict] = Field(None, description="업데이트된 간소화된 여행 일정 (JSON dict)")# app/schema.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# =========================================
# 1. 입력 (Input Schema)
# =========================================

class TravelConstraints(BaseModel):
    start_date: str = Field(..., description="여행 시작 날짜 (YYYY-MM-DD)")
    end_date: str = Field(..., description="여행 종료 날짜 (YYYY-MM-DD)")
    travel_style: str = Field(..., description="여행 스타일 (힐링, 액티비티, 문화&역사 등)")
    companions: str = Field(..., description="동반자 (가족, 연인, 친구, 혼자)")
    age_group: str = Field(..., description="연령대 (20대, 30대 등)")
    additional_request: Optional[str] = Field(None, description="추가 요청 사항")

class ChatRequest(BaseModel):
    user_id: int
    action: str = Field(..., description="요청 유형 (create_schedule, chat 등)")
    constraints: Optional[TravelConstraints] = None # 일정 생성 시 필수
    message: Optional[str] = None # 수정/대화 시 필수


# =========================================
# 2. 출력 (Output Schema) - 하위 객체들
# =========================================

class PlaceDetail(BaseModel):
    name: str
    category: str
    latitude: float
    longitude: float
    address: str
    description: Optional[str] = None

class ScheduleItem(BaseModel):
    day: int
    time_slot: Literal["morning", "afternoon", "evening"]
    place: PlaceDetail

class TargetPlace(BaseModel):
    day: int
    time_slot: Literal["morning", "afternoon", "evening"]
    place: str # 장소 이름

# =========================================
# 3. 출력 (Output Schema) - 최종 응답
# =========================================

class AgentResponse(BaseModel):
    """LLM이 최종적으로 반환해야 하는 응답 스키마"""
    response_text: str = Field(..., description="사용자에게 보여줄 텍스트 답변")
    action: Literal["create_schedule", "suggest_alternative", "update_schedule", "remove_place", "chat"] = Field(..., description="백엔드 동작 트리거")
    
    # action='create_schedule' 일 때 사용
    schedule: Optional[List[ScheduleItem]] = None
    
    # action='suggest_alternative', 'update_schedule', 'remove_place' 일 때 사용
    target: Optional[TargetPlace] = None
    
    # action='suggest_alternative' 일 때 사용
    alternative_places: Optional[List[PlaceDetail]] = None
    
    # action='update_schedule' 일 때 사용
    new_place: Optional[PlaceDetail] = None
    
    # LangGraph 상태 관리를 위한 내부용 (JSON 출력엔 포함 안 됨)
    class Config:
        arbitrary_types_allowed = True