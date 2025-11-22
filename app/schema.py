from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# =========================================
# 1. 입력 (Input Schema)
# =========================================

class TravelConstraints(BaseModel):
    """
    여행 일정 생성에 필요한 상세 조건
    """
    start_date: str = Field(..., description="여행 시작 날짜 (YYYY-MM-DD)")
    end_date: str = Field(..., description="여행 종료 날짜 (YYYY-MM-DD)")
    travel_style: str = Field(..., description="여행 스타일 (예: 힐링, 액티비티, 등산 등)")
    companions: str = Field(..., description="동반자 (가족, 연인, 친구, 혼자)")
    age_group: str = Field(..., description="연령대")
    additional_request: Optional[str] = Field(None, description="추가 요청 사항")

class ChatMessage(BaseModel):
    """ API 통신을 위한 메시지 단일 객체 """
    role: str = Field(..., description="메시지 역할", examples=["user", "ai"])
    content: str = Field(..., description="메시지 내용")

class ChatRequest(BaseModel):
    """ /chat 엔드포인트 요청 본문 (통합형) """
    user_id: int = Field(..., description="사용자 ID")
    action: Literal["create_schedule", "chat", "update_schedule"] = Field(..., description="요청 유형")
    
    # 필수 데이터 (create_schedule 시)
    constraints: Optional[TravelConstraints] = Field(None, description="여행 제약 조건")
    message: Optional[str] = Field(None, description="사용자 입력 메시지")
    
    # 대화 히스토리 (선택)
    messages: Optional[List[ChatMessage]] = Field(default=[], description="대화 내역")
    
    # 백엔드 상태 유지용 (팀원 로직 호환)
    travel_plan: Optional[dict] = Field(None, description="현재까지 확정된 간소화된 여행 일정 (JSON dict)")


# =========================================
# 2. 출력 (Output Schema) - 하위 객체들
# =========================================

class PlaceDetail(BaseModel):
    """(JSON 출력용) 장소 상세 정보 스키마"""
    name: str = Field(..., description="장소 이름")
    category: str = Field(..., description="장소 카테고리")
    latitude: float = Field(..., description="위도 (WGS84)")
    longitude: float = Field(..., description="경도 (WGS84)")
    address: str = Field(..., description="도로명 주소")
    description: Optional[str] = Field(None, description="장소 설명")

class ScheduleItem(BaseModel):
    """(JSON 출력용) 시간대별 일정 아이템"""
    day: int = Field(..., description="여행 일차 (예: 1)")
    
    # ⭐️ [추가] 사용자 요구사항: 실제 날짜
    date: str = Field(..., description="해당 일정의 날짜 (YYYY-MM-DD)")
    
    time_slot: Literal["morning", "afternoon", "evening"] = Field(..., description="시간대")
    place: PlaceDetail = Field(..., description="상세 정보")

# 팀원분 코드와의 호환성을 위해 Alias (TimeSlotScheduleItem = ScheduleItem)
TimeSlotScheduleItem = ScheduleItem 

class TargetPlace(BaseModel):
    """(JSON 출력용) 수정/삭제 대상"""
    day: int = Field(..., description="대상 여행 일차")
    time_slot: Literal["morning", "afternoon", "evening"] = Field(..., description="대상 시간대")
    place: str = Field(..., description="대상 장소 이름")


# =========================================
# 3. AgentResponse (LLM 최종 응답)
# =========================================

class AgentResponse(BaseModel):
    """LLM 최종 응답 스키마"""
    response_text: str = Field(..., description="사용자에게 보여줄 텍스트 답변")
    action: Literal["chat", "create_schedule", "suggest_alternative", "update_schedule", "remove_place"] = Field(..., description="백엔드 동작 트리거")
    
    # 사용자 요구사항: 여행 기간 및 항공권 정보
    start_date: Optional[str] = Field(None, description="여행 시작일 (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="여행 종료일 (YYYY-MM-DD)")
    flight_info: Optional[str] = Field(None, description="항공권 조회 결과 텍스트")

    # 일정 관련 필드
    schedule: Optional[List[ScheduleItem]] = Field(None, description="생성된 전체 일정 목록")
    
    # 수정/대안 관련 필드
    target: Optional[TargetPlace] = Field(None, description="대상 일정")
    alternative_places: Optional[List[PlaceDetail]] = Field(None, description="추천 대안 장소 목록")
    new_place: Optional[PlaceDetail] = Field(None, description="새로 변경할 장소")
    
    class Config:
        arbitrary_types_allowed = True


# =========================================
# 4. 백엔드 상태 저장용 (팀원 로직 유지)
# =========================================

class SimpleDailyPlan(BaseModel):
    """(백엔드 상태 저장용) 장소 이름 목록만 가진 간단한 일차별 계획"""
    day: int
    locations: List[str]
    description: str

class SimpleTravelPlan(BaseModel):
    """(백엔드 상태 저장용) 에이전트가 '기억'할 간단한 여행 계획"""
    title: str
    plan: List[SimpleDailyPlan]