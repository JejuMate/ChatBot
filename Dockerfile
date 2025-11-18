# 1. 파이썬 3.9 슬림 버전 사용 (가볍고 빠름)
FROM python:3.9-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 필수 시스템 패키지 설치 (혹시 모를 의존성 문제 방지)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 4. 라이브러리 목록 복사 및 설치
# (캐싱 효율을 위해 코드보다 먼저 복사함)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 전체 복사
COPY . .

# 6. 포트 노출 (컨테이너 내부 8000번)
EXPOSE 8000

# 7. 서버 실행 명령어 (0.0.0.0으로 열어야 외부 접속 가능)
CMD ["uvicorn", "main:app_fastapi", "--host", "0.0.0.0", "--port", "8000"]