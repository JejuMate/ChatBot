import uvicorn
from app.api import app_fastapi

# uvicorn이 이 'app_fastapi' 객체를 참조하여 서버를 실행합니다.

if __name__ == "__main__":
    # 터미널에서 'python main.py'로 직접 실행할 경우
    print("FastAPI 서버를 시작합니다. http://127.0.0.1:8000/docs 에서 문서를 확인하세요.")
    uvicorn.run("main:app_fastapi", host="0.0.0.0", port=8000, reload=True)