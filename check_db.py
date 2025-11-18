import weaviate
from weaviate.classes.init import Auth
import os
from dotenv import load_dotenv

load_dotenv()

# 클라이언트 연결
client = weaviate.connect_to_wcs(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)

try:
    # 제주 스팟 컬렉션 가져오기
    collection = client.collections.get("JejuSpot")
    
    # 개수 세기
    count = collection.aggregate.over_all(total_count=True).total_count
    print(f"\n🎉 대성공! 현재 저장된 데이터 개수: {count}개")

finally:
    client.close()