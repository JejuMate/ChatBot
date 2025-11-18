# 1. dotenv 라이브러리 및 os import
import os
from dotenv import load_dotenv

# 2. 로드 함수 호출 (가장 먼저 실행되어야 함)
load_dotenv()
BATCH_SIZE = 50       # ⭐️ 배치를 100 -> 50으로 줄임 (안전제일)
SLEEP_TIME = 1.5      # ⭐️ 배치마다 1.5초씩 쉼 (OpenAI 과부하 방지)

import time  # ⭐️ 시간 지연을 위해 추가
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
import os
from dotenv import load_dotenv
import pandas as pd

# 현재 스크립트의 기본 경로 (프로젝트 루트)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data') # CSV 파일이 data 폴더에 있다고 가정

# 파일 경로 설정 (경로를 하드코딩하지 않음)
CATEGORY_PATH = os.path.join(DATA_DIR, 'V3_관광지카테고리_END.csv')
VISIT_PATH = os.path.join(DATA_DIR, 'visit_jeju.csv')

## 데이터 병합 및 결측치 처리 ##

# 1. 파일 로드 (경로 에러 방지를 위해 경로 변수 사용)
try:
    df_category = pd.read_csv(CATEGORY_PATH)
    df_visit = pd.read_csv(VISIT_PATH)
except FileNotFoundError as e:
    print(f"오류: 파일을 찾을 수 없습니다. {e}")
    # 파일 경로를 확인하세요. (예: /ChatBot/data/V3_관광지카테고리_END.csv)
    exit()

# 2. Left Join 실행 (V3_관광지카테고리_END.csv 기준으로 visit_jeju.csv 병합)
df_merged = pd.merge(
    df_category, 
    df_visit, 
    left_on='VISIT_AREA_NM', 
    right_on='이름', 
    how='left'
)

# 3. 불필요한 중복 컬럼 및 Nan 처리
df_merged = df_merged.drop(columns=['이름'], errors='ignore')

# 4. Weaviate 임베딩에 사용할 '테마' 필드의 결측치를 빈 문자열로 대체
# (병합 시 visit_jeju에 없던 장소의 '테마' 필드는 NaN이 됨)
df_merged['테마'] = df_merged['테마'].fillna('')

print(f"✅ 데이터 병합 완료. 총 {len(df_merged)}개의 장소 데이터 준비됨.")


import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = weaviate.connect_to_wcs(
    cluster_url=WEAVIATE_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    headers={"X-OpenAI-Api-Key": OPENAI_API_KEY}
)

collection_name = "JejuSpot"

try:
    # 1. 기존 데이터 삭제 (초기화)
    if client.collections.exists(collection_name):
        print(f"🧹 기존 {collection_name} 삭제 중 (데이터 불일치 해결)...")
        client.collections.delete(collection_name)
    
    # 2. 스키마 다시 생성
    print(f"🔨 새로운 {collection_name} 스키마 생성 중...")
    client.collections.create(
        name=collection_name,
        vectorizer_config=Configure.Vectorizer.text2vec_openai(),
        properties=[
            Property(name="name", data_type=DataType.TEXT),
            Property(name="address", data_type=DataType.TEXT),
            Property(name="category", data_type=DataType.TEXT),
            Property(name="themeTags", data_type=DataType.TEXT),
            Property(name="searchText", data_type=DataType.TEXT),
            Property(name="totalScore", data_type=DataType.NUMBER),
            Property(name="xCoord", data_type=DataType.NUMBER),
            Property(name="yCoord", data_type=DataType.NUMBER),
            Property(name="poiId", data_type=DataType.TEXT),
        ]
    )

    # 3. 천천히 업로드 (Rate Limit 회피)
    print("📦 데이터 업로드 시작 (안전 모드)...")
    collection = client.collections.get(collection_name)
    
    total_rows = len(df_merged)
    counter = 0

    # 수동 배치 처리 (with batch 대신 직접 제어)
    buffer = []
    
    for index, row in df_merged.iterrows():
        # 텍스트 생성
        theme_text = str(row['테마']) if pd.notna(row['테마']) else ''
        search_text = (
            f"장소명: {row['VISIT_AREA_NM']}. 주소: {row['ROAD_NM_ADDR']}. "
            f"카테고리: {row['소분류']}. 상세 테마: {theme_text}"
        )

        data_object = {
            "name": row['VISIT_AREA_NM'],
            "address": row['ROAD_NM_ADDR'],
            "category": row['소분류'],
            "themeTags": theme_text,
            "searchText": search_text,
            "totalScore": float(row['Total_Score']) if pd.notna(row['Total_Score']) else 0.0,
            "xCoord": float(row['X_COORD']) if pd.notna(row['X_COORD']) else 0.0,
            "yCoord": float(row['Y_COORD']) if pd.notna(row['Y_COORD']) else 0.0,
            "poiId": str(row['POI_ID']) if pd.notna(row['POI_ID']) else ""
        }
        
        buffer.append(data_object)

        # 버퍼가 꽉 차면 전송
        if len(buffer) >= BATCH_SIZE:
            try:
                # 데이터 전송
                collection.data.insert_many(buffer)
                counter += len(buffer)
                
                # 진행률 출력
                print(f" -> {counter}/{total_rows} 완료 (OpenAI 쉬는 중...)")
                
                # ⭐️ 중요: OpenAI가 쉴 시간을 줍니다
                time.sleep(SLEEP_TIME) 
                
            except Exception as e:
                print(f"❌ 배치 전송 오류: {e}")
            
            buffer = [] # 버퍼 비우기

    # 남은 데이터 처리
    if buffer:
        collection.data.insert_many(buffer)
        counter += len(buffer)
        print(f" -> {counter}/{total_rows} 최종 완료.")

    print("✨ 안전하게 업로드 완료되었습니다! Weaviate 콘솔에서 개수를 다시 확인하세요.")

finally:
    client.close()