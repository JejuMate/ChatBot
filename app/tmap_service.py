# app/tmap_service.py
import os
import requests
import json
import uuid
import folium
from folium.plugins import AntPath
from dotenv import load_dotenv

load_dotenv()

TMAP_APP_KEY = os.getenv("TMAP_API_KEY", "")
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

def get_tmap_coords(keyword: str) -> dict:
    """Tmap API로 좌표 검색"""
    if not TMAP_APP_KEY: 
        print("❌ TMAP_APP_KEY가 없습니다.")
        return None
        
    url = "https://apis.openapi.sk.com/tmap/pois"
    params = {
        "version": 1, "searchKeyword": keyword, "count": 1,
        "resCoordType": "WGS84GEO", "reqCoordType": "WGS84GEO", "appKey": TMAP_APP_KEY
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        pois = data.get("searchPoiInfo", {}).get("pois", {}).get("poi", [])
        if pois:
            return {
                "name": pois[0]["name"],
                "lat": float(pois[0]["frontLat"]),
                "lon": float(pois[0]["frontLon"])
            }
    except Exception as e:
        print(f"❌ Tmap POI Error: {e}")
    return None

def generate_route_html(locations: list[str]) -> str:
    """장소 리스트 -> 경로 지도 HTML 생성"""
    print(f"🔍 지도 생성 시작: {len(locations)}개 장소 -> {locations}")
    
    # 1. 좌표 변환
    points = []
    for loc in locations:
        search_key = loc if "제주" in loc else f"제주 {loc}"
        coords = get_tmap_coords(search_key)
        if not coords: # 실패 시 원본으로 재시도
            coords = get_tmap_coords(loc)
        
        if coords: points.append(coords)
    
    if len(points) < 2:
        print(f"❌ 좌표 부족 (찾은 개수: {len(points)})")
        return None

    start, end = points[0], points[-1]

    # 2. API 분기 처리 (핵심 수정)
    headers = {"appKey": TMAP_APP_KEY, "Content-Type": "application/json"}
    
    # 기본 바디 (공통)
    body = {
        "startName": start["name"], "startX": str(start["lon"]), "startY": str(start["lat"]),
        "endName": end["name"], "endX": str(end["lon"]), "endY": str(end["lat"]),
        "reqCoordType": "WGS84GEO", "resCoordType": "WGS84GEO", "searchOption": 0
    }

    if len(points) == 2:
        # [CASE A] 점 2개 -> 일반 경로 API (/routes) 사용
        # 이게 훨씬 안정적입니다.
        url = "https://apis.openapi.sk.com/tmap/routes?version=1&format=json"
    else:
        # [CASE B] 점 3개 이상 -> 다중 경유지 API (/routeSequential30) 사용
        url = "https://apis.openapi.sk.com/tmap/routes/routeSequential30?version=1&format=json"
        vias = points[1:-1][:5]
        body["viaPoints"] = [
            {"viaPointId": f"v{i}", "viaPointName": p["name"], "viaX": str(p["lon"]), "viaY": str(p["lat"])}
            for i, p in enumerate(vias)
        ]

    try:
        r = requests.post(url, headers=headers, data=json.dumps(body), timeout=10)
        r.raise_for_status()
        route_data = r.json()
    except Exception as e:
        print(f"❌ Tmap Route API Error: {e}")
        return None

    # 3. 지도 그리기
    try:
        center_lat = (start["lat"] + end["lat"]) / 2
        center_lon = (start["lon"] + end["lon"]) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

        path_coords = []
        for feat in route_data.get("features", []):
            geo = feat.get("geometry", {})
            if geo.get("type") == "LineString":
                # Tmap(Lon,Lat) -> Folium(Lat,Lon)
                path_coords.extend([[y, x] for x, y in geo.get("coordinates", [])])
        
        if path_coords:
            AntPath(path_coords, delay=1000, weight=5, color="blue").add_to(m)

        for i, p in enumerate(points):
            color = "red" if i == 0 or i == len(points)-1 else "blue"
            folium.Marker(
                [p["lat"], p["lon"]], 
                popup=p["name"], 
                tooltip=f"{i+1}. {p['name']}",
                icon=folium.Icon(color=color)
            ).add_to(m)

        filename = f"route_map_{uuid.uuid4()}.html"
        save_path = os.path.join(STATIC_DIR, filename)
        m.save(save_path)
        
        print(f"✅ 지도 생성 완료: {filename}")
        return f"/static/{filename}"

    except Exception as e:
        print(f"❌ 지도 그리기 오류: {e}")
        return None