# Python 3.12 이미지 사용
FROM python:3.12

# 작업 디렉토리 설정
WORKDIR /

# requirements.txt 파일을 컨테이너로 복사
COPY requirements.txt .

# 필요한 시스템 패키지 설치 및 Python 패키지 설치
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY . .

# 포트 노출
EXPOSE 5000

# 환경 변수 설정
ENV FLASK_APP=app.py

# 애플리케이션 실행
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
