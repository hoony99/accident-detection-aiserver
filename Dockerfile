# 베이스 이미지로 Python 3.9를 사용합니다.
FROM python:3.9-slim

# 작업 디렉토리를 설정합니다.
WORKDIR /app

# 필요 라이브러리와 도구들을 설치합니다.
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libgtk2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 패키지 의존성을 설치합니다.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드를 컨테이너로 복사합니다.
COPY . .

# 환경변수를 설정합니다.
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

# 포트를 엽니다.
EXPOSE 5000

# Flask 애플리케이션을 실행합니다.
CMD ["flask", "run"]
