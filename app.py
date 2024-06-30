import json
from flask import Flask, request, jsonify, redirect, flash
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from werkzeug.utils import secure_filename
import os
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import densenet121
from io import BytesIO
import torch.nn as nn
import time
from datetime import datetime
import boto3
from tempfile import NamedTemporaryFile
from ultralytics import YOLO
import logging
from logging.handlers import RotatingFileHandler
import requests
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # CORS 설정 추가
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB로 요청 본문 크기 제한 설정
app.config['SECRET_KEY'] = os.urandom(24).hex()
app.config['ALLOWED_EXTENSIONS'] = {'mov', 'mp4', 'avi'}
app.config['JWT_SECRET_KEY'] = 'your_jwt_secret_key'  # JWT 암호화 키 설정

load_dotenv()

# AWS 설정
S3_IMG_BUCKET = os.getenv('S3_IMG_BUCKET')
S3_VIDEO_BUCKET = os.getenv('S3_VIDEO_BUCKET')
S3_KEY = os.getenv('S3_KEY')
S3_SECRET = os.getenv('S3_SECRET')
S3_REGION = os.getenv('S3_REGION')
# S3 클라이언트 생성
s3_client = boto3.client(
    's3',
    aws_access_key_id=S3_KEY,
    aws_secret_access_key=S3_SECRET,
    region_name=S3_REGION
)
# JWT 매니저 설정
jwt = JWTManager(app)

# 로깅 설정 함수
def setup_logging():
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)

# 앱 시작 시 로깅 설정 호출
setup_logging()
# 기본 로그 메시지
app.logger.info('Flask app has started!')

# 허용된 파일 확장자 검사
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# DenseNet모델 로드
def load_densenet_model():
    device = torch.device("cpu")
    model = densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load('densenet_model50.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device

# YOLO 모델 로드
def load_yolo_model(model_path):
    model = YOLO(model_path)
    return model

# AWS S3 video 버킷에서 파일을 메모리로 직접 로드
def load_video_from_s3_to_tempfile(bucket, key):
    app.logger.info(f"Loading video from S3: bucket={bucket}, key={key}")
    try:
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket, Key=key)
        file_stream = response['Body']

        # 임시 파일 생성 및 파일 스트림 쓰기
        temp_file = NamedTemporaryFile(delete=False, suffix='.mp4')  # 삭제하지 않고, .mp4 확장자 사용
        temp_file.write(file_stream.read())
        temp_file.close()
        app.logger.info(f"(tmp)Video downloaded successfully to {temp_file.name}")
        return temp_file.name  # 파일 경로 반환
    except Exception as e:
        app.logger.error(f"Error while download tmp video: {e}")
        raise

# AWS S3 image 버킷에서 파일을 메모리로 직접 로드
def load_image_from_s3_to_tempfile(bucket, key):
    app.logger.info(f"Loading image from S3: bucket={bucket}, key={key}")
    try:
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket, Key=key)
        file_stream = response['Body']

        # 임시 파일 생성 및 파일 스트림 쓰기
        temp_file = NamedTemporaryFile(delete=False, suffix='.png')  # 삭제하지 않고, .png 확장자 사용
        temp_file.write(file_stream.read())
        temp_file.close()
        app.logger.info(f"(tmp)Image downloaded successfully to {temp_file.name}")
        return temp_file.name  # 파일 경로 반환
    except Exception as e:
        app.logger.error(f"Error while download tmp image: {e}")
        raise

def sendData(bucket, key, accident_info, headers):
    app.logger.info(f"sendData called with bucket: {bucket}, key: {key}, accident_info: {accident_info}, header: {headers}")

    # 자바 스프링 부트 서버의 URL
    url = 'http://backend-capstone.site:8080/api/accident/receiving-data'
    # accident_info를 문자열로 변환
    requestDtoStr = json.dumps(accident_info)
    # 이미지 URL에서 이미지 파일 다운로드
    image_path = load_image_from_s3_to_tempfile(bucket, key)

    # 이미지 파일이 잘 다운로드되었는지 확인
    if not os.path.exists(image_path):
        app.logger.error("Downloaded file does not exist")
        return

    file_size = os.path.getsize(image_path)
    if file_size == 0:
        app.logger.error("Downloaded file is empty")
        return
    app.logger.info(f"Downloaded file size: {file_size} bytes")

    # 멀티파트 폼 데이터 준비
    files = {
        'image': ('accident.png', open(image_path, 'rb'), 'image/png'),
        'requestDto': (None, requestDtoStr, 'application/json')
    }
    # 헤더 정보를 문자열로 변환
    headerStr = json.dumps(headers)
    # POST 요청 보내기
    response = requests.post(url, headers=headers, files=files)
    app.logger.info(f"response : {response}")
    if response.status_code == 200:
        app.logger.info("Data sent successfully")
    else:
        app.logger.error(f"Failed to send data: {response.status_code}, {response.text}")
    os.unlink(image_path)  # 임시 파일 삭제

def process_streaming_link(video_link, densenet_model, yolo_model, device, gps_info, header):
    cap = cv2.VideoCapture(video_link)
    transform = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((224, 224)), 
        transforms.ToTensor()])
    
    frame_count = 0
    accident_count = 0
    frame_skip = 1  # 초기 frame_skip 값

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break        
        if frame_count % frame_skip == 0:
            input_tensor = transform(frame).unsqueeze(0).to(device)

            with torch.no_grad(): 
                output = densenet_model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

                accident = 0 if predicted.item() == 1 else 1.
                if accident == 1:
                    accident_count += 1
                    if accident_count == 3:
                        filename = f'accident_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                        _, img_encoded = cv2.imencode('.png', frame)
                        img_bytes = img_encoded.tobytes()

                        s3_client.put_object(
                            Bucket=S3_IMG_BUCKET,
                            Key=filename,
                            Body=img_bytes,
                            ContentType='image/png'
                        )
                        # YOLO 모델로 프레임 분석
                        results_yolo = yolo_model(frame)
                        yolo_class = "unknown"
                        for result in results_yolo:
                            boxes = result.boxes
                            for box in boxes:
                                cls = int(box.cls[0].item())
                                if str(cls) in yolo_model.model.names:
                                    yolo_class = yolo_model.model.names[str(cls)]
                                    break
                            
                        # GPS 정보 추가
                        gps_data = json.loads(gps_info)
                        # 변수에 저장
                        lat = str(round(gps_data['lat'], 5))
                        lon = str(round(gps_data['lng'], 5))

                        accident_info = {
                            "accident": True,
                            "latitude": lat,
                            "longitude": lon,
                            "date": datetime.now().strftime(' %Y-%m-%d %H:%M:%S'),
                            "sorting": yolo_class,
                            "accuracy": f"{confidence.item() * 100:.2f}%"
                        }
                        app.logger.info(f"accident_info {accident_info}")
                        sendData(S3_IMG_BUCKET, filename, accident_info, header)
                        # 사고 탐지 후 5초 동안 대기
                        time.sleep(5)
                        accident_count = 0  # 사고 카운트 초기화
                    frame_skip = 1  # 사고가 발생하면 다음 프레임 검사
                else:
                    frame_skip = 5  # 사고가 없으면 5 프레임 후 검사        
        frame_count += 1
    cap.release()
    return accident_info

def process_video(bucket, key, densenet_model, yolo_model, device, gps_info, headers):
    video_path = load_video_from_s3_to_tempfile(bucket, key)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        os.unlink(video_path)  # 임시 파일 삭제
        app.logger.error("Failed to open video source")
        return ['Failed to open video source']

    fps = cap.get(cv2.CAP_PROP_FPS)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    frame_count = 0
    accident_count = 0
    frame_skip = 1
    
    while cap.isOpened() and accident_count < 3:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            input_tensor = transform(frame).unsqueeze(0).to(device)

            with torch.no_grad():
                output = densenet_model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

                accident = 0 if predicted.item() == 1 else 1
                if accident == 1:
                    accident_count += 1
                    if accident_count == 3:
                        filename = f'accident_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                        _, img_encoded = cv2.imencode('.png', frame)
                        img_bytes = img_encoded.tobytes()

                        s3_client.put_object(
                            Bucket=S3_IMG_BUCKET,
                            Key=filename,
                            Body=img_bytes,
                            ContentType='image/png'
                        )
                        # YOLO 모델로 프레임 분석
                        results_yolo = yolo_model(frame)
                        yolo_class = "unknown"
                        for result in results_yolo:
                            boxes = result.boxes
                            for box in boxes:
                                cls = int(box.cls[0].item())
                                if str(cls) in yolo_model.model.names:
                                    yolo_class = yolo_model.model.names[str(cls)]
                                    break  
                                                 
                        gps_data = json.loads(gps_info)
                        # 변수에 저장
                        lat = str(round(gps_data['lat'], 5))
                        lon = str(round(gps_data['lng'], 5))

                        accident_info = {
                            "accident": True,
                            "latitude": lat,
                            "longitude": lon,
                            "date": datetime.now().strftime(' %Y-%m-%d %H:%M:%S'),
                            "sorting": yolo_class,
                            "accuracy": f"{confidence.item() * 100:.2f}%"
                        }
                        app.logger.info(f"accident_info {accident_info}")
                        sendData(S3_IMG_BUCKET, filename, accident_info, headers)
                        os.unlink(video_path)  # 임시 파일 삭제
                        return accident_info
                else:
                    frame_skip = 5
        frame_count += 1
    cap.release()
    os.unlink(video_path)  # 임시 파일 삭제
    return accident_info    

# 비디오 링크 업로드 라우트
@app.route('/api/v1/public/upload-link', methods=['POST'])
def upload_link():
    if 'video_link' not in request.json:
        return jsonify({'error': 'No video link provided'}), 400

    video_link = request.json['video_link']
    gps_info = request.json.get('location', '')

    # 헤더에서 토큰 정보 받기
    token = request.headers.get('Authorization')
    refresh_token = request.headers.get('Refresh')

    app.logger.info(f'Token: {token}, Refresh Token: {refresh_token}, GPS Info: {gps_info}')
    densenet_model, device = load_densenet_model()
    yolo_model = load_yolo_model('YOLOv8_best2.pt')
    try:
        # 링크를 텍스트 파일로 변환
        video_link_bytes = BytesIO(video_link.encode('utf-8'))
        # 파일 이름 생성
        filename = f"video_link_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        s3_client.upload_fileobj(
            video_link_bytes,
            S3_VIDEO_BUCKET,
            filename,
            ExtraArgs={'ContentType': 'text/plain'}
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    header = {
        'token': token,
        'refresh_token': refresh_token
    }
    results = process_streaming_link(video_link, densenet_model, yolo_model, device, gps_info, header)
    app.logger.info(f'Accident Link Information: {results}')
    return jsonify(results)

# 비디오 파일 업로드 라우트
@app.route('/api/v1/public/upload-video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 100
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 200
        
    if file and allowed_file(file.filename):
        densenet_model, device = load_densenet_model()
        yolo_model = load_yolo_model('YOLOv8_best2.pt')
        filename = secure_filename(file.filename)
        # 헤더에서 토큰 정보 받기
        token = request.headers.get('Authorization')
        refresh_token = request.headers.get('Refresh')
        # GPS 정보 받기
        gps_info = request.form.get('location', '')
        #app.logger.info(f'Token: {token}, Refresh Token: {refresh_token}, GPS Info: {gps_info}')

        try:
            # 파일 내용을 읽어 S3에 저장
            s3_client.upload_fileobj(
                file,
                S3_VIDEO_BUCKET,
                filename,
                ExtraArgs={'ContentType': file.content_type}
            )
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
        # 모델 로딩 및 동영상 처리
        try:
            headers = {
                'Authorization': token,
                'Refresh': refresh_token
            }
            result = []
            result = process_video(S3_VIDEO_BUCKET, filename, densenet_model, yolo_model, device, gps_info, headers)
            return jsonify({'success': 'File processed successfully'}), 200
        except Exception as e:
            app.logger.error(f"An error occurred while processing the video at route: {e}")
            return jsonify({'error': str(e)}), 700
    else:
        return jsonify({'error': 'File not allowed or missing'}), 600
    
if __name__ == '__main__':    
    app.run(host='0.0.0.0', port=5000, debug=True)