import socket
import cv2
import numpy as np
import pickle
import struct
import datetime
import os
import atexit
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import firebase_admin
from firebase_admin import credentials, db, storage

# 설정 상수들
CONFIDENCE_THRESHOLD = 0.8  # 신뢰도 임계값
GREEN = (0, 255, 0)  # 초록색 (Bounding Box 색상)
WHITE = (255, 255, 255)  # 흰색 (텍스트 색상)

# 캡쳐된 이미지 저장 디렉토리 설정
CAPTURE_DIR = r"/home/park/PyProject/smart_refrigerator/captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)  # 캡처된 이미지를 저장할 디렉토리 생성

# 전체 프레임 이미지 저장 디렉토리 설정
FULL_FRAME_CAPTURE_DIR = r"/home/park/PyProject/smart_refrigerator/full_frame_captures"
os.makedirs(FULL_FRAME_CAPTURE_DIR, exist_ok=True)  # 전체 프레임 이미지를 저장할 디렉토리 생성

# 트래킹된 객체들의 캡쳐 상태를 저장할 딕셔너리
captured_objects = {}  # {트랙 ID: 캡처된 이미지 경로}

# 클래스 이름들을 저장할 딕셔너리
class_names = []
with open(r"/home/park/PyProject/smart_refrigerator/classes.txt", "r") as f:
    class_names = f.read().strip().split("\n")  # 클래스 이름들을 파일에서 읽어옴

# 커스텀 학습된 YOLOv8 모델 로드
model = YOLO(r"/home/park/PyProject/smart_refrigerator/model/best_4.pt")
# DeepSort 트래커 설정
tracker = DeepSort(max_age=50, n_init=3, max_iou_distance=0.7)  # 트래커 설정 조정

# 각 클래스 별로 트랙 ID를 관리하기 위한 딕셔너리
class_id_counters = {class_name: 1 for class_name in class_names}  # 클래스별 트랙 ID 카운터 초기화
track_id_map = {}  # {트랙 ID: 클래스 이름 + 카운터}

# Firebase 인증 및 초기화
cred = credentials.Certificate(r"/home/park/PyProject/smart_refrigerator/rasptoapp-firebase-adminsdk-5je3c-9439d363b8.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://rasptoapp-default-rtdb.asia-southeast1.firebasedatabase.app/',
    'storageBucket': 'rasptoapp.appspot.com'
})

# 서버 설정
HOST = '0.0.0.0'  # 모든 인터페이스에서 수신
PORT = 9999  # 포트 번호

# 소켓 생성
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(5)
print("Server listening on {}:{}".format(HOST, PORT))

def process_frame(frame):
    detections = model(frame)[0]  # 모델을 사용하여 프레임에서 객체 감지
    results = []
    for data in detections.boxes.data.tolist():  # 감지된 객체들을 반복
        confidence = data[4]  # 신뢰도 값 추출
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue  # 신뢰도가 낮으면 건너뜀
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])  # 경계 상자 좌표 추출
        class_id = int(data[5])  # 클래스 ID 추출
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])  # 결과 리스트에 추가
    return results

def update_tracker(results, frame):
    global class_id_counters
    tracks = tracker.update_tracks(results, frame=frame)  # 트래커 업데이트
    current_tracked_ids = set()
    for track in tracks:
        if not track.is_confirmed():
            continue  # 트랙이 확정되지 않은 경우 건너뜀
        track_id = track.track_id
        class_id = track.get_det_class()
        class_name = class_names[class_id]
        if track_id not in track_id_map:
            track_id_map[track_id] = f"{class_name}{class_id_counters[class_name]}"
            class_id_counters[class_name] += 1  # 클래스별 트랙 ID 증가
        assigned_track_id = track_id_map[track_id]
        ltrb = track.to_ltrb()
        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        current_tracked_ids.add(assigned_track_id)
        draw_tracking_info(frame, xmin, ymin, xmax, ymax, assigned_track_id, class_id)
        if assigned_track_id not in captured_objects:
            capture_object(frame, xmin, ymin, xmax, ymax, assigned_track_id, class_name)
    return current_tracked_ids

def draw_tracking_info(frame, xmin, ymin, xmax, ymax, track_id, class_id):
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)  # 경계 상자 그림
    label = f"{track_id}"  # 트랙 ID 레이블
    cv2.rectangle(frame, (xmin, ymin - 20), (xmin + len(label) * 10, ymin), GREEN, -1)  # 레이블 배경 그림
    cv2.putText(frame, label, (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)  # 레이블 텍스트 그림

def capture_object(frame, xmin, ymin, xmax, ymax, track_id, class_name):
    bbox_img = frame[ymin:ymax, xmin:xmax]  # 경계 상자 영역의 이미지 추출
    if bbox_img.size != 0:
        if class_name in ['apple', 'egg', 'tomato']:
            capture_path = os.path.join(CAPTURE_DIR, f"{class_name}_{track_id}.jpg")
        elif class_name in ['bottle', 'plastic_bag']:
            capture_path = os.path.join(CAPTURE_DIR, f"{class_name}_{track_id}.jpg")
        else:
            capture_path = os.path.join(CAPTURE_DIR, f"track_{track_id}.jpg")
        cv2.imwrite(capture_path, bbox_img)  # 이미지 저장
        captured_objects[track_id] = capture_path  # 캡처된 객체 기록
        print(f"Captured {class_name} with track ID {track_id} at {capture_path}")

def cleanup_tracked_objects(current_tracked_ids):
    for track_id in list(captured_objects.keys()):
        if track_id not in current_tracked_ids:
            if os.path.exists(captured_objects[track_id]):
                os.remove(captured_objects[track_id])  # 이미지 파일 삭제
                print(f"Removed capture for track ID {track_id}")
            del captured_objects[track_id]  # 딕셔너리에서 삭제

def capture_full_frame(frame):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    capture_path = os.path.join(FULL_FRAME_CAPTURE_DIR, f"full_frame_{timestamp}.jpg")
    cv2.imwrite(capture_path, frame)
    print(f"Captured full frame at {capture_path}")

def check_firebase_trigger():
    ref = db.reference('camera-control/trigger')
    return ref.get()

def reset_firebase_trigger():
    ref = db.reference('camera-control/trigger')
    ref.set(False)

def cleanup_directories():
    # 디렉토리 내의 모든 파일 삭제
    for directory in [CAPTURE_DIR, FULL_FRAME_CAPTURE_DIR]:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

atexit.register(cleanup_directories)

while True:
    client_socket, addr = server_socket.accept()
    print('Connection from:', addr)
    data = b""
    payload_size = struct.calcsize("Q")
    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(4*1024)  # 4K chunks
            if not packet: break
            data += packet
        if len(data) < payload_size:
            break
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += client_socket.recv(4*1024)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)

        start = datetime.datetime.now()
        results = process_frame(frame)  # 프레임에서 객체 감지
        current_tracked_ids = update_tracker(results, frame)  # 트래커 업데이트 및 캡처
        if check_firebase_trigger():
            capture_full_frame(frame)
            reset_firebase_trigger()
        end = datetime.datetime.now()
        cleanup_tracked_objects(current_tracked_ids)  # 트래킹되지 않는 객체 삭제
        print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")

        cv2.imshow("Frame", frame)  # 프레임 출력
        if cv2.waitKey(1) == ord('q'):
            break
    client_socket.close()
cv2.destroyAllWindows()

