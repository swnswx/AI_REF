import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
import pandas as pd
from paddleocr import PaddleOCR
from threading import Thread, Event
import firebase_admin
from firebase_admin import credentials, db, storage
from datetime import datetime

# Firebase 인증 및 초기화
cred = credentials.Certificate(r"/home/park/PyProject/smart_refrigerator/rasptoapp-firebase-adminsdk-5je3c-9439d363b8.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://rasptoapp-default-rtdb.asia-southeast1.firebasedatabase.app/',
    'storageBucket': 'rasptoapp.appspot.com'
})

# OCR 결과를 저장할 리스트 초기화
results = []

# OCR 모델 로드
ocr = PaddleOCR(use_angle_cls=True, lang='korean')

# captures 디렉토리 설정
CAPTURE_DIR = r"/home/park/PyProject/smart_refrigerator/captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# full_frame_captures 디렉토리 설정
FULL_FRAME_CAPTURE_DIR = r"/home/park/PyProject/smart_refrigerator/full_frame_captures"
os.makedirs(FULL_FRAME_CAPTURE_DIR, exist_ok=True)

# 클래스별 제품 이름 매핑
CLASS_NAME_TO_PRODUCT_NAME = {
    'apple': '사과',
    'egg': '계란'
}

def make_safe_filename(filename):
    # 파일 이름에서 사용할 수 없는 문자를 밑줄로 대체
    return filename.replace('/', '_').replace('.', '_').replace('#', '_').replace('$', '_').replace('[', '_').replace(']', '_')

# Firebase 스토리지에 이미지 업로드
def upload_to_firebase_storage(local_path, storage_path):
    bucket = storage.bucket()
    blob = bucket.blob(storage_path)
    blob.upload_from_filename(local_path)
    blob.make_public()
    return blob.public_url

# Firebase 스토리지에서 이미지 삭제
def delete_from_firebase_storage(storage_path):
    bucket = storage.bucket()
    blob = bucket.blob(storage_path)
    blob.delete()

# 파일 모니터링 및 OCR 처리 함수
def ocr_processing_thread(stop_event):
    processed_files = set()
    while not stop_event.is_set():
        current_files = set(os.listdir(CAPTURE_DIR))
        new_files = current_files - processed_files
        removed_files = processed_files - current_files

        for filename in new_files:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(CAPTURE_DIR, filename)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Failed to load image: {filename}")
                    continue

                try:
                    ocr_result = ocr.ocr(image, cls=True)
                except Exception as e:
                    print(f"Failed to process OCR for {filename}: {e}")
                    continue

                texts = [filename, "text"]
                confidences = [filename, "confidence"]
                text_data = []
                confidence_data = []
                product_name = None
                for line in ocr_result:
                    if line is None:
                        continue
                    for word_info in line:
                        texts.append(word_info[1][0])
                        confidences.append(word_info[1][1])
                        text_data.append(word_info[1][0])
                        confidence_data.append(word_info[1][1])
                
                # 파일명에서 클래스 이름 추출
                class_name = filename.split('_')[0].lower()
                if class_name in CLASS_NAME_TO_PRODUCT_NAME:
                    product_name = CLASS_NAME_TO_PRODUCT_NAME[class_name]

                results.append(texts)
                results.append(confidences)

                # Firebase에 데이터 저장
                safe_filename = make_safe_filename(filename)
                ref = db.reference(f'ocr_results/{safe_filename}')
                ref.set({
                    'texts': text_data,
                    'confidences': confidence_data,
                    'productName': product_name
                })

                # 이미지 Firebase 스토리지에 업로드
                storage_path = f'captures/{safe_filename}'
                image_url = upload_to_firebase_storage(image_path, storage_path)
                ref.update({'image_url': image_url})

                print(f"Processed and uploaded {filename}")

        for filename in removed_files:
            safe_filename = make_safe_filename(filename)
            ref = db.reference(f'ocr_results/{safe_filename}')
            ref.delete()

            # 이미지 Firebase 스토리지에서 삭제
            storage_path = f'captures/{safe_filename}'
            delete_from_firebase_storage(storage_path)

            print(f"{filename} removed from {CAPTURE_DIR} and Firebase")

        processed_files = current_files

        # 결과를 DataFrame으로 변환하여 CSV 파일로 저장
        df = pd.DataFrame(results)
        csv_path = r"/home/park/PyProject/smart_refrigerator/ocr_results_combined.csv"
        df.to_csv(csv_path, index=False, header=False, encoding='utf-8-sig')

        stop_event.wait(1)

    print(f"OCR 결과가 {csv_path}에 저장되었습니다.")

# full_frame_captures 디렉토리의 사진을 Firebase 스토리지에 업로드하고 최신 사진만 남기는 함수
def upload_full_frame_captures(stop_event):
    while not stop_event.is_set():
        for filename in os.listdir(FULL_FRAME_CAPTURE_DIR):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                local_path = os.path.join(FULL_FRAME_CAPTURE_DIR, filename)
                storage_path = f'full_capture/{make_safe_filename(filename)}'
                
                try:
                    image_url = upload_to_firebase_storage(local_path, storage_path)
                    os.remove(local_path)  # 업로드 후 파일 삭제
                    print(f"Uploaded and deleted {filename} to Firebase at {image_url}")

                    # Firebase 스토리지에서 이전 파일 삭제
                    delete_old_files_from_firebase_storage('full_capture')
                except Exception as e:
                    print(f"Failed to upload {filename} to Firebase: {e}")
        
        stop_event.wait(10)  # 10초마다 디렉토리 확인

def delete_old_files_from_firebase_storage(folder_name):
    bucket = storage.bucket()
    blobs = list(bucket.list_blobs(prefix=folder_name))
    if len(blobs) > 1:
        # 가장 최신 파일을 제외한 모든 파일 삭제
        latest_blob = max(blobs, key=lambda b: b.time_created)
        for blob in blobs:
            if blob != latest_blob:
                print(f"Deleting old file from Firebase: {blob.name}")
                blob.delete()

if __name__ == "__main__":
    stop_event = Event()
    ocr_thread = Thread(target=ocr_processing_thread, args=(stop_event,))
    upload_thread = Thread(target=upload_full_frame_captures, args=(stop_event,))
    
    ocr_thread.start()
    upload_thread.start()
    
    try:
        while True:
            pass
    except KeyboardInterrupt:
        stop_event.set()
    
    ocr_thread.join()
    upload_thread.join()

