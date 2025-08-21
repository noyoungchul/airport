import cv2
from ultralytics import solutions
from PIL import ImageFont, ImageDraw, Image
import numpy as np

cap = cv2.VideoCapture("airport.mp4")
# ROI 영역 지정
queue_region = [(50, 650), (1200, 650), (1200, 150), (50, 150)]

queuemanager = solutions.QueueManager(
    model="yolov8m.pt",   # YOLO 모델
    region=queue_region,  # ROI 영역
    line_width=3,         # ROI 라인 두께
    show=False,           # 디버그용 내부 창 표시 여부
    conf=0.2,           # 신뢰도 50% 이상 인식
    classes=[0]          # 사람만 인식
)

MAX_CAPACITY = 25          # 혼잡도 계산 기준 최대 인원
GAUGE_MAX_WIDTH = 300     # 게이지바 최대 길이 (px)
GAUGE_HEIGHT = 25         # 게이지바 높이

#  한글 폰트 경로 (Windows 기준: 맑은고딕)
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
font = ImageFont.truetype(FONT_PATH, 32)  # 크기 32px

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break

    # 영상 크기 줄이기 (예: 50%)
    im0 = cv2.resize(im0, (0, 0), fx=0.5, fy=0.5)

    results = queuemanager(im0)

     # ROI 내 사람 수
    people_count = results.queue_count

    # 혼잡도 계산
    congestion = min(int((people_count / MAX_CAPACITY) * 100), 100)

    # 혼잡도 색상 선택
    if congestion < 50:
        color = (0, 255, 0)    # 초록
    elif congestion < 80:
        color = (0, 165, 255)  # 주황
    else:
        color = (0, 0, 255)    # 빨강

    # OpenCV 이미지 → PIL 변환
    img_pil = Image.fromarray(cv2.cvtColor(results.plot_im, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 기본 정보 표시 (한글)
    draw.text((30, 50), f"인원 수: {people_count}", font=font, fill=(0, 255, 0))
    draw.text((30, 90), f"혼잡도: {congestion}%", font=font, fill=color)

    # 경고 메시지
    if congestion >= 80:
        warning_text = "혼잡 지역입니다. 이동 경로를 분산하세요!"
        draw.rectangle((25, 140, 650, 190), fill=(255, 0, 0))  # 빨간 배경
        draw.text((30, 150), warning_text, font=font, fill=(255, 255, 255))

    # 다시 OpenCV 이미지로 변환
    results.plot_im = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 게이지바 배경
    start_x, start_y = 30, results.plot_im.shape[0] - 50
    end_x = start_x + GAUGE_MAX_WIDTH
    cv2.rectangle(results.plot_im, (start_x, start_y),
                  (end_x, start_y + GAUGE_HEIGHT), (200, 200, 200), -1)

    # 게이지바 채움 너비
    fill_width = int(GAUGE_MAX_WIDTH * (congestion / 100))

    # 게이지 채운 부분
    cv2.rectangle(results.plot_im, (start_x, start_y),
                  (start_x + fill_width, start_y + GAUGE_HEIGHT), color, -1)
    # 외곽선(테두리)
    cv2.rectangle(results.plot_im, (start_x, start_y),
                  (end_x, start_y + GAUGE_HEIGHT), (0, 0, 0), 2)

    # 결과 프레임
    cv2.imshow("Queue Monitoring", results.plot_im)
    
    # q키 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
