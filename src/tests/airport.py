import cv2
from ultralytics import solutions

cap = cv2.VideoCapture("walking.avi")

# ROI 영역 지정
queue_region = [(100, 500), (1000, 500), (1000, 200), (100, 200)]


queuemanager = solutions.QueueManager(
    model="yolo11n.pt",   # YOLO 모델
    region=queue_region,  # ROI 영역
    line_width=3,         # ROI 라인 두께
    show=False,           # 디버그용 내부 창 표시 여부
    #conf=0.5,             # 신뢰도 50% 이상 인식
    #classes=[0]           # 사람만 인식
)

MAX_CAPACITY = 6  # 혼잡도 계산 기준 최대 인원
GAUGE_MAX_WIDTH = 300  # 게이지바 최대 길이 (px)
GAUGE_HEIGHT = 25      # 게이지바 높이

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break

    results = queuemanager(im0)

    # ROI 내 사람 수
    people_count = results.queue_count

    # 혼잡도 계산
    congestion = min(int((people_count / MAX_CAPACITY) * 100), 100)

    # 혼잡도 색상 선택
    if congestion < 50:
        color = (0, 255, 0)   # 초록 (여유)
    elif congestion < 80:
        color = (0, 165, 255) # 주황 (보통)
    else:
        color = (0, 0, 255)   # 빨강 (혼잡)

    # 혼잡도 텍스트
    cv2.putText(results.plot_im, f"People: {people_count}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(results.plot_im, f"Congestion: {congestion}%", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    if congestion >= 80:
        cv2.putText(results.plot_im, "WARNING: High congestion!", (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # 게이지바 배경
    start_x, start_y = 30, results.plot_im.shape[0] - 50
    end_x = start_x + GAUGE_MAX_WIDTH
    cv2.rectangle(results.plot_im, (start_x, start_y),
                  (end_x, start_y + GAUGE_HEIGHT), (200, 200, 200), -1)

    # 게이지바 채우기
    fill_width = int(GAUGE_MAX_WIDTH * (congestion / 100))
    cv2.rectangle(results.plot_im, (start_x, start_y),
                  (start_x + fill_width, start_y + GAUGE_HEIGHT), color, -1)

    # 테두리
    cv2.rectangle(results.plot_im, (start_x, start_y),
                  (end_x, start_y + GAUGE_HEIGHT), (0, 0, 0), 2)

    # 단일 창 출력
    cv2.imshow("Queue Monitoring", results.plot_im)

    # 창 종료
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()