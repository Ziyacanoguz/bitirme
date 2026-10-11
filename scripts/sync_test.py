import argparse
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="İki kamera videosunu birlikte incele")
    parser.add_argument(
        "--camera-1",
        type=Path,
        default=PROJECT_ROOT / "assets/videos/camera1.mp4",
    )
    parser.add_argument(
        "--camera-2",
        type=Path,
        default=PROJECT_ROOT / "assets/videos/camera2.mp4",
    )
    return parser.parse_args()


args = parse_args()

# =========================
# VIDEO DOSYALARI
# =========================

video1 = cv2.VideoCapture(str(args.camera_1))
video2 = cv2.VideoCapture(str(args.camera_2))

if not video1.isOpened():
    print("Kamera 1 videosu açılamadı.")
    exit()

if not video2.isOpened():
    print("Kamera 2 videosu açılamadı.")
    exit()


# =========================
# VIDEO BİLGİLERİ
# =========================

fps1 = video1.get(cv2.CAP_PROP_FPS)
fps2 = video2.get(cv2.CAP_PROP_FPS)

frames1 = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
frames2 = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"CAM 1 FPS: {fps1}")
print(f"CAM 2 FPS: {fps2}")

print(f"CAM 1 Frame Sayısı: {frames1}")
print(f"CAM 2 Frame Sayısı: {frames2}")

print("\n--- KONTROLLER ---")
print("SPACE = Durdur / Devam")
print("N     = Durmuşken 1 frame ilerle")
print("1     = 5. dakikaya git")
print("2     = 30. dakikaya git")
print("3     = 55. dakikaya git")
print("D     = 10 saniye ileri")
print("A     = 10 saniye geri")
print("E     = 1 dakika ileri")
print("W     = 1 dakika geri")
print("P     = Anlık zaman farkını yazdır")
print("Q     = Çık")


# =========================
# BAŞLANGIÇ
# =========================

paused = False

ret1, frame1 = video1.read()
ret2, frame2 = video2.read()

if not ret1 or not ret2:
    print("Videolar okunamadı.")
    exit()


# =========================
# ANA DÖNGÜ
# =========================

while True:

    if not paused:
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        if not ret1 or not ret2:
            print("Videolardan biri sona ulaştı.")
            break

    # Anlık zaman bilgileri
    time1 = video1.get(cv2.CAP_PROP_POS_MSEC)
    time2 = video2.get(cv2.CAP_PROP_POS_MSEC)

    frame_num1 = int(video1.get(cv2.CAP_PROP_POS_FRAMES))
    frame_num2 = int(video2.get(cv2.CAP_PROP_POS_FRAMES))

    # Görüntüleri küçült
    display1 = cv2.resize(frame1, (640, 360))
    display2 = cv2.resize(frame2, (640, 360))

    # CAM 1 bilgisi
    cv2.putText(
        display1,
        f"CAM1 | Frame: {frame_num1} | {time1 / 1000:.3f}s",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # CAM 2 bilgisi
    cv2.putText(
        display2,
        f"CAM2 | Frame: {frame_num2} | {time2 / 1000:.3f}s",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Pause yazısı
    if paused:
        cv2.putText(
            display1,
            "PAUSED",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display2,
            "PAUSED",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # İki görüntüyü yan yana birleştir
    combined = cv2.hconcat([display1, display2])

    cv2.imshow("CAM 1 | CAM 2", combined)

    # Oynuyorsa yaklaşık 25 FPS
    # Pause durumundaysa tuşa basılana kadar bekle
    key = cv2.waitKey(40 if not paused else 0) & 0xFF


    # =========================
    # Q = ÇIKIŞ
    # =========================

    if key == ord("q"):
        break


    # =========================
    # SPACE = PAUSE / DEVAM
    # =========================

    elif key == ord(" "):

        paused = not paused

        print("\n--- PAUSE / PLAY ---")
        print(f"CAM1 Frame: {frame_num1} | {time1 / 1000:.3f}s")
        print(f"CAM2 Frame: {frame_num2} | {time2 / 1000:.3f}s")
        print(f"Timestamp farkı: {abs(time1 - time2):.2f} ms")


    # =========================
    # N = 1 FRAME İLERLE
    # =========================

    elif key == ord("n") and paused:

        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        if not ret1 or not ret2:
            print("Video sonuna ulaşıldı.")
            break


    # =========================
    # 1 = 5. DAKİKA
    # =========================

    elif key == ord("1"):

        target_ms = 5 * 60 * 1000

        video1.set(cv2.CAP_PROP_POS_MSEC, target_ms)
        video2.set(cv2.CAP_PROP_POS_MSEC, target_ms)

        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        print("\n5. dakikaya gidildi.")


    # =========================
    # 2 = 30. DAKİKA
    # =========================

    elif key == ord("2"):

        target_ms = 30 * 60 * 1000

        video1.set(cv2.CAP_PROP_POS_MSEC, target_ms)
        video2.set(cv2.CAP_PROP_POS_MSEC, target_ms)

        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        print("\n30. dakikaya gidildi.")


    # =========================
    # 3 = 55. DAKİKA
    # =========================

    elif key == ord("3"):

        target_ms = 55 * 60 * 1000

        video1.set(cv2.CAP_PROP_POS_MSEC, target_ms)
        video2.set(cv2.CAP_PROP_POS_MSEC, target_ms)

        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        print("\n55. dakikaya gidildi.")


    # =========================
    # D = 10 SANİYE İLERİ
    # =========================

    elif key == ord("d"):

        current_time = video1.get(cv2.CAP_PROP_POS_MSEC)

        target_ms = current_time + 10_000

        video1.set(cv2.CAP_PROP_POS_MSEC, target_ms)
        video2.set(cv2.CAP_PROP_POS_MSEC, target_ms)

        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        print(f"\n10 saniye ileri → {target_ms / 1000:.1f} saniye")


    # =========================
    # A = 10 SANİYE GERİ
    # =========================

    elif key == ord("a"):

        current_time = video1.get(cv2.CAP_PROP_POS_MSEC)

        target_ms = max(0, current_time - 10_000)

        video1.set(cv2.CAP_PROP_POS_MSEC, target_ms)
        video2.set(cv2.CAP_PROP_POS_MSEC, target_ms)

        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        print(f"\n10 saniye geri → {target_ms / 1000:.1f} saniye")


    # =========================
    # E = 1 DAKİKA İLERİ
    # =========================

    elif key == ord("e"):

        current_time = video1.get(cv2.CAP_PROP_POS_MSEC)

        target_ms = current_time + 60_000

        video1.set(cv2.CAP_PROP_POS_MSEC, target_ms)
        video2.set(cv2.CAP_PROP_POS_MSEC, target_ms)

        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        print(f"\n1 dakika ileri → {target_ms / 1000:.1f} saniye")


    # =========================
    # W = 1 DAKİKA GERİ
    # =========================

    elif key == ord("w"):

        current_time = video1.get(cv2.CAP_PROP_POS_MSEC)

        target_ms = max(0, current_time - 60_000)

        video1.set(cv2.CAP_PROP_POS_MSEC, target_ms)
        video2.set(cv2.CAP_PROP_POS_MSEC, target_ms)

        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        print(f"\n1 dakika geri → {target_ms / 1000:.1f} saniye")


    # =========================
    # P = ANLIK FARK
    # =========================

    elif key == ord("p"):

        print(
            f"\nCAM1: {time1 / 1000:.3f} sn | "
            f"CAM2: {time2 / 1000:.3f} sn | "
            f"Fark: {abs(time1 - time2):.2f} ms"
        )


# =========================
# KAPAT
# =========================

video1.release()
video2.release()

cv2.destroyAllWindows()
