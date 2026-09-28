"""
野球スイング解析 MVP スクリプト（第2版：Webアプリからも呼び出せる形に整理）
--------------------------------------------------------------
スマホで撮影したスイング動画(MP4/MOV)を読み込み、MediaPipeの姿勢推定(Pose Landmarker)で
全身の関節点を検出し、以下を動画に描画して保存します。

  1. 頭の位置（軸ブレ確認用）
     - 構え（最初に検出できたフレーム）の頭の位置を黄色い縦線で表示（基準軸）
     - 現在の頭の位置を赤い点で表示、直近の軌跡を黄色い線でつなぐ
     - 基準軸からのズレをピクセル数で画面左上に表示
  2. 肩のライン（青）・腰のライン（緑）
     - 各ラインの水平に対する角度（度）を画面左上に表示
     - 腰のラインの角度が「腰の回転角度」の目安になります
       （※カメラの正面/真横からの2D映像による近似値です。3Dの正確な回転角度ではありません）

■ コマンドラインから直接使う場合:
    python swing_analyzer.py [入力動画] [出力動画]

    例:
        python swing_analyzer.py input.mp4 output_analyzed.mp4

    引数を省略した場合は、同じフォルダの input.mp4 を読み込み、
    output_analyzed.mp4 として保存します。

■ 他のプログラム（Webアプリ app.py など）から使う場合:
    from swing_analyzer import analyze_video
    stats = analyze_video("input.mp4", "output.mp4", progress_callback=my_func)

初回実行時、姿勢推定用のモデルファイル(pose_landmarker_lite.task)を
自動でダウンロードします（数MB、数秒〜数十秒かかります）。
"""

import sys
import os
import math
import urllib.request

import cv2
import mediapipe as mp

# ------------------------------------------------------------
# 設定
# ------------------------------------------------------------
MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

# MediaPipe Pose のランドマーク番号（全33点のうち、今回使うものだけ）
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24

TRAIL_MAX_LEN = 40  # 頭の軌跡を何フレーム分残すか


def download_model_if_needed(path: str, url: str) -> None:
    if os.path.exists(path):
        return
    print(f"モデルファイルをダウンロード中...\n  {url}")
    urllib.request.urlretrieve(url, path)
    print("ダウンロード完了:", path)


def to_pixel(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)


def line_angle_deg(p1, p2):
    """2点を結ぶ線の、水平線に対する角度（度）。左右の傾きの目安。"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))


def analyze_video(input_path, output_path, progress_callback=None,
                   model_path=MODEL_PATH, model_url=MODEL_URL):
    """
    動画を解析し、頭の位置(軸ブレ)・肩腰ラインを描画した動画を output_path に保存する。

    progress_callback: 任意。frame_count, total_frames を引数に取る関数。
        1フレーム処理するたびに呼び出される（total_framesが0の場合は総数不明）。
        Webアプリの進捗バー更新や、CLIでのログ出力に使う。

    戻り値: {"total_frames": 処理した総フレーム数,
             "detected_frames": 姿勢を検出できたフレーム数,
             "output_path": 出力動画の絶対パス}

    入力動画が見つからない場合は FileNotFoundError、
    動画を開けなかった場合は RuntimeError を送出する。
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"入力動画が見つかりません: {input_path}")

    download_model_if_needed(model_path, model_url)

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
    )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"動画を開けませんでした: {input_path}\n"
            "（iPhoneのMOV(HEVC)形式が原因の場合があります。"
            "カメラ設定を「互換性優先」にするか、MP4に変換してから再度お試しください）"
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    head_trail = []
    address_head_x = None  # 構え（最初に検出できたフレーム）の頭のX座標＝基準軸
    frame_count = 0
    detected_count = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(frame_count * 1000 / fps)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                detected_count += 1
                lm = result.pose_landmarks[0]

                nose_px = to_pixel(lm[NOSE], width, height)
                l_sh = to_pixel(lm[LEFT_SHOULDER], width, height)
                r_sh = to_pixel(lm[RIGHT_SHOULDER], width, height)
                l_hip = to_pixel(lm[LEFT_HIP], width, height)
                r_hip = to_pixel(lm[RIGHT_HIP], width, height)

                # --- 軸ブレ（頭の位置）---
                if address_head_x is None:
                    address_head_x = nose_px[0]

                cv2.line(frame_bgr, (address_head_x, 0), (address_head_x, height),
                         (0, 255, 255), 1)  # 構えの基準線（黄色）

                head_trail.append(nose_px)
                if len(head_trail) > TRAIL_MAX_LEN:
                    head_trail.pop(0)
                for i in range(1, len(head_trail)):
                    cv2.line(frame_bgr, head_trail[i - 1], head_trail[i], (0, 255, 255), 2)

                cv2.circle(frame_bgr, nose_px, 6, (0, 0, 255), -1)  # 現在の頭位置（赤）

                deviation_px = nose_px[0] - address_head_x
                cv2.putText(frame_bgr, f"Head shift: {deviation_px:+d}px",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # --- 肩のライン（青）---
                cv2.line(frame_bgr, l_sh, r_sh, (255, 0, 0), 3)
                shoulder_angle = line_angle_deg(l_sh, r_sh)
                cv2.putText(frame_bgr, f"Shoulder angle: {shoulder_angle:.1f} deg",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                # --- 腰のライン（緑）---
                cv2.line(frame_bgr, l_hip, r_hip, (0, 255, 0), 3)
                hip_angle = line_angle_deg(l_hip, r_hip)
                cv2.putText(frame_bgr, f"Hip angle: {hip_angle:.1f} deg",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            writer.write(frame_bgr)
            frame_count += 1

            if progress_callback:
                progress_callback(frame_count, total_frames)

    cap.release()
    writer.release()

    return {
        "total_frames": frame_count,
        "detected_frames": detected_count,
        "output_path": os.path.abspath(output_path),
    }


def _cli_progress(frame_count, total_frames):
    """CLI実行時：30フレームごとに進捗をprintする（フリーズに見えないように）。"""
    if frame_count % 30 != 0:
        return
    if total_frames > 0:
        percent = frame_count / total_frames * 100
        print(f"  処理中... {frame_count}/{total_frames} フレーム ({percent:.0f}%)")
    else:
        print(f"  処理中... {frame_count} フレーム")


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output_analyzed.mp4"

    print("解析を開始します...")
    try:
        stats = analyze_video(input_path, output_path, progress_callback=_cli_progress)
    except (FileNotFoundError, RuntimeError) as e:
        print(str(e))
        return

    print(f"完了！ 全{stats['total_frames']}フレーム中、"
          f"{stats['detected_frames']}フレームで姿勢を検出しました。")
    print("保存先:", stats["output_path"])


if __name__ == "__main__":
    main()
