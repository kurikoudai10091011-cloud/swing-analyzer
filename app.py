"""
野球スイング解析 Webアプリ（Phase 1: Web化）
--------------------------------------------
swing_analyzer.py の解析ロジックを、ブラウザから使えるようにする
Streamlit製のシンプルなWeb UI。

■ 使い方（ローカルで動かす場合）:
    pip install -r requirements.txt
    streamlit run app.py

    ブラウザが自動で開き、http://localhost:8501 でアプリが表示されます。
"""

import os
import subprocess
import tempfile

import imageio_ffmpeg
import streamlit as st

from swing_analyzer import analyze_video

st.set_page_config(page_title="スイング解析", page_icon="⚾")

st.title("⚾ スイング解析（軸ブレ・肩腰チェック）")
st.write(
    "スマホで撮ったスイング動画をアップロードすると、"
    "頭の位置（軸ブレ）と肩・腰のラインを自動で描画します。"
)
st.caption(
    "※ 単眼カメラでの2D解析です。カメラをスイング平面に正対させて撮影すると精度が上がります。"
)

uploaded_file = st.file_uploader("スイング動画を選択（MP4 / MOV）", type=["mp4", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("解析開始", type="primary"):
        with tempfile.TemporaryDirectory() as tmpdir:
            # アップロードされた動画を一時フォルダに保存
            suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
            input_path = os.path.join(tmpdir, f"input{suffix}")
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            raw_output_path = os.path.join(tmpdir, "raw_output.mp4")

            progress_bar = st.progress(0, text="解析を準備しています...")

            def on_progress(frame_count, total_frames):
                # 更新しすぎると画面がちらつくので、5フレームごとに間引く
                if frame_count % 5 != 0:
                    return
                if total_frames > 0:
                    pct = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(
                        pct, text=f"解析中... {frame_count}/{total_frames} フレーム"
                    )
                else:
                    progress_bar.progress(
                        0, text=f"解析中... {frame_count} フレーム"
                    )

            try:
                with st.spinner(
                    "姿勢推定モデルを準備しています（初回はダウンロードに時間がかかります）..."
                ):
                    stats = analyze_video(
                        input_path, raw_output_path, progress_callback=on_progress
                    )
            except (FileNotFoundError, RuntimeError) as e:
                progress_bar.empty()
                st.error(str(e))
                st.stop()

            progress_bar.progress(1.0, text="解析完了！")

            # ブラウザでの再生互換性を上げるため、H.264に変換する。
            # OpenCVで書き出したmp4v形式はブラウザで再生できないことが多いため、
            # imageio-ffmpeg（ffmpeg本体を自動同梱するパッケージ）で変換する。
            final_output_path = raw_output_path
            try:
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                web_output_path = os.path.join(tmpdir, "output_web.mp4")
                result = subprocess.run(
                    [
                        ffmpeg_exe, "-y", "-i", raw_output_path,
                        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", web_output_path,
                    ],
                    capture_output=True,
                )
                if result.returncode == 0 and os.path.exists(web_output_path):
                    final_output_path = web_output_path
                else:
                    st.warning(
                        "動画のブラウザ再生用変換に失敗しました。"
                        "ダウンロードした動画はPC上のプレイヤーで再生してください。"
                    )
            except Exception:
                st.warning(
                    "動画のブラウザ再生用変換をスキップしました。"
                    "ダウンロードした動画はPC上のプレイヤーで再生してください。"
                )

            with open(final_output_path, "rb") as f:
                video_bytes = f.read()

            st.success(
                f"解析完了！ 全{stats['total_frames']}フレーム中、"
                f"{stats['detected_frames']}フレームで姿勢を検出しました。"
            )
            st.video(video_bytes)
            st.download_button(
                "解析済み動画をダウンロード",
                data=video_bytes,
                file_name="output_analyzed.mp4",
                mime="video/mp4",
            )

            if stats["detected_frames"] == 0:
                st.warning(
                    "姿勢を検出できたフレームがありませんでした。"
                    "全身が画角に収まっているか、明るさが十分かを確認して"
                    "撮り直してみてください。"
                )
