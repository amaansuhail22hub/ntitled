#!/usr/bin/env bash
# Full rebuild of the ACHERON "Ahead of Time" intro at 4K.
#   ./make.sh            TTS placeholder VO
#   ./make.sh --human    real read from build/vo_human/NN.wav (see README)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build out/stems

python3 audio.py "$@"
python3 render.py --scale 2 --out build/video_4k_intermediate.mov

# Delivery master: H.264 High 5.1, 2160x3840, 24 fps, BT.709, AAC 320k.
ffmpeg -hide_banner -loglevel error -y \
  -i build/video_4k_intermediate.mov -i build/audio_master.wav \
  -map 0:v -map 1:a \
  -c:v libx264 -preset slow -tune grain -crf 18 -maxrate 22M -bufsize 44M \
  -profile:v high -level 5.1 -pix_fmt yuv420p \
  -vf "scale=out_color_matrix=bt709:out_range=tv" \
  -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
  -c:a aac -b:a 320k -ar 48000 \
  -movflags +faststart -shortest \
  out/ACHERON_AheadOfTime_4K.mp4

cp build/stems/vo.wav out/stems/vo.wav
cp build/stems/fx.wav out/stems/music_fx_no_vo.wav
ffprobe -hide_banner -v error -show_entries stream=codec_name,width,height,r_frame_rate,bit_rate -of compact out/ACHERON_AheadOfTime_4K.mp4
