#!/usr/bin/env bash
# Full rebuild of the ACHERON "Ahead of Time" intro at 4K.
#   ./make.sh            TTS placeholder VO
#   ./make.sh --human    real read from build/vo_human/NN.wav (see README)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build out/stems

python3 audio.py "$@"
python3 render.py --scale 2 --out build/video_4k_intermediate.mov

# Two H.264 High 5.1 encodes, 2160x3840, 24 fps, BT.709, AAC 320k:
#   _master  ~22 Mbps (~127 MB) for upload to Instagram / Shopify
#   plain    ~14 Mbps (~77 MB), fits under GitHub's 100 MB file limit
enc() {
  ffmpeg -hide_banner -loglevel error -y \
    -i build/video_4k_intermediate.mov -i build/audio_master.wav \
    -map 0:v -map 1:a \
    -c:v libx264 -preset slow -tune grain -crf "$1" -maxrate "$2" -bufsize "$3" \
    -profile:v high -level 5.1 -pix_fmt yuv420p \
    -vf "scale=out_color_matrix=bt709:out_range=tv" \
    -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
    -c:a aac -b:a 320k -ar 48000 \
    -movflags +faststart -shortest "$4"
}
enc 18 22M 44M out/ACHERON_AheadOfTime_4K_master.mp4
enc 19 14M 28M out/ACHERON_AheadOfTime_4K.mp4

# Small 4K HEVC copy, two-pass, sized to stay under a 30 MB upload limit.
X265="-c:v libx265 -preset medium -b:v 4500k -pix_fmt yuv420p -tag:v hvc1 -color_primaries bt709 -color_trc bt709 -colorspace bt709"
P265="log-level=error:aq-mode=3:psy-rd=2.0:psy-rdoq=1.0:stats=build/x265.log"
ffmpeg -hide_banner -loglevel error -y -i build/video_4k_intermediate.mov -vf "scale=out_color_matrix=bt709:out_range=tv" \
  $X265 -x265-params "pass=1:$P265" -an -f mp4 /dev/null
ffmpeg -hide_banner -loglevel error -y -i build/video_4k_intermediate.mov -i build/audio_master.wav -map 0:v -map 1:a \
  -vf "scale=out_color_matrix=bt709:out_range=tv" $X265 -x265-params "pass=2:$P265" \
  -c:a aac -b:a 192k -movflags +faststart -shortest out/ACHERON_AheadOfTime_4K_HEVC.mp4

cp build/stems/vo.wav out/stems/vo.wav
cp build/stems/fx.wav out/stems/music_fx_no_vo.wav
ffprobe -hide_banner -v error -show_entries stream=codec_name,width,height,r_frame_rate,bit_rate -of compact out/ACHERON_AheadOfTime_4K.mp4
