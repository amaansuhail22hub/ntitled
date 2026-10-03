#!/usr/bin/env bash
# Full rebuild of both ACHERON "Ahead of Time" cuts at 4K.
#   site:  3.5 s headphones card + 45 s intro (Shopify hero, YouTube)
#   reels: hook-first cut for Instagram, stereo heartbeat open, burned-in captions
#
#   ./make.sh            TTS placeholder VO
#   ./make.sh --human    real read from build/vo_human/NN.wav (see README)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build out/stems

python3 audio.py "$@"        # writes build/audio_master.wav and build/audio_master_reels.wav
python3 render.py --cut site --scale 2 --out build/video_4k_intermediate.mov
python3 render.py --cut reels --scale 2 --out build/video_4k_reels.mov

# H.264 High 5.1, 2160x3840, 24 fps, BT.709, AAC 320k.
#   crf/maxrate 18/22M for upload masters, 19/14M for the copies kept in git (<100 MB)
enc() {  # crf maxrate bufsize video audio out
  ffmpeg -hide_banner -loglevel error -y -i "$4" -i "$5" -map 0:v -map 1:a \
    -c:v libx264 -preset slow -tune grain -crf "$1" -maxrate "$2" -bufsize "$3" \
    -profile:v high -level 5.1 -pix_fmt yuv420p \
    -vf "scale=out_color_matrix=bt709:out_range=tv" \
    -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
    -c:a aac -b:a 320k -ar 48000 -movflags +faststart -shortest "$6"
}

# Small 4K HEVC copy, two-pass, sized to stay under a 30 MB upload limit.
hevc() {  # video audio out
  local X="-c:v libx265 -preset medium -b:v 4500k -pix_fmt yuv420p -tag:v hvc1 -color_primaries bt709 -color_trc bt709 -colorspace bt709"
  local P="log-level=error:aq-mode=3:psy-rd=2.0:psy-rdoq=1.0:stats=build/x265.log"
  ffmpeg -hide_banner -loglevel error -y -i "$1" -vf "scale=out_color_matrix=bt709:out_range=tv" \
    $X -x265-params "pass=1:$P" -an -f mp4 /dev/null
  ffmpeg -hide_banner -loglevel error -y -i "$1" -i "$2" -map 0:v -map 1:a \
    -vf "scale=out_color_matrix=bt709:out_range=tv" $X -x265-params "pass=2:$P" \
    -c:a aac -b:a 192k -movflags +faststart -shortest "$3"
}

SITE_V=build/video_4k_intermediate.mov; SITE_A=build/audio_master.wav
REELS_V=build/video_4k_reels.mov;       REELS_A=build/audio_master_reels.wav

enc 18 22M 44M $SITE_V  $SITE_A  out/ACHERON_AheadOfTime_4K_master.mp4
enc 19 14M 28M $SITE_V  $SITE_A  out/ACHERON_AheadOfTime_4K.mp4
hevc           $SITE_V  $SITE_A  out/ACHERON_AheadOfTime_4K_HEVC.mp4

enc 18 22M 44M $REELS_V $REELS_A out/ACHERON_AheadOfTime_Reels_4K_master.mp4
enc 19 14M 28M $REELS_V $REELS_A out/ACHERON_AheadOfTime_Reels_4K.mp4
hevc           $REELS_V $REELS_A out/ACHERON_AheadOfTime_Reels_4K_HEVC.mp4

cp build/stems/vo.wav out/stems/vo.wav
cp build/stems/fx.wav out/stems/music_fx_no_vo.wav
ffprobe -hide_banner -v error -show_entries stream=codec_name,width,height,r_frame_rate,bit_rate -of compact out/*.mp4
