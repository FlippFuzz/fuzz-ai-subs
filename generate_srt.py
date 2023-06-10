import json
import logging
import os
import sys
from importlib.metadata import version

import git
import pysubs2
import yt_dlp
from faster_whisper import download_model, WhisperModel

# SETTINGS
VIDEO_ID = sys.argv[1]
MODEL_SIZE = "tiny"
COMPUTE_TYPE = "float32"
AUDIO_DIR = "tmp_audio"
OUTPUT_DIR = "tmp_output"

DISCLAIMER_TIME = 30000
DISCLAIMER = 'DISCLAIMER: Subtitles are machine generated.\n'
DISCLAIMER += 'There is no guarantee that the translation is accurate or correct.'

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
log = logging.getLogger("fuzz-ai-subs")

# Get Video Details
log.info(f"Retrieving Video Metadata")
metadata = None
with yt_dlp.YoutubeDL() as ydl:
    metadata = ydl.extract_info(f"https://www.youtube.com/watch?v={VIDEO_ID}", download=False)

# Prepare the tmp_output filename
uploader_id = metadata['uploader_id'][1:]  # Get rid of the @
release_timestamp = metadata['release_timestamp']
if release_timestamp is None:
    release_timestamp = metadata['upload_date']  # It seems that only livestreams have release-timestamps

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
short_sha = repo.git.rev_parse(sha, short=7)

output_filename = f"{uploader_id}-{release_timestamp}-{VIDEO_ID}-FuzzAiSubs-{short_sha}.srt"
log.info(f"SRT tmp_output filename will be {output_filename}")

audio_file_location = os.path.join(AUDIO_DIR, f"{VIDEO_ID}.webm")
log.info(f"Downloading tmp_audio to {audio_file_location}")

ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': audio_file_location
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download(f"https://www.youtube.com/watch?v={VIDEO_ID}")

# Use faster-whisper to translate
log.info(f"About to translate {audio_file_location}")
model_dir = download_model(MODEL_SIZE)
model = WhisperModel(model_dir, compute_type=COMPUTE_TYPE)

segments, info = model.transcribe(audio_file_location, language="ja", task="translate",
                                  beam_size=1,
                                  vad_filter=True, vad_parameters=dict(min_silence_duration_ms=2000))

# Loop through generated translations
subs = pysubs2.SSAFile()
for segment in segments:
    text = segment.text.strip()

    # Output to screen
    log.info("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    # Save to SRT
    subs.append(pysubs2.SSAEvent(start=pysubs2.make_time(s=segment.start),
                                 end=pysubs2.make_time(s=segment.end),
                                 text=text))

log.info(f"Finished translating {audio_file_location}")

# Insert a disclaimer right into the first DISCLAIMER_TIME s.
# Note that we are going to basically cut off the first DISCLAIMER_TIME seconds.
# This shouldn't be an issue because this should be the intro for most videos
while len(subs) > 0 and subs[0].end < DISCLAIMER_TIME:
    sub = subs.pop(0)
    log.info('Removed sub: ')
    log.info(sub)

subs.insert(index=0,
            value=pysubs2.SSAEvent(start=pysubs2.make_time(ms=1),
                                   end=pysubs2.make_time(ms=DISCLAIMER_TIME - 100),
                                   text=DISCLAIMER))

# Insert a json containing the settings right at the start
# Hopefully this will allow us to re-generate in future if necessary
settings = {
    'VIDEO_ID': VIDEO_ID,
    'MODEL_SIZE': MODEL_SIZE,
    'COMPUTE_TYPE': COMPUTE_TYPE,
    'VERSIONS': {
        'ctranslate2': version('ctranslate2'),
        'faster_whisper': version('faster_whisper'),
        'yt-dlp': version('yt-dlp'),
        'onnxruntime': version('onnxruntime'),
    },
    'TranscriptionOptions': info.transcription_options._asdict(),
    'VadOptions': info.vad_options._asdict()
}

log.info(json.dumps(settings, indent=2))
subs.insert(index=0,
            value=pysubs2.SSAEvent(start=pysubs2.make_time(ms=0),
                                   end=pysubs2.make_time(ms=1),
                                   text=json.dumps(settings, indent=None, separators=(',', ':'))))

# Save SRT to file
subs.save(os.path.join(OUTPUT_DIR, output_filename), keep_ssa_tags=True)
