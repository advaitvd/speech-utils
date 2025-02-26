import xml.etree.ElementTree as ET
from datetime import timedelta
from collections import defaultdict
import os

def format_timestamp(seconds):
    return str(timedelta(seconds=seconds))

def get_speaker(timestamps, start_time, end_time):
    speaker_durations = defaultdict(float)

    for (start, end), speaker in timestamps.items():
        if start_time >= start and end_time <= end:
            duration = end_time - start_time
        elif start_time <= start and end_time >= end:
            duration = end - start
        elif start_time <= start < end_time or start_time < end <= end_time:
            duration = min(end, end_time) - max(start, start_time)
        else:
            continue

        speaker_durations[speaker] += duration

    max_speaker = None
    max_duration = 0.0
    for speaker, duration in speaker_durations.items():
        if duration > max_duration:
            max_duration = duration
            max_speaker = speaker

    return max_speaker

def create_transcript_xml(word_offset, diarization_result, main_audio_filename, chunk_number, xml_dir, lang):
    transcript_elem = ET.Element("transcript", lang=lang)

    def create_line_element(start_offset):
        line = ET.SubElement(transcript_elem, "line", start_time=format_timestamp(start_offset))
        return line

    line_start_offset = word_offset[0]['start_offset']
    line_elem = create_line_element(line_start_offset)

    prev_word_end = word_offset[0]['end_offset']
    for word in word_offset:
        word_start = word['start_offset']
        word_end = word['end_offset']
        token = word['token']

        if word_start - prev_word_end > 0.8:
            print(f"New line: {word_start} - {prev_word_end} ...")
            line_elem.set("end_time", format_timestamp(prev_word_end))
            line_start_offset = word_start
            line_elem = create_line_element(line_start_offset)

        word_elem = ET.SubElement(line_elem, "word", start_time=format_timestamp(word_start), end_time=format_timestamp(word_end), is_valid="1")
        word_elem.text = token

        prev_word_end = word_end

    line_elem.set("end_time", format_timestamp(prev_word_end))

    xml_str = ET.tostring(transcript_elem, encoding="unicode")
    xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml_str = xml_declaration + xml_str

    os.makedirs(xml_dir, exist_ok=True)

    xml_filename = os.path.join(xml_dir, f"{main_audio_filename}_chunk_{chunk_number}_transcript.xml")
    
    with open(xml_filename, "w", encoding="utf-8") as f:
        f.write(xml_str)

    print(f"{xml_filename} file created successfully.")
