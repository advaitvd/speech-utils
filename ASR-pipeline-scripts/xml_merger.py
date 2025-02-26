import xml.etree.ElementTree as ET
import os
from datetime import timedelta
import re
from xml_creation import get_speaker

def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return root

def delete_attribute(element, attribute):
    if attribute in element.attrib:
        del element.attrib[attribute]

def extract_chunk_number(file_path):
    match = re.search(r'chunk_(\d+)_transcript\.xml', os.path.basename(file_path))
    if match:
        return int(match.group(1))
    return -1

def time_str_to_seconds(time_str):
    h, m, s = map(float, time_str.split(':'))
    return h * 3600 + m * 60 + s

def seconds_to_time_str(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = td.total_seconds()
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours)}:{int(minutes):02}:{seconds:06.3f}"
    return time_str

def adjust_timestamps(root, time_offset):
    for element in root.iter():
        if 'start_time' in element.attrib:
            new_start_time = time_str_to_seconds(element.attrib['start_time']) + time_offset
            element.attrib['start_time'] = seconds_to_time_str(new_start_time)
        if 'end_time' in element.attrib:
            new_end_time = time_str_to_seconds(element.attrib['end_time']) + time_offset
            element.attrib['end_time'] = seconds_to_time_str(new_end_time)

def merge_xml_files(xml_dir, output_file, lang, chunk_length_s):
    all_files = [os.path.join(xml_dir, f) for f in os.listdir(xml_dir) if f.endswith('.xml')]
    all_files.sort(key=extract_chunk_number)  # Ensure files are in the correct order

    merged_root = ET.Element("transcript", lang=lang)
    current_time_offset = 0

    for file in all_files:
        root = parse_xml(file)
        adjust_timestamps(root, current_time_offset)

        lines = list(root.findall('line'))
        for line in lines:
            new_line = ET.SubElement(merged_root, 'line')
            new_line.set('start_time', line[0].get('start_time'))
            new_line.set('end_time', line[-1].get('end_time'))

            for word in line:
                word_copy = ET.SubElement(new_line, 'word', start_time=word.get('start_time'), end_time=word.get('end_time'), is_valid=word.get('is_valid'))
                word_copy.text = word.text

        current_time_offset += chunk_length_s

    post_process_lines(merged_root)

    merged_tree = ET.ElementTree(merged_root)
    merged_tree.write(output_file, encoding="unicode", xml_declaration=True)

def post_process_lines(merged_root):
    all_words = []
    for line in merged_root.findall('line'):
        words = list(line)
        all_words.extend(words)
        merged_root.remove(line)

    if not all_words:
        return

    new_line = ET.SubElement(merged_root, 'line')
    new_line.set('start_time', all_words[0].get('start_time'))

    prev_word_end = time_str_to_seconds(all_words[0].get('end_time'))
    new_line.append(all_words[0])
    word_count = 1

    for word in all_words[1:]:
        word_start = time_str_to_seconds(word.get('start_time'))
        word_end = time_str_to_seconds(word.get('end_time'))

        if word_start - prev_word_end > 0.8 or word_count >= 20:
            new_line.set('end_time', seconds_to_time_str(prev_word_end))
            new_line = ET.SubElement(merged_root, 'line')
            new_line.set('start_time', word.get('start_time'))
            word_count = 0

        new_line.append(word)
        prev_word_end = word_end
        word_count += 1

    new_line.set('end_time', seconds_to_time_str(prev_word_end))

def update_merged_xml_with_diarization(merged_xml_path, diarization_result):
    tree = ET.parse(merged_xml_path)
    root = tree.getroot()
    previous_speaker = "SPEAKER_00"
    
    for line in root.findall('line'):
        start_time = time_str_to_seconds(line.get('start_time'))
        end_time = time_str_to_seconds(line.get('end_time'))
        speaker = get_speaker(diarization_result, start_time, end_time)
        if speaker is None:
            speaker = previous_speaker
        line.set('speaker', speaker)
        previous_speaker = speaker

        line.set('timestamp', line.attrib.pop('end_time', ''))
        delete_attribute(line, 'start_time')

    for word in root.iter('word'):
        word.set('timestamp', word.attrib.pop('start_time', ''))
        delete_attribute(word, 'end_time')

    tree.write(merged_xml_path, encoding='unicode', xml_declaration=True)

# Example usage:
# xml_dir = r"D:\Business\IITB\audio_transcript\transcripts"
# output_file = "merged_transcript.xml"
# merge_xml_files(xml_dir, output_file, lang="en", chunk_length_s=600)
