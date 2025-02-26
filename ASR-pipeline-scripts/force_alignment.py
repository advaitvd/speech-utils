# import torchaudio
import torch
import librosa
import torchaudio.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import json
import os
import torch
# import nemo.collections.asr as nemo_asr
from espnet_inference import EspnetInference

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

hi_espnet_model = EspnetInference()

# ta_indicconf_model_path = '/home/rishabh.kumar/abhinav/transcription/models/indicconformer/ai4b_indicConformer_ta.nemo'
# ta_indicconf_model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=ta_indicconf_model_path)
# ta_indicconf_model.freeze()  # inference mode
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ta_indicconf_model = ta_indicconf_model.to(device)  # transfer model to device
# ta_indicconf_model.cur_decoder = "ctc"

# hi_indicconf_model_path = '/workspace/advait/workspace/pipeline/kritarth/model/ai4b_indicConformer_hi.nemo'
# hi_indicconf_model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=hi_indicconf_model_path)
# hi_indicconf_model.freeze()  # inference mode
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# hi_indicconf_model = hi_indicconf_model.to(device)  # transfer model to device
# hi_indicconf_model.cur_decoder = "rnnt"

# te_indicconf_model_path = '/workspace/kritarth/IndicConformerAsr/NeMo/models/ai4b_indicConformer_te.nemo'
# te_indicconf_model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=te_indicconf_model_path)
# te_indicconf_model.freeze()  # inference mode
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# te_indicconf_model = te_indicconf_model.to(device)  # transfer model to device
# te_indicconf_model.cur_decoder = "ctc"


# mr_indicconf_model_path = '/workspace/advait/workspace/pipeline/kritarth/model/ai4b_indicConformer_hi.nemo'
# mr_indicconf_model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=mr_indicconf_model_path)
# mr_indicconf_model.freeze()  # inference mode
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# mr_indicconf_model = mr_indicconf_model.to(device)  # transfer model to device
# mr_indicconf_model.cur_decoder = "ctc"


# indicwhisper_pipe = pipeline(task="automatic-speech-recognition", model="parthiv11/indic_whisper_hi_multi_gpu", chunk_length_s=30, device=device)
# indicwhisper_pipe.model.config.forced_decoder_ids = indicwhisper_pipe.tokenizer.get_decoder_prompt_ids(language="hi", task="transcribe")

# indicwhisper_pipe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-hindi-large-v2", chunk_length_s=30, device=device)
# indicwhisper_pipe.model.config.forced_decoder_ids = indicwhisper_pipe.tokenizer.get_decoder_prompt_ids(language="hi", task="transcribe")

# indicwhisper_pipe = pipeline(task="automatic-speech-recognition", model="parthiv11/indic_whisper_nodcil", chunk_length_s=30, device=device, model_kwargs={"ignore_mismatched_sizes": True})
# indicwhisper_pipe.model.config.forced_decoder_ids = indicwhisper_pipe.tokenizer.get_decoder_prompt_ids(language="hi", task="transcribe")

# # for tamil
# tamil_w2v_processor = Wav2Vec2Processor.from_pretrained("Harveenchadha/vakyansh-wav2vec2-tamil-tam-250")
# tamil_w2v_model = Wav2Vec2ForCTC.from_pretrained("Harveenchadha/vakyansh-wav2vec2-tamil-tam-250")
# tamil_dict = tamil_w2v_processor.tokenizer.get_vocab()

# # for telugu
# telugu_w2v_processor = Wav2Vec2Processor.from_pretrained("Harveenchadha/vakyansh-wav2vec2-telugu-tem-100")
# telugu_w2v_model = Wav2Vec2ForCTC.from_pretrained("Harveenchadha/vakyansh-wav2vec2-telugu-tem-100")
# telugu_dict = telugu_w2v_processor.tokenizer.get_vocab()


# for marathi
# marathi_w2v_processor = Wav2Vec2Processor.from_pretrained("Harveenchadha/vakyansh-wav2vec2-marathi-mrm-100")
# marathi_w2v_model = Wav2Vec2ForCTC.from_pretrained("Harveenchadha/vakyansh-wav2vec2-marathi-mrm-100")
# marathi_dict = marathi_w2v_processor.tokenizer.get_vocab()


# #for hindi
hi_w2v_processor = Wav2Vec2Processor.from_pretrained("ai4bharat/indicwav2vec-hindi")
hi_w2v_model = Wav2Vec2ForCTC.from_pretrained("ai4bharat/indicwav2vec-hindi")
hi_dict = hi_w2v_processor.tokenizer.get_vocab()

# #for sanskrit
# sanskrit_w2v_processor = Wav2Vec2Processor.from_pretrained("addy88/wav2vec2-sanskrit-stt")
# sanskrit_w2v_model = Wav2Vec2ForCTC.from_pretrained("addy88/wav2vec2-sanskrit-stt")
# sanskrit_dict = sanskrit_w2v_processor.tokenizer.get_vocab()

def get_token_offsets(emission, transcript, waveform, dictionary, sample_rate=16000):
    # Function to align the emission with the transcript and get token offsets
    def align(emission, tokens):
        targets = torch.tensor([tokens], dtype=torch.int32)
        alignments, scores = F.forced_align(emission, targets, blank=0)

        alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
        scores = scores.exp()  # convert back to probability
        return alignments, scores

    def unflatten(list_, lengths):
        assert len(list_) == sum(lengths)
        i = 0
        ret = []
        for l in lengths:
            ret.append(list_[i : i + l])
            i += l
        return ret

    # def compute_alignments(emission, transcript, dictionary):
    #     tokens = [dictionary[char] for word in transcript for char in word]
    #     alignment, scores = align(emission, tokens)
    #     token_spans = F.merge_tokens(alignment, scores)
    #     word_spans = unflatten(token_spans, [len(word) for word in transcript])
    #     return word_spans

    def compute_alignments(emission, transcript, dictionary):
        tokens = []
        for word in transcript:
            for char in word:
                try:
                    tokens.append(dictionary[char])  
                except KeyError:
                    tokens.append(dictionary.get('<unk>', -1))  
                    print(f"Character '{char}' not found in dictionary. Using <UNK>")
        alignment, scores = align(emission, tokens)
        token_spans = F.merge_tokens(alignment, scores)
        word_spans = unflatten(token_spans, [len(word) for word in transcript])
        return word_spans

    def _score(spans):
        return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)

    # Compute alignments and token spans
    word_spans = compute_alignments(emission, transcript, dictionary)
    num_frames = emission.shape[1]
    ratio = waveform.shape[0] / num_frames
    # Create a list of dictionaries with token, start, and end offsets
    token_offsets = []
    for word, spans in zip(transcript, word_spans):
        x0 = int(ratio * spans[0].start)
        x1 = int(ratio * spans[-1].end)
        start = x0 / sample_rate
        end = x1 / sample_rate
        token_offsets.append({
            'token': word.lower(),  # Token corresponding to the span start
            'start_offset': round(start, 3),
            'end_offset': round(end, 3),
        })

    return token_offsets

def convert_to_token_format(word_timestamps):
    token_format = []
    for entry in word_timestamps:
        token_format.append({
            "token": entry['text'].strip(),
            "start_offset": round(entry['timestamp'][0], 3),
            "end_offset": round(entry['timestamp'][1], 3)
        })
    return token_format

def clean_text(vocab_dict, text):
    # Convert the text to lowercase
    text = text.lower()
    
    valid_chars = set(vocab_dict.keys())
    valid_chars.add(' ')

    cleaned_text = ''.join(char for char in text if char in valid_chars)

    return cleaned_text


def transcribe_audio(audio_path, source_lang, json_dir, model):

    model_id = None
    if model == "wav2vec2" and source_lang == "Hindi":
        model_id = "ai4bharat/indicwav2vec-hindi"
    elif model == "whisper" and source_lang == "English":
        model_id = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    elif model == "whisper" and source_lang == "Hindi":
        # model_id = "vasista22/whisper-hindi-large-v2"
        model_id = "parthiv11/indic_whisper_nodcil"
    elif model == "wav2vec2" and source_lang == "Tamil":
        model_id = "Harveenchadha/vakyansh-wav2vec2-tamil-tam-250" 
    elif model == "wav2vec2" and source_lang == "Sanskrit":
        model_id = "addy88/wav2vec2-sanskrit-stt"
    elif model == "indicconformer":
        model_id = "No need"

    if not model_id:
        raise ValueError(f"Unsupported language: {source_lang}")

    audio_array, sr = librosa.load(audio_path, sr=16_000)

    if source_lang == "Marathi" and model == "indicconformer":
        print(f"Transcribing {audio_path}")
        transcription = mr_indicconf_model.transcribe([audio_path], batch_size=1, logprobs=False, language_id="mr")[0]
        transcription[0] = transcription[0].replace("ऍ","एॅ")
        transcription[0] = transcription[0].replace("ऱ","र")
        print(transcription)
        # transcription[0] = transcription[0].replace("ऍ", "एॅ")
        TRANSCRIPT = transcription[0].strip().split()
        inputs = marathi_w2v_processor(audio_array, sampling_rate=sr, return_tensors="pt")

        with torch.no_grad():
            logits = marathi_w2v_model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            emission = torch.nn.functional.log_softmax(logits, dim=-1)        
        
        print("Transscript: ",TRANSCRIPT)
        if len(TRANSCRIPT)>0:
            word_offset = get_token_offsets(emission, TRANSCRIPT, audio_array, marathi_dict, sr)
        else:
            word_offset = []

    elif source_lang == "Telugu" and model == "indicconformer":
        print(f"Transcribing {audio_path}")
        transcription = te_indicconf_model.transcribe([audio_path], batch_size=1, logprobs=False, language_id="te")[0]
        print(transcription)
        TRANSCRIPT = transcription[0].strip().split()
        inputs = telugu_w2v_processor(audio_array, sampling_rate=sr, return_tensors="pt")

        with torch.no_grad():
            logits = telugu_w2v_model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            emission = torch.nn.functional.log_softmax(logits, dim=-1)        
        
        print("Transscript: ",TRANSCRIPT)
        if len(TRANSCRIPT)>0:
            word_offset = get_token_offsets(emission, TRANSCRIPT, audio_array, telugu_dict, sr)
        else:
            word_offset = []

    elif source_lang == "Hindi" and model == "indicconformer":
        print(f"Transcribing {audio_path}")
        transcription = hi_espnet_model(audio_path)
        # transcription = hi_indicconf_model.transcribe([audio_path], batch_size=1)[0]
        # print(transcription)
        TRANSCRIPT = transcription.strip().split()
    
        inputs = hi_w2v_processor(audio_array, sampling_rate=sr, return_tensors="pt")

        with torch.no_grad():
            logits = hi_w2v_model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            emission = torch.nn.functional.log_softmax(logits, dim=-1)        
        
        print("Transcript: ",TRANSCRIPT)
        if len(TRANSCRIPT)>0:
            word_offset = get_token_offsets(emission, TRANSCRIPT, audio_array, hi_dict, sr)
        else:
            word_offset = []

    elif source_lang == "Hindi" and model == "wav2vec2":
        print("src lang and model type: ", source_lang, model, model_id)
        # print(DICTIONARY)
        print(f"Transcribing {audio_path}")
        inputs = hi_w2v_processor(audio_array, sampling_rate=sr, return_tensors="pt")

        with torch.no_grad():
            
            logits = hi_w2v_model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            emission = torch.nn.functional.log_softmax(logits, dim=-1)

        transcription = hi_w2v_processor.batch_decode(predicted_ids, skip_special_tokens=True)
        TRANSCRIPT = transcription[0].strip().split()

        print("TRANSCRIPT: ",TRANSCRIPT)
        if len(TRANSCRIPT)>0:
            word_offset = get_token_offsets(emission, TRANSCRIPT, audio_array, hi_dict, sr)
        else:
            word_offset = []
            
    elif source_lang == "Tamil" and model == "wav2vec2":
        print("src lang and model type: ", source_lang, model, model_id)
        # print(DICTIONARY)
        print(f"Transcribing {audio_path}")
        inputs = tamil_w2v_processor(audio_array, sampling_rate=sr, return_tensors="pt")

        with torch.no_grad():
            
            logits = tamil_w2v_model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            emission = torch.nn.functional.log_softmax(logits, dim=-1)

        transcription = tamil_w2v_processor.batch_decode(predicted_ids, skip_special_tokens=True)
        TRANSCRIPT = transcription[0].strip().split()

        print("TRANSCRIPT: ",TRANSCRIPT)
        if len(TRANSCRIPT)>0:
            word_offset = get_token_offsets(emission, TRANSCRIPT, audio_array, tamil_w2v_processor.tokenizer.get_vocab(), sr)
        else:
            word_offset = []
    elif source_lang == "Hindi" and model == "whisper":
        # print("src lang and model type: ", source_lang, model, model_id, "parthiv11/indic_whisper_hi_multi_gpu")
        print("src lang and model type: ", source_lang, model, model_id, "parthiv11/indic_whisper_nodcil")
        # print("Dict: ", DICTIONARY)
        print(f"Transcribing {audio_path}")

        inputs = hi_w2v_processor(audio_array, sampling_rate=sr, return_tensors="pt")

        with torch.no_grad():
            logits = hi_w2v_model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            emission = torch.nn.functional.log_softmax(logits, dim=-1)
        
        result = indicwhisper_pipe(audio_path)
        text = clean_text(hi_dict, result["text"])
        # print(text)
        TRANSCRIPT = text.strip().split()
        print("Trans: ",TRANSCRIPT)
        if len(TRANSCRIPT)>0:
            word_offset = get_token_offsets(emission, TRANSCRIPT, audio_array, hi_dict, sr)
        else:
            word_offset = []
    # for sanskrit
    elif source_lang == "Sanskrit" and model == "wav2vec2":
        print("src lang and model type: ", source_lang, model, model_id)
        print(f"Transcribing {audio_path}")
        inputs = sanskrit_w2v_processor(audio_array, sampling_rate=sr, return_tensors="pt")

        with torch.no_grad():
            logits = sanskrit_w2v_model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            emission = torch.nn.functional.log_softmax(logits, dim=-1)

        transcription = sanskrit_w2v_processor.batch_decode(predicted_ids, skip_special_tokens=True)
        TRANSCRIPT = transcription[0].strip().split()

        print("TRANSCRIPT: ", TRANSCRIPT)
        if len(TRANSCRIPT) > 0:
            word_offset = get_token_offsets(emission, TRANSCRIPT, audio_array, sanskrit_dict, sr)
        else:
            word_offset = []

    else:
        print("src lang and model type: ", source_lang, model)
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id)
        DICTIONARY = processor.tokenizer.get_vocab()
        print("Dict: ", DICTIONARY)

        inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt")

        with torch.no_grad():
            
            logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            emission = torch.nn.functional.log_softmax(logits, dim=-1)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-medium"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )
        
        result = pipe(audio_path, return_timestamps="word", generate_kwargs={"language": "english"})
        text = clean_text(DICTIONARY, result["text"])
        TRANSCRIPT = text.strip().split()
        
        if len(TRANSCRIPT)>0:
            word_offset = get_token_offsets(emission, TRANSCRIPT, audio_array, DICTIONARY, sr)
        else:
            word_offset = []


    print(word_offset)

    json_filename = os.path.splitext(os.path.basename(audio_path))[0] + '_word_offset.json'
    json_path = os.path.join(json_dir, json_filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(word_offset, f, ensure_ascii=False, indent=4)

    print(f"Data saved to {json_path}")

    return word_offset

# # Example usage
# audio_path = '/home/rishabh.kumar/abhinav/trash/natamil_audio_vad_1.wav'
# source_lang = 'Tamil'  
# json_dir = '/home/rishabh.kumar/abhinav/asr_pipeline/abhinav'
# model = 'indicconformer'
# word_offsets = transcribe_audio(audio_path, source_lang, json_dir, model)
# print(word_offsets)
