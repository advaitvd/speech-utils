path=$1
cd $path
mkdir -p logs/

file Audio/* > logs/file.format

echo "********************************************************************************"
echo "File Corruption, Format & Sample Rate check:"
echo "********************************************************************************"
grep -v RIFF logs/file.format | tee logs/check.format-corruption.log
grep -vE "(16000|44100|48000)" logs/file.format | tee logs/check.sample-rate.log

echo "Done!"

readlink -f Audio/*_Left.wav > wav.Left.scp
readlink -f Audio/*_Right.wav > wav.Right.scp

sort -u wav.Left.scp -o wav.Left.scp
sort -u wav.Right.scp -o wav.Right.scp

paste <(cut -d '.' -f1 wav.Left.scp| rev | cut -d '/' -f1 | rev) wav.Left.scp > tmp && mv tmp wav.Left.scp
paste <(cut -d '.' -f1 wav.Right.scp| rev | cut -d '/' -f1 | rev) wav.Right.scp > tmp && mv tmp wav.Right.scp

utt2dur wav.Left.scp > utt2dur.Left
utt2dur wav.Right.scp > utt2dur.Right

filter-dur --utt2dur utt2dur.Left --scp wav.Left.scp
filter-dur --utt2dur utt2dur.Right --scp wav.Right.scp

cat wav.Left.filtered.scp wav.Right.filtered.scp > wav.filtered.scp

echo "********************************************************************************"
total-dur utt2dur.Left | tee logs/total-dur.Left.log
total-dur utt2dur.Right | tee logs/total-dur.Right.log
echo "********************************************************************************"
echo "Done!"

utt2freq <(cat *.filtered.scp) > utt2freq

echo "********************************************************************************"
echo "Speech-Silence Check:"
echo "********************************************************************************"
python /workspace/advait/SpeechData/Shaip/scripts/silence_speech_check.py wav.filtered.scp logs/check.speech-silence.log
echo "Done!"
