import argparse
import os
import re
from ai4bharat.transliteration import XlitEngine

# Author: Advait Vinay Dhopeshwarkar (advait.dhopeshwarkar@tihiitb.org)
# Date  : 22-04-2025

class Normalizer:
    def __init__(self, lang_id, beam_width=10, rescore=True):
        self.lang_id = lang_id
        self.lang_norm = {
                'hi': self.normalize_hi,
                'mr': self.normalize_mr,
        }.get(self.lang_id, self.normalize_x)
        self.xlit_engine = XlitEngine(self.lang_id, beam_width=beam_width, rescore=rescore)

    def transliterate_word(self, word):
        words = word.split(' ')
        word = ' '.join([self.xlit_engine.translit_word(word, topk=1)[self.lang_id][0] for word in words])
        return word

    def handle_initials(self, word):
        return self.transliterate_word(' '.join([*word]))

    def normalize_hi(self, text):
        raise NotImplementedError("normalize_hi() is not supported in Normalizer class.")

    def normalize_mr(self, text):
        # Remove [xyz] tags
        pattern = r'\[[^\[]*\]'
        text, _ = re.subn(pattern, '', text)

        # Handle 'initial' tag
        pattern = r'<initial>(.*?)</initial>'
        def replacer(match):
            word = match.group(1)
            return self.handle_initials(word)
        text = re.sub(pattern, replacer, text)

        # Handle 'foreign' tag
        pattern = r'<lang:Foreign>(.*?)</lang:Foreign>'
        def replacer(match):
            word = match.group(1)
            return self.transliterate_word(word)
        text = re.sub(pattern, replacer, text)
        
        # Handle consecutive white spaces and replace with single space.
        pattern = r'\s{2,}'
        text, _ = re.subn(pattern, ' ', text)
        
        return text

    def normalize_x(self, text):
        raise NotImplementedError(f"normalize_{self.lang}() is not supported in Normalizer class.")

    def __call__(self, text):
        return self.lang_norm(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--lang_id", required=True)

    args = parser.parse_args()
    text, lang_id = args.text, args.lang_id
    
    normalizer = Normalizer(lang_id)
    #sample = "आणि #अह साउथ इंडिया मधल्या ना चार ज्या बिग सिटी आहेत. त्याच्या पैकी <lang:Foreign>one</lang:Foreign> ऑफ दि #अह बिग सिटी म्हटलं तर हे विशाखापट्टणम आहे."
    #sample = "<lang:Foreign>very good</lang:Foreign>"
    #sample = "त्याचे काहीतरी मग प्रोसेसिंग चार्जेस वगेरे देतात, जसे दहा हजारला काहीतरी दोनशे रुपये वगेरे मग ते <initial>EMI</initial> करून देतात विथ नो इंटरेस्ट वगैरे अस काहीतरी असत, ती चांगली प्रोसेस येते. [no-speech] हो."
    sample = "<lang:Foreign><initial>EMI</initial></lang:Foreign> नाही त्याच्यामध्ये सुद्धा काय आहे की अशे पेमेंट च्या वेगवेगळ्या पद्धती तुम्ही जर सांगितल्या तर त्या डेबिट कार्डच्या थ्रू करता येतात, क्रेडीट कार्डच्या थ्रू करता येतात. आणि <lang:Foreign><initial>EMI</initial></lang:Foreign> ने पण करता येतात. [no-speech] अस पण आहे."
    print(sample)
    print(normalizer(sample))
