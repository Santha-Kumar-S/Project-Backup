import stone
from json import dumps

class ToneAnalyser:
    def __init__(self):
        self.palette = ["#FFDDC1", "#E4C9B0", "#D0BFAE", "#A69E9D", "#8B8378", "#695D53", "#4A412A"]
        self.label = ["Fair", "Light", "Medium", "Olive", "Tan", "Brown", "Dark"]
    
    def analyse_skin_tone(self, img_path, img_format):
        result = stone.process(img_path, img_format, self.palette, self.label, return_report_image=True)
        label = result['faces'][0]['tone_label']
        accuracy = result['faces'][0]['accuracy']
        skin_tone = result['faces'][0]['skin_tone']
        dominant_colors = result['faces'][0]['dominant_colors']

        self.model = {
            "label": label,
            "accuracy": accuracy,
            "skin_tone": skin_tone,
            "dominant_colors": dominant_colors,
             "result" :result
        }

        return self.model