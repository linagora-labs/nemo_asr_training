import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

datasets_names = {
    "mls_facebook_french": "MLS",
    "youtubefr_split6": "YouTube",
}

CER_WHISPER_PER_DATASET = {
    "CommonVoice": 4.1,
    "FLEURS": 2,
    "YouTube": 20,
    "MLS": 2.5,
    "CFPP2000": 38.6,
    "AfricanAccentedFrench": 3.1,
}

WER_WHISPER_PER_DATASET = {
    "CommonVoice": 10.8,
    "FLEURS": 5.3,
    "YouTube": 29.9,
    "MLS": 5.1,
    "CFPP2000": 50.7,
    "AfricanAccentedFrench": 6.9,
    "MLS": 5,
    "OFROM": 39.7,
    "SUMM-RE": 29.9,
    "TCOF_Adultes": 45.4,
    "TCOF_Enfants": 59.4,
    "Voxpopuli": 12.1,
    "TEDX": 9
}

COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'lime', 'teal', 'navy', 'maroon', 'silver', 'gold', 'indigo', 'violet', 'coral', 'tan', 'aquamarine', 'turquoise', 'salmon', 'khaki', 'plum', 'orchid', 'peru', 'lavender', 'cyan', 'darkgreen', 'darkred', 'darkblue', 'darkorange', 'darkviolet', 'darkturquoise', 'darkmagenta', 'darkcyan', 'darkkhaki', 'darkolive', 'darkorchid', 'darkplum', 'darksalmon', 'darkseagreen', 'darkslategray', 'darkslategrey', 'darktan', 'darkteal']