from sources.common.common import logger, processControl, log_
import json
import time
import os
from os.path import isdir
import tempfile
from pydub import AudioSegment

def mkdir(dir_path):
    """
    @Desc: Creates directory if it doesn't exist.
    @Usage: Ensures a directory exists before proceeding with file operations.
    """
    if not isdir(dir_path):
        os.makedirs(dir_path)


def dbTimestamp():
    """
    @Desc: Generates a timestamp formatted as "YYYYMMDDHHMMSS".
    @Result: Formatted timestamp string.
    """
    timestamp = int(time.time())
    formatted_timestamp = str(time.strftime("%Y%m%d%H%M%S", time.gmtime(timestamp)))
    return formatted_timestamp

class configLoader:
    """
    @Desc: Loads and provides access to JSON configuration data.
    @Usage: Instantiates with path to config JSON file.
    """
    def __init__(self, config_path='config.json'):
        self.base_path = os.path.realpath(os.getcwd())
        realConfigPath = os.path.join(self.base_path, config_path)
        self.config = self.load_config(realConfigPath)

    def load_config(self, realConfigPath):
        with open(realConfigPath, 'r') as config_file:
            return json.load(config_file)

    def get_environment(self):
        environment =  self.config.get("environment", None)
        environment["realPath"] = self.base_path
        return environment

    def get_defaults(self):
        return self.config.get("defaults", {})

def grabaJson(data, path):
    try:

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    except Exception as e:
        log_("error", logger, f"Error write json path:{path}, error:{e}")
        return False

    log_("info", logger, f"JSON written path:{path}")
    return True

def leeJson(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            results = json.load(f)

    except Exception as e:
        log_("error", logger, f"Error read json path:{path}, error:{e}")
        return None

    return results

def save_results(resultsPath, lemma, sexo, edad, results_data):
    new_entry = {
        "query": {"lemma": lemma, "sexo": sexo, "edad": edad},
        "results": results_data
    }

    if os.path.exists(resultsPath):
        with open(resultsPath, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
        existing_results.append(new_entry)
        grabaJson(existing_results, resultsPath)

    else:
        grabaJson([new_entry], resultsPath)


def get_next_combination(resultsPath):
    if not os.path.exists(resultsPath):
        return {
            "lemma": "bar",
            "sexo": "H",
            "edad": "1"
        }

    # Read existing results
    with open(resultsPath, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # Extract used combinations from results
    used_combinations = set()
    for entry in results:
        query = entry["query"]
        used_combinations.add((query["lemma"], query["sexo"], query["edad"]))

    # Generate all possible combinations
    combinations = {
        "lemma":["bar", "iglesia"],
        "sexo":["H", "M"],
        "edad":["1", "2", "3"],
        "estudios":["1", "2", "3"],
    }

    all_combinations = [
        {"lemma": lemma, "sexo": sexo, "edad": edad}
        for lemma in combinations["lemma"]
        for sexo in combinations["sexo"]
        for edad in combinations["edad"]
    ]

    # Find the first unused combination
    for comb in all_combinations:
        if (comb["lemma"], comb["sexo"], comb["edad"]) not in used_combinations:
            return comb

    return None  # All combinations are used

def extract_sexo_from_path(audioFile):
    """Extrae el sexo del locutor a partir de la ruta del archivo."""
    directory = os.path.dirname(audioFile)
    last_dir = os.path.basename(directory)
    parts = last_dir.split('-')
    return parts[1] if len(parts) >= 2 else None

def convert_to_wav(audio_path):
    """Convierte un archivo MP3 a WAV si es necesario."""
    if audio_path.lower().endswith('.mp3'):
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_path = temp_wav.name
            audio = AudioSegment.from_file(audio_path, format="mp3")
            audio = audio.set_channels(1)  # Convertir a mono
            audio.export(temp_path, format="wav", parameters=["-ar", "44100"])
            print(f"Converted {audio_path} to {temp_path}")
            return temp_path
        except Exception as e:
            log_("error", logger, f"Error converting {audio_path} to WAV: {str(e)}")
            return None
    return audio_path