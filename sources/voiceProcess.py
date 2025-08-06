from sources.common.common import logger, log_
from sources.common.utils import grabaJson, leeJson, dbTimestamp
import parselmouth
from parselmouth.praat import call
import numpy as np
import os
from pathlib import Path
from scipy.stats import zscore
import json
from pydub import AudioSegment
import tempfile

# Ruta de caché
cache_path = "prosody_cache.json"
cache = leeJson(cache_path) if os.path.exists(cache_path) else {}

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

def extraer_caracteristicas_prosodicas(snd, point_process, min_pitch, max_pitch):
    """Extrae características prosódicas avanzadas como jitter, shimmer y HNR."""
    features = {"jitter_local": None, "shimmer_local": None, "hnr": None}

    num_points = call(point_process, "Get number of points")
    if num_points < 5:
        log_("info", logger, "Advertencia: Muy pocos puntos para análisis prosódico; ajusta min_pitch o revisa audio")
        return features

    intensity = snd.to_intensity(minimum_pitch=min_pitch)

    try:
        jitter_local = call([snd, point_process], "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        features["jitter_local"] = jitter_local
    except Exception as e:
        log_("info", logger, f"Error al calcular jitter: {str(e)}")

    try:
        shimmer_local = call([snd, intensity, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        features["shimmer_local"] = shimmer_local
    except Exception as e:
        log_("info", logger, f"Error al calcular shimmer: {str(e)}")

    try:
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, min_pitch, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        features["hnr"] = hnr
    except Exception as e:
        log_("info", logger, f"Error al calcular HNR: {str(e)}")

    return features

def extract_sexo_from_path(audioFile):
    """Extrae el sexo del locutor a partir de la ruta del archivo."""
    directory = os.path.dirname(audioFile)
    last_dir = os.path.basename(directory)
    parts = last_dir.split('-')
    return parts[1] if len(parts) >= 2 else None

def extraePitch(audioFile, include_advanced=False, threshold_silencio_db=-30.0, min_duracion_pausa=0.2):
    """Extrae características prosódicas detalladas de un archivo de audio."""
    #if audioFile in cache:
    #    return cache[audioFile]

    log_("info", logger, f"Attempting to load: {audioFile}")
    if not os.path.exists(audioFile):
        log_("error", logger, f"File does not exist: {audioFile}")
        return {}

    try:
        snd = parselmouth.Sound(audioFile)
        log_("info", logger, f"Successfully loaded sound with duration: {snd.get_total_duration()} seconds, channels: {snd.get_number_of_channels()}")
    except parselmouth.PraatError:
        wav_path = convert_to_wav(audioFile)
        if wav_path and os.path.exists(wav_path):
            snd = parselmouth.Sound(wav_path)
            print(f"Loaded converted sound from {wav_path} with duration: {snd.get_total_duration()} seconds, channels: {snd.get_number_of_channels()}")
        else:
            log_("error", logger, f"Failed to convert or load {audioFile}")
            return {}

    if snd.get_number_of_channels() > 1:
        snd = snd.convert_to_mono()
        print(f"Converted to mono, channels: {snd.get_number_of_channels()}")

    # Filtrado de ruido
    snd = call(snd, "Filter (pass Hann band)", 50, 5000, 100)

    # Ajuste dinámico de rango de pitch con fallback
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    valid_pitches = [p for p in pitch_values if p > 0]
    min_pitch = np.percentile(valid_pitches, 10) if valid_pitches and len(valid_pitches) > 10 else 75
    max_pitch = np.percentile(valid_pitches, 90) if valid_pitches and len(valid_pitches) > 10 else 400
    sexo = extract_sexo_from_path(audioFile)
    if sexo == "M":
        min_pitch = max(100, min_pitch)
        max_pitch = min(400, max_pitch)
    min_pitch = max(50, min_pitch)  # Fallback mínimo de 50 Hz
    print(f"Using pitch range: {min_pitch}-{max_pitch} Hz for sexo {sexo}")

    # Cálculo de silencios con manejo de errores
    silences = None
    try:
        silences = call(snd, "To TextGrid (silences)", threshold_silencio_db, min_duracion_pausa, 0.1, "silent", "sounding")
    except parselmouth.PraatError as e:
        log_("warning", logger, f"Failed to calculate silences for {audioFile}: {str(e)}. Skipping silences analysis.")

    # Recorte de silencios solo si se calculó correctamente
    snd_trimmed = call(snd, "Trim silences", threshold_silencio_db, min_duracion_pausa, 0.1) if silences else snd
    point_process = call(snd_trimmed, "To PointProcess (periodic, cc)", min_pitch, max_pitch)

    features = {}
    features['prosodia'] = extraer_caracteristicas_prosodicas(snd_trimmed, point_process, min_pitch, max_pitch)

    # Análisis de pitch
    pitch = snd_trimmed.to_pitch(time_step=0.005, pitch_floor=min_pitch, pitch_ceiling=max_pitch)
    try:
        mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
        features['mean_pitch'] = zscore([mean_pitch])[0] if mean_pitch else None
        if features['mean_pitch'] and features['mean_pitch'] < 50:
            features['mean_pitch'] = None
    except Exception as e:
        print(f"Error al calcular mean_pitch: {str(e)}")
        features['mean_pitch'] = None

    # Análisis de intensidad
    intensity = snd_trimmed.to_intensity(minimum_pitch=min_pitch)
    try:
        mean_intensity = call(intensity, "Get mean", 0, 0, "energy")
        features['mean_energy'] = zscore([mean_intensity])[0] if mean_intensity else None
    except Exception as e:
        print(f"Error al calcular mean_energy: {str(e)}")
        features['mean_energy'] = None

    # Duración y tasa de habla
    duration = snd_trimmed.get_total_duration()
    features['duration'] = duration
    try:
        shortest_period = 0.8 / max_pitch
        longest_period = 0.02
        maximum_period_factor = 1.3
        num_voiced = call(point_process, "Get number of periods", 0, 0, shortest_period, longest_period, maximum_period_factor)
        speech_rate = num_voiced / duration if duration > 0 else 0
        features['speech_rate'] = zscore([speech_rate])[0] if speech_rate else None
    except Exception as e:
        print(f"Error al calcular speech_rate: {str(e)}")
        features['speech_rate'] = None

    # Características avanzadas
    if include_advanced:
        try:
            formant = snd_trimmed.to_formant_burg(time_step=0.005, maximum_formant=5500, window_length=0.025, pre_emphasis_from=50)
            formants = [call(formant, "Get mean", 0, 0, i) for i in range(1, 4)]
            features['formants'] = [zscore([f])[0] if f else None for f in formants]
        except Exception as e:
            print(f"Error al calcular formantes: {str(e)}")
            features['formants'] = [None, None, None]

        try:
            spectrum = snd_trimmed.to_spectrum()
            total_power = call(spectrum, "Get total power")
            if total_power > 0:
                spectral_centroid = call(spectrum, "Get centre of gravity", 0, max_pitch)
                features['spectral_centroid'] = zscore([spectral_centroid])[0] if spectral_centroid else None
            else:
                print("Spectral power too low; skipping centroid calculation")
                features['spectral_centroid'] = None
        except Exception as e:
            print(f"Error al calcular spectral_centroid: {str(e)}")
            features['spectral_centroid'] = None

    # Cálculo de pausas solo si silences se calculó correctamente
    if silences:
        try:
            num_pauses = call(silences, "Get number of intervals", 1) - 1
            pause_durations = []
            for i in range(1, num_pauses + 1):
                duration = call(silences, "Get end point", 1, i) - call(silences, "Get start point", 1, i)
                pause_durations.append(duration)
            features['num_pauses'] = num_pauses
            features['mean_pause_duration'] = zscore([sum(pause_durations) / num_pauses if num_pauses > 0 else 0])[0] if num_pauses > 0 else 0
        except Exception as e:
            print(f"Error al calcular pausas: {str(e)}")
            features['num_pauses'] = 0
            features['mean_pause_duration'] = 0
    else:
        features['num_pauses'] = None
        features['mean_pause_duration'] = None

    cache[audioFile] = features
    grabaJson(cache, cache_path)
    return features

def openSmileProcess(audio):
    """Procesa el audio con openSMILE si está instalado (funcionalidad opcional)."""
    try:
        import opensmile
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        features = smile.process_file(audio)
        return features.to_dict() if hasattr(features, 'to_dict') else features
    except ImportError:
        print("openSMILE not installed; skipping openSMILE processing")
        return {}

def audioAnalysis(audioFile):
    """Realiza un análisis completo del audio."""
    analisis = extraePitch(audioFile, include_advanced=True)
    # Opcional: Integrar openSMILE si está disponible
    # result.update(openSmileProcess(audioFile))
    return analisis

def processAudio(resultsPath):
    """Procesa los audios secuencialmente y guarda los resultados."""
    countQuerys = countAudios = 0
    timeSince = dbTimestamp()
    try:
        startResults = leeJson(resultsPath)
        if not startResults:
            log_("error", logger, f"No results file found at {resultsPath}")
            return
        for item in startResults:
            if "timestamp" not in item:
                item["timestamp"] = "00000000000000"
            # Convert numeric timestamps to string for consistent sorting
            elif isinstance(item["timestamp"], (int, float)):
                item["timestamp"] = str(item["timestamp"])

        # Step 2: Sort by timestamp (ascending, oldest first)
        results = sorted(startResults, key=lambda x: x["timestamp"])
        grabaJson(results, resultsPath)

        for idx, entry in enumerate(results):
            if "results" in entry and "downloadAudio" in entry["results"]:
                countQuerys += 1
                if countQuerys > 3:
                    break
                download_audio = entry["results"]["downloadAudio"]
                audioFiles = [download_audio] if isinstance(download_audio, str) else download_audio
                for audioFile in audioFiles:
                    #if audioFile not in [a["audioFile"] for a in entry.get("analysis", [])]: Esto solo procesa los que no estan
                    log_("info", logger, f"Processing Audio {audioFile}")
                    countAudios += 1
                    try:
                        if os.path.exists(audioFile):
                            analysis_result = audioAnalysis(audioFile)
                            results = leeJson(resultsPath)
                            if "analysis" not in results[idx]:
                                results[idx]["analysis"] = []
                            results[idx]["analysis"].append({
                                "audioFile": audioFile,
                                "result": analysis_result,
                            })
                            results[idx]["timestamp"] = dbTimestamp()
                            grabaJson(results, resultsPath)
                        else:
                            log_("error", logger, f"Audio file not found: {audioFile}")
                    except Exception as e:
                        log_("exception", logger, f"Error al procesar audio {audioFile}: {str(e)}")

        log_("info", logger, f"Processing completed Querys: {countQuerys}, Audios {countAudios}. Updated {resultsPath}")
        return True
    except Exception as e:
        log_("exception", logger, f"Error al procesar prosodia: {str(e)}")
        return None