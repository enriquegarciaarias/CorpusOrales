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

def extraerProsodia(snd, point_process, min_pitch, max_pitch):
    """Extrae características prosódicas avanzadas como jitter, shimmer y HNR."""
    print(f"Tipo de point_process: {type(point_process)}")
    features = {"jitter_local": None, "shimmer_local": None, "hnr": None}
    if not hasattr(snd, "get_total_duration"):
        log_("info", logger, "point_process no es un PointProcess válido, omito get_start_time")
        start_time, end_time = 0, 0
    else:
        start_time = 0
        end_time = snd.get_total_duration()
        #start_time = point_process.get_start_time()
        #end_time = point_process.get_end_time()


    min_period = 1 / max_pitch
    try:


        jitter_local = call(point_process, "Get jitter (local)", start_time, end_time, min_period, 1.3, 1.6)
    except Exception as e:
        log_("error", logger, f"Error getting jitter: {str(e)}")
        jitter_local = None
    try:
        intensity = snd.to_intensity(minimum_pitch=min_pitch)
        shimmer_local = call([snd, intensity, point_process], "Get shimmer (local)", start_time, end_time, min_period, 1.3,
                             1.6, 1.6)
    except Exception as e:

        log_("error", logger, f"Error getting shimmer: {str(e)}")
        shimmer_local = None

    features["jitter_local"] = jitter_local
    features["shimmer_local"] = shimmer_local



    num_points = call(point_process, "Get number of points")
    if num_points < 5:
        log_("info", logger, "Advertencia: Muy pocos puntos para análisis prosódico; ajusta min_pitch o revisa audio")
        return features

    #intensity = snd.to_intensity(minimum_pitch=min_pitch)
    intensity = snd.to_intensity(minimum_pitch=50)

    """
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
    """


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

def safe_pitch(pitch):
    try:
        pitch_f = float(pitch)
        if pitch_f <= 0 or np.isnan(pitch_f):
            return 75.0
        return pitch_f
    except Exception:
        return 75.0


def tiene_silencios(snd, threshold_db=-25.0, min_duracion=0.3):
    """
    Devuelve True si hay al menos un silencio detectado según los parámetros.
    """
    # Convertir a intensidad (dB)
    intensity = snd.to_intensity()
    dur_total = snd.get_total_duration()

    # Obtener valores y tiempo
    times = []
    values = []
    for i in range(intensity.get_number_of_frames()):
        t = intensity.xs()[i]
        v = intensity.values[0][i]  # dB en este frame
        times.append(t)
        values.append(v)

    # Buscar tramos por debajo del umbral
    dur_silencio = 0.0
    en_silencio = False
    inicio_silencio = 0.0

    for t, v in zip(times, values):
        if v < threshold_db:
            if not en_silencio:
                en_silencio = True
                inicio_silencio = t
        else:
            if en_silencio:
                en_silencio = False
                if t - inicio_silencio >= min_duracion:
                    return True
    # Último tramo si terminó en silencio
    if en_silencio and dur_total - inicio_silencio >= min_duracion:
        return True

    return False

def extraePitch(audioFile, include_advanced=False, threshold_silencio_db=-30.0, min_duracion_pausa=0.2):
    min_duracion_pausa = float(min_duracion_pausa)
    threshold_silencio_db = float(threshold_silencio_db)
    log_("info", logger, f"Attempting to load: {audioFile}")
    if not os.path.exists(audioFile):
        log_("error", logger, f"File not found: {audioFile}")
        return {}

    try:
        snd = parselmouth.Sound(audioFile)
        log_("info", logger, f"Successfully loaded sound with duration: {snd.get_total_duration()} seconds, channels: {snd.get_number_of_channels()}")
    except parselmouth.PraatError:
        wav_path = convert_to_wav(audioFile)
        if wav_path and os.path.exists(wav_path):
            snd = parselmouth.Sound(wav_path)
            log_("info", logger, f"Loaded converted sound from {wav_path} with duration: {snd.get_total_duration()} seconds, channels: {snd.get_number_of_channels()}")
        else:
            log_("error", logger, f"Failed to convert or load {audioFile}")
            return {}

    if snd.get_number_of_channels() > 1:
        snd = snd.convert_to_mono()
        log_("info", logger, f"Converted to mono, channels: {snd.get_number_of_channels()}")

    #signal = sound.values
    #sampling_rate = sound.sampling_frequency


    # Filtrado de ruido
    """
    Filtrar puede ayudar a reducir ruido de baja frecuencia (vibraciones, pops) y alta frecuencia (hiss, ruido electrónico), 
    lo que en teoría puede mejorar la detección de pitch y por tanto de jitter/shimmer.
    Sin embargo, o el rango para cubrir todo el rango armónico relevante: 20-8000
    """
    #snd = call(snd, "Filter (pass Hann band)", 50, 5000, 100)
    snd = call(snd, "Filter (pass Hann band)", 20, 8000, 100)

    # Ajuste dinámico de rango de pitch con validación
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    log_("debug", logger, f"Raw pitch values: {pitch_values[:10]}")
    valid_pitches = [p for p in pitch_values if p > 0]  # Initial filter
    log_("debug", logger, f"Valid pitches count: {len(valid_pitches)}")
    min_pitch = np.percentile(valid_pitches, 10) if valid_pitches and len(valid_pitches) > 10 else 75
    max_pitch = np.percentile(valid_pitches, 90) if valid_pitches and len(valid_pitches) > 10 else 400
    sexo = extract_sexo_from_path(audioFile)
    if sexo in ["M", "H"]:  # Handle both 'M' and 'H'
        min_pitch = max(75, min_pitch)
        max_pitch = min(400, max_pitch)
    else:
        min_pitch = max(100, min_pitch)
        max_pitch = min(500, max_pitch)
    min_pitch = max(50, min_pitch)  # Slightly lower minimum
    log_("info", logger, f"Using pitch range: {min_pitch}-{max_pitch} Hz for sexo {sexo}")

    silences = None
    features = {}

    # Validar antes de la primera llamada
    if not min_pitch or np.isnan(min_pitch) or min_pitch <= 0:
        min_pitch = 75.0  # valor por defecto seguro

    if not max_pitch or np.isnan(max_pitch) or max_pitch <= min_pitch:
        max_pitch = 400.0  # valor por defecto seguro

    min_pitch_safe = safe_pitch(min_pitch)

    log_("debug", logger, f"min_pitch_safe before To TextGrid (silences): {min_pitch_safe}")
    try:
        silences = call(
            snd,
            "To TextGrid (silences)",
            min_pitch_safe,  # minPitch Hz
            0.1,  # timeThreshold seg
            float(threshold_silencio_db),  # silenceThreshold en dB
            float(min_duracion_pausa),  # minSilenceDuration seg
            0.1,  # minSoundingDuration seg
            "silent",  # etiqueta silencio
            "sounding"  # etiqueta sonido
        )
        log_("info", logger, f"Se han extraido silencios: {silences}")
    except parselmouth.PraatError as e:
        log_("warning", logger, f"Failed to calculate silences: {str(e)}. Skipping silences analysis.")
        silences = None

    # Recorte de silencios
    # trim silences: A command that creates from the selected Sound a new sound with silence durations not longer than a specified value.
    # Volver a validar antes de la segunda llamada
    if not min_pitch or np.isnan(min_pitch) or min_pitch <= 0:
        min_pitch = 75.0
    min_pitch_safe = safe_pitch(min_pitch)
    log_("debug", logger, f"min_pitch_safe before Trim silences: {min_pitch_safe}")
    min_pitch_safe = 170.49318959927774
    threshold_silencio_db = -25.0
    min_duracion_pausa = 0.3
    min_pitch_safe_for_trim = max(75.0, min_pitch_safe)

    if tiene_silencios(snd, threshold_db=threshold_silencio_db, min_duracion=min_duracion_pausa):
        try:
            args_trim = (
                float(min_pitch_safe_for_trim),
                0.1,
                float(threshold_silencio_db),
                float(min_duracion_pausa),
                0.1,
                float(min_pitch_safe_for_trim),
                400.0,
                "yes",  # Save trimming info as TextGrid
                "Sound"  # Output type
            )
            snd_trimmed = call(snd, "Trim silences", *args_trim)
            log_("info", logger, f"Se ha extraido snd_trimmed: {snd_trimmed}")

        except parselmouth.PraatError as e:
            log_("warning", logger, f"Failed to trim silences: {str(e)}")
            print("snd duration:", snd.get_total_duration())
            print("snd sampling rate:", snd.sampling_frequency)
            print("min_pitch_safe:", min_pitch_safe, type(min_pitch_safe))
            print("args_trim:", args_trim)
            snd_trimmed = snd
    else:
        log_("info", logger, "No se detectaron silencios significativos, se omite Trim silences.")
        snd_trimmed = snd


    min_pitch_safe = safe_pitch(min_pitch)
    max_pitch_safe = safe_pitch(max_pitch)
    if max_pitch_safe <= min_pitch_safe:
        max_pitch_safe = min_pitch_safe + 100

    print(f"Tipo de snd: {type(snd)}")
    print(f"Tipo de snd_trimmed: {type(snd_trimmed)}")
    print(f"min_pitch_safe: {min_pitch_safe}, max_pitch_safe: {max_pitch_safe}")

    point_process = call(snd_trimmed, "To PointProcess (periodic, cc)", min_pitch_safe, max_pitch_safe)
    print(f"Tipo de point_process2: {type(point_process)}")


    features['prosodia'] = extraerProsodia(snd_trimmed, point_process, min_pitch_safe, max_pitch_safe)

    # Análisis de pitch
    features['mean_pitch'] = np.nan
    pitch = snd_trimmed.to_pitch(time_step=0.005, pitch_floor=min_pitch, pitch_ceiling=max_pitch)
    pitch_values = pitch.selected_array['frequency']
    valid_pitches = [p for p in pitch_values if p > 0 and min_pitch <= p <= max_pitch]  # Combined filter
    log_("debug", logger, f"Filtered valid pitches count: {len(valid_pitches)}")
    if valid_pitches:
        mean_pitch = np.mean(valid_pitches)
        features['mean_pitch'] = float(mean_pitch)
    else:
        features['mean_pitch'] = np.nan
        log_("warning", logger, f"No valid pitch data for {audioFile}: {len(valid_pitches)} samples")

    # Análisis de intensidad
    features['mean_energy'] = np.nan
    intensity = snd_trimmed.to_intensity(minimum_pitch=min_pitch)
    try:
        mean_intensity = call(intensity, "Get mean", 0, 0, "dB")
        if mean_intensity is not None and mean_intensity > 0:
            features['mean_energy'] = float(mean_intensity)
        else:
            features['mean_energy'] = np.nan
    except Exception as e:
        log_("error", logger, f"Error al calcular mean_energy: {str(e)}")

    # Duración y tasa de habla
    features['speech_rate'] = np.nan
    duration = snd_trimmed.get_total_duration()
    features['duration'] = duration
    if duration > 0:
        shortest_period = 0.8 / max_pitch
        longest_period = 0.02
        maximum_period_factor = 1.3
        num_voiced = call(point_process, "Get number of periods", 0, 0, shortest_period, longest_period, maximum_period_factor)
        log_("debug", logger, f"Number of voiced periods: {num_voiced}")
        if num_voiced > 0:
            speech_rate = num_voiced / duration
            features['speech_rate'] = float(zscore([speech_rate])[0])
        else:
            log_("warning", logger, f"No voiced periods detected for {audioFile}")

    # Características avanzadas
    features['formants'] = [None, None, None]
    features['spectral_centroid'] = None
    if include_advanced:
        try:
            formant = snd_trimmed.to_formant_burg(time_step=0.005, maximum_formant=5500, window_length=0.025, pre_emphasis_from=50)
            formants = [call(formant, "Get mean", 1, 0, i, "Hertz") for i in range(1, 4)]
            features['formants'] = [float(zscore([f])[0]) if f is not None and f > 0 else None for f in formants]
        except Exception as e:
            log_("error", logger, f"Error al calcular formantes: {str(e)}")

        try:
            spectrum = snd_trimmed.to_spectrum()
            spectral_centroid = call(spectrum, "Get centre of gravity", 0, 5000)
            features['spectral_centroid'] = float(spectral_centroid) if spectral_centroid and not np.isnan(
                spectral_centroid) else None
            """
           #total_power = call(spectrum, "Get total power in band", 0, max_pitch)  # Try band-limited power
            total_power = call(spectrum, "Get total power in band", 0, 5000)
            if total_power is not None and total_power > 0:
                #spectral_centroid = call(spectrum, "Get centre of gravity", 0, max_pitch)
                spectral_centroid = call(spectrum, "Get centre of gravity", 0, 5000)
                features['spectral_centroid'] = float(spectral_centroid) if spectral_centroid and not np.isnan(
                    spectral_centroid) else None
                #features['spectral_centroid'] = float(zscore([spectral_centroid])[0]) if spectral_centroid else None
            else:
                log_("warning", logger, f"Total power invalid or zero: {total_power}")            
            """


        except Exception as e:
            log_("error", logger, f"Error al calcular spectral_centroid: {str(e)}")

    # Cálculo de pausas
    features['num_pauses'] = 0
    features['mean_pause_duration'] = np.nan
    if silences:
        try:
            num_pauses = call(silences, "Get number of intervals", 1) - 1
            features['num_pauses'] = max(0, num_pauses)
            if num_pauses > 0:
                pause_durations = []
                for i in range(1, num_pauses + 1):
                    duration = call(silences, "Get end point", 1, i) - call(silences, "Get start point", 1, i)
                    if duration > 0:
                        pause_durations.append(duration)
                features['mean_pause_duration'] = float(zscore([np.mean(pause_durations)])[0]) if pause_durations else np.nan
        except Exception as e:
            log_("error", logger, f"Error al calcular pausas: {str(e)}")

    cache[audioFile] = features
    grabaJson(cache, cache_path)
    log_("info", logger, f"Ended: {features}")
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