from sources.common.common import logger, log_
from sources.common.utils import grabaJson, leeJson, dbTimestamp
from sources.classTrimmer import Trimmer
from sources.classMeasureShimmer import MeasureShimmer
from sources.classFormants import MeasureFormant
from sources.classSpectrum import MeasureSpectralShape
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
    features = {"jitter_local": None, "shimmer": {}, "hnr": None}
    """
    if not hasattr(snd, "get_total_duration"):
        log_("info", logger, "point_process no es un PointProcess válido, omito get_start_time")
        start_time, end_time = 0, 0
    else:
        start_time = 0
        end_time = snd.get_total_duration()
        #start_time = point_process.get_start_time()
        #end_time = point_process.get_end_time()    
    """
    num_points = call(point_process, "Get number of points")
    if num_points > 0:
        start_time = call(point_process, "Get time from index", 1)
        end_time = call(point_process, "Get time from index", num_points)
    else:
        log_("warning", logger, "PointProcess vacío, usando todo el rango de snd")
        start_time = 0
        end_time = snd.get_total_duration()


    min_period = 1 / max_pitch
    try:
        jitter_local = call(point_process, "Get jitter (local)", start_time, end_time, min_period, 1.3, 1.6)
    except Exception as e:
        log_("error", logger, f"Error getting jitter: {str(e)}")
        jitter_local = None


    try:
        signal = snd.values  # numpy array con la señal
        sampling_rate = snd.sampling_frequency

        measure = MeasureShimmer(
            voice=(signal, sampling_rate),
            start_time=start_time,
            end_time=end_time,
            shortest_period=min_period,
            longest_period=0.02,  # o tu valor deseado
            maximum_period_factor=1.3,
            maximum_amplitude=1.6,
            measure_pca=False  # solo quieres el shimmer local
        )

        shimmer = measure.process()

    except Exception as e:
        log_("error", logger, f"Error getting shimmer: {str(e)}")
        shimmer = None

    features["jitter_local"] = jitter_local
    features["shimmer"] = shimmer



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

    modo_trim = 1  # ← Cambia aquí para probar

    if modo_trim == 3:
        # Ajuste de sensibilidad: bajar umbral y/o duración mínima
        threshold_silencio_db_adj = threshold_silencio_db + 3  # menos estricto
        min_duracion_pausa_adj = max(min_duracion_pausa * 0.5, 0.05)  # aceptar pausas más cortas
    else:
        threshold_silencio_db_adj = threshold_silencio_db
        min_duracion_pausa_adj = min_duracion_pausa

    snd_trimmed = snd
    silencios = False
    if tiene_silencios(snd, threshold_db=threshold_silencio_db_adj, min_duracion=min_duracion_pausa_adj):
        silencios = True
        try:
            args_trim = (
                float(min_pitch_safe_for_trim),
                0.1,
                float(threshold_silencio_db_adj),
                float(min_duracion_pausa_adj),
                0.1,
                float(min_pitch_safe_for_trim),
                400.0,
                "yes",
                "Sound"
            )
            snd_trimmed = call(snd, "Trim silences", *args_trim)
            log_("info", logger, f"Se ha extraído snd_trimmed: {snd_trimmed}")

        except parselmouth.PraatError as e:
            log_("warning", logger, f"Failed to trim silences: {str(e)}")

            safer_args = (
                75.0,
                0.1,
                float(threshold_silencio_db_adj),
                float(min_duracion_pausa_adj),
                0.1,
                75.0,
                500.0,
                "yes",
                "Sound"
            )
            try:
                snd_trimmed = call(snd, "Trim silences", *safer_args)
            except parselmouth.PraatError:
                log_("warning", logger, "Second trim attempt failed, using manual cut...")
                start = snd.get_total_duration() * 0.25
                end = snd.get_total_duration() * 0.75
                try:
                    snd_trimmed = call(snd, "Extract part", start, end, "rectangular", 1.0, "yes")
                except parselmouth.PraatError:
                    log_("warning", logger, "Third trim attempt failed, using snd")
                    snd_trimmed = snd
    else:
        log_("info", logger, "No se detectaron silencios significativos.")
        try:
            if modo_trim == 1:
                # Forzar recorte mínimo (10% inicio y fin)
                dur = snd.get_total_duration()
                start = dur * 0.05
                end = dur * 0.95
                snd_trimmed = call(snd, "Extract part", start, end, "rectangular", 1.0, "yes")
                log_("info", logger, "snd_trimmed con modo 1.")

            elif modo_trim == 2:
                # Mantener original pero filtrado
                snd_trimmed = call(snd, "Filter (pass Hann band)", 50, 5000, 100)
                log_("info", logger, "snd_trimmed con modo 2.")

            else:
                # Modo normal: usar snd tal cual
                log_("info", logger, "snd_trimmed con snd")
                snd_trimmed = snd
        except parselmouth.PraatError:
            log_("warning", logger, "Second trim attempt failed, using snd")
            snd_trimmed = snd



    min_pitch_safe = safe_pitch(min_pitch)
    max_pitch_safe = safe_pitch(max_pitch)
    if max_pitch_safe <= min_pitch_safe:
        max_pitch_safe = min_pitch_safe + 100

    print(f"Tipo de snd: {type(snd)}")
    print(f"Tipo de snd_trimmed: {type(snd_trimmed)}")
    print(f"min_pitch_safe: {min_pitch_safe}, max_pitch_safe: {max_pitch_safe}")

    point_process = call(snd_trimmed, "To PointProcess (periodic, cc)", min_pitch_safe, max_pitch_safe)

    trimmer = Trimmer(trim_silences=silencios, silence_ratio=10.0, trim_sound=True, percent_trim=10.0)
    snd_trimmed = trimmer.process(snd)

    # Ahora snd_trimmed está listo para:
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
    features['formants'] = {'mean': 0, 'median': 0}
    features['spectral'] = {}
    duracion = snd_trimmed.get_total_duration()

    if include_advanced and duracion > 0.05:
        try:
            args = {
                "voice": (snd_trimmed.values.flatten(), snd_trimmed.sampling_frequency),  # Mono: extraigo canal 0 y freq
                "time step": 0.005,
                "max number of formants": 5.5,  # Puedes ajustar si quieres
                "window length(s)": 0.025,
                "pre emphasis from": 50,
                "max_formant (To Formant Burg...)": 5500,
                "method": "To Formant Burg...",  # O "Formant Path"
            }
            mf = MeasureFormant(args)
            results = mf.process()

            # Extraigo las medias de los primeros 3 formantes
            formant_means = [results[f"F{i} Mean"] for i in range(1, 4)]
            formant_medians = [results[f"F{i} Median"] for i in range(1, 4)]

            def filter_formants(values):
                return [float(v) if v is not None and v > 0 else None for v in values]

            features['formants'] = {
                'mean': filter_formants(formant_means),
                'median': filter_formants(formant_medians)
            }

        except Exception as e:
            log_("error", logger, f"Error al calcular formantes: {str(e)}")

        try:
            args = {
                "voice": (snd_trimmed.values.flatten(), snd_trimmed.sampling_frequency),
                "Low band floor (Hz)": 0.0,
                "Low band ceiling (Hz)": 5000.0,
                "High band floor (Hz)": 0.0,  # Opcional, ajusta si quieres
                "High band ceiling (Hz)": 5000.0,  # Opcional, ajusta si quieres
                "Power": 1,  # Para obtener el centro de gravedad igual que Get centre of gravity
            }

            ms = MeasureSpectralShape(args)
            features['spectral'] = ms.process()

        except Exception as e:
            log_("error", logger, f"Error al calcular spectral: {str(e)}")

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
                if pause_durations:
                    mean_pause = np.mean(pause_durations)
                    # Solo zscore si tienes más de 1 pausa
                    if len(pause_durations) > 1:
                        mean_pause_z = float(zscore(pause_durations).mean())
                    else:
                        mean_pause_z = mean_pause  # Sin normalizar
                    features['mean_pause_duration'] = mean_pause_z
                else:
                    features['mean_pause_duration'] = np.nan
            else:
                features['mean_pause_duration'] = np.nan
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