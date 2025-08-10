from sources.common.common import logger, log_
import parselmouth
from parselmouth.praat import call

class Trimmer:
    """
    Trimmer para audio en memoria usando Parselmouth.
    Puede:
    - Recortar silencios
    - Recortar un porcentaje desde el inicio y el final
    """

    def __init__(self, trim_silences=True, silence_ratio=10.0, trim_sound=True, percent_trim=10.0, min_duration=0.05):
        self.trim_silences = trim_silences
        self.silence_ratio = silence_ratio
        self.trim_sound = trim_sound
        self.percent_trim = percent_trim
        self.min_duration = min_duration  # Duración mínima aceptable en segundos

    def process(self, snd):
        """
        Procesa el audio y devuelve un snd_trimmed seguro.
        Si el trimming falla o queda demasiado corto, devuelve el original.

        Parameters
        ----------
        snd : parselmouth.Sound

        Returns
        -------
        parselmouth.Sound
        """
        snd_trimmed = snd

        try:
            if self.trim_silences:
                snd_trimmed = self._trim_silences(snd_trimmed)

            if self.trim_sound:
                snd_trimmed = self._trim_sound(snd_trimmed)

            # Si queda demasiado corto, volver al original
            if snd_trimmed.get_total_duration() < self.min_duration:
                print(f"[WARN] Audio muy corto tras trimming ({snd_trimmed.get_total_duration():.3f}s), usando original.")
                snd_trimmed = snd

        except Exception as e:
            print(f"[ERROR] Fallo en trimming: {e}")
            snd_trimmed = snd

        return snd_trimmed

    def _trim_silences(self, snd):
        """
        Recorta silencios usando el umbral relativo al rango de intensidad.
        """
        intensity = snd.to_intensity(50)
        intensity_range = intensity.get_maximum() - intensity.get_minimum()

        if intensity_range > 10:
            proportion_intensity = (intensity_range * self.silence_ratio) / 100
            tg = call(intensity, "To TextGrid (silences)",
                      proportion_intensity, 0.1, 0.05, "silent", "sounding")
            _, trimmed_sound, _ = call([snd, tg], "Extract all intervals", 1, "yes")
            return trimmed_sound
        else:
            return snd

    def _trim_sound(self, snd):
        """
        Recorta un porcentaje desde el inicio y el final.
        """
        start = snd.get_total_duration() * (self.percent_trim / 100)
        end = snd.get_total_duration() * (1 - (self.percent_trim / 100))
        trimmed_sound = call(snd, "Extract part", start, end, 'rectangular', 1.0, 'no')
        return trimmed_sound