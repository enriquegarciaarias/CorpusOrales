import parselmouth
import numpy as np
from parselmouth.praat import call

class MeasureFormant:
    """
    Measure formant frequencies using Praat's Formant Path or Formant Burg functions.

    Args:
    -----
        args : dict
            Dictionary with parameters:
            - 'time step': float, default=0.0025
            - 'max number of formants': float, default=5.5
            - 'window length(s)': float, default=0.025
            - 'pre emphasis from': float, default=50
            - 'max_formant (To Formant Burg...)': float, default=5500
            - 'Center Formant (Formant Path)': float, default=5500
            - 'Ceiling Step Size (Formant Path)': float, default=0.05
            - 'Number of Steps (Formant Path)': int, default=4
            - 'method': str, default='Formant Path' (or 'To Formant Burg...')
            - 'voice': tuple(signal: np.ndarray, sampling_rate: int)
    """

    def __init__(self, args=None):
        if args is None:
            args = {}

        self.args = {
            "time step": args.get("time step", 0.0025),
            "max number of formants": args.get("max number of formants", 5.5),
            "window length(s)": args.get("window length(s)", 0.025),
            "pre emphasis from": args.get("pre emphasis from", 50),
            "max_formant (To Formant Burg...)": args.get("max_formant (To Formant Burg...)", 5500),
            "Center Formant (Formant Path)": args.get("Center Formant (Formant Path)", 5500),
            "Ceiling Step Size (Formant Path)": args.get("Ceiling Step Size (Formant Path)", 0.05),
            "Number of Steps (Formant Path)": args.get("Number of Steps (Formant Path)", 4),
            "method": args.get("method", "Formant Path"),
            "voice": args.get("voice", None),
        }

        # Ensure max number of formants is divisible by 0.5
        max_num = self.args["max number of formants"]
        if max_num % 0.5 != 0:
            self.args["max number of formants"] = round(max_num)

    def process(self):
        signal, sampling_rate = self.args["voice"]
        sound = parselmouth.Sound(signal, sampling_rate)

        method = self.args["method"]

        try:
            if method.lower() in ['to formant burg...', 't']:
                max_formant = self.args["max_formant (To Formant Burg...)"]
                formant_object = self._measure_formants_burg(
                    sound,
                    self.args["time step"],
                    self.args["max number of formants"],
                    max_formant,
                    self.args["window length(s)"],
                    self.args["pre emphasis from"]
                )

            elif method.lower() == 'formant path' or method.lower() == 'f':
                center_formant = self.args["Center Formant (Formant Path)"]
                if center_formant == 0:
                    center_formant = self._max_formant(sound)

                formant_path_object = call(
                    sound,
                    "To FormantPath (burg)",
                    self.args["time step"],
                    self.args["max number of formants"],
                    center_formant,
                    self.args["window length(s)"],
                    self.args["pre emphasis from"],
                    self.args["Ceiling Step Size (Formant Path)"],
                    self.args["Number of Steps (Formant Path)"],
                )
                formant_object = call(formant_path_object, "Extract Formant")

            else:
                raise ValueError(f"Unsupported method: {method}")

            # Get means and medians of first 4 formants
            f_means, f_medians = [], []
            for formant_number in range(1, 5):
                mean_val = call(formant_object, "Get mean", formant_number, 0, 0, "Hertz")
                median_val = call(formant_object, "Get quantile", formant_number, 0, 0, "Hertz", 0.5)
                f_means.append(mean_val)
                f_medians.append(median_val)

            results = {
                "F1 Mean": f_means[0],
                "F2 Mean": f_means[1],
                "F3 Mean": f_means[2],
                "F4 Mean": f_means[3],
                "F1 Median": f_medians[0],
                "F2 Median": f_medians[1],
                "F3 Median": f_medians[2],
                "F4 Median": f_medians[3],
                "Formants": formant_object,
            }

        except Exception as e:
            results = {
                "F1 Mean": str(e),
                "F2 Mean": str(e),
                "F3 Mean": str(e),
                "F4 Mean": str(e),
                "F1 Median": str(e),
                "F2 Median": str(e),
                "F3 Median": str(e),
                "F4 Median": str(e),
                "Formants": str(e),
            }

        return results

    def _measure_formants_burg(self, sound, time_step, max_number_of_formants, max_formant, window_length, pre_emphasis):
        if max_formant == 0:
            max_formant = self._max_formant(sound)
        print(f"Duración snd_trimmed: {sound.get_total_duration()} segundos")
        formant_object = sound.to_formant_burg(
            time_step,
            max_number_of_formants,
            max_formant,
            window_length,
            pre_emphasis,
        )
        return formant_object

    def _max_formant(self, sound):
        # Simple heuristic: Nyquist frequency
        return sound.sampling_frequency / 2

# Ejemplo de uso:
# args = {
#     "voice": (signal_array, sampling_rate),
#     "method": "Formant Path",
# }
# mf = MeasureFormant(args)
# results = mf.process()
# print(results)
