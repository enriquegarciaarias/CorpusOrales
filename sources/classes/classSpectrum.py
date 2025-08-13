import parselmouth

class MeasureSpectralShape:
    """Measure characteristics of the spectral shape

    Arguments
    =========

    args: dict
        A dictionary with the following keys:

            "voice": tuple (signal, sampling_rate)
            "Low band floor (Hz)": float or int, default=0.0
            "Low band ceiling (Hz)": float or int, default=500.0
            "High band floor (Hz)": float or int, default=500.0
            "High band ceiling (Hz)": float or int, default=4000.0
            "Power": int, default=2
            "file_path": str, optional, for reference/logging

    """

    def __init__(self, args=None):
        if args is None:
            args = {}

        self.args = {
            "Low band floor (Hz)": 0.0,
            "Low band ceiling (Hz)": 500.0,
            "High band floor (Hz)": 500.0,
            "High band ceiling (Hz)": 4000.0,
            "Power": 2,
        }
        self.args.update(args)

    def process(self):
        try:
            signal, sampling_rate = self.args["voice"]
            sound = parselmouth.Sound(signal, sampling_rate)
            spectrum = sound.to_spectrum()

            power = self.args["Power"]
            low_band_floor = self.args["Low band floor (Hz)"]
            low_band_ceiling = self.args["Low band ceiling (Hz)"]
            high_band_floor = self.args["High band floor (Hz)"]
            high_band_ceiling = self.args["High band ceiling (Hz)"]

            centre_of_gravity = spectrum.get_centre_of_gravity(power)
            standard_deviation = spectrum.get_standard_deviation(power)
            kurtosis = spectrum.get_kurtosis(power)
            skewness = spectrum.get_skewness(power)
            band_energy_difference = spectrum.get_band_energy_difference(
                low_band_floor, low_band_ceiling, high_band_floor, high_band_ceiling
            )
            band_density_difference = spectrum.get_band_density_difference(
                low_band_floor, low_band_ceiling, high_band_floor, high_band_ceiling
            )

            return {
                "Centre of Gravity": centre_of_gravity,
                "Standard Deviation": standard_deviation,
                "Kurtosis": kurtosis,
                "Skewness": skewness,
                "Band Energy Difference": band_energy_difference,
                "Band Density Difference": band_density_difference,
            }

        except Exception as e:
            error_msg = str(e)
            return {
                "Centre of Gravity": error_msg,
                "Standard Deviation": error_msg,
                "Kurtosis": error_msg,
                "Skewness": error_msg,
                "Band Energy Difference": error_msg,
                "Band Density Difference": error_msg,
            }
