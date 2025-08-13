import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import parselmouth
from parselmouth.praat import call


class MeasureShimmer:
    def __init__(
        self,
        file_path=None,
        voice=None,  # debe ser (signal, sampling_rate)
        start_time=0.0,
        end_time=0.0,
        shortest_period=0.0001,
        longest_period=0.02,
        maximum_period_factor=1.3,
        maximum_amplitude=1.6,
        measure_pca=True,
    ):
        self.file_path = file_path
        self.voice = voice  # (signal, sampling_rate)
        self.start_time = start_time
        self.end_time = end_time
        self.shortest_period = shortest_period
        self.longest_period = longest_period
        self.maximum_period_factor = maximum_period_factor
        self.maximum_amplitude = maximum_amplitude
        self.measure_pca = measure_pca

        self.local_shimmer = []
        self.localdb_shimmer = []
        self.apq3_shimmer = []
        self.aqpq5_shimmer = []
        self.apq11_shimmer = []
        self.dda_shimmer = []

    def pitch_floor(self):
        # Aquí puedes ajustar tu lógica para calcular pitch floor según file_path o voice
        # Por ahora un valor fijo ejemplo:
        return 75.0

    def pitch_ceiling(self):
        # Igual que pitch_floor
        return 500.0

    def process(self):
        if self.voice is None:
            raise ValueError("Se requiere el parámetro 'voice' (signal, sampling_rate)")

        signal, sampling_rate = self.voice
        sound = parselmouth.Sound(signal, sampling_rate)

        pitch_floor = self.pitch_floor()
        pitch_ceiling = self.pitch_ceiling()

        if self.end_time == 0.0:
            self.end_time = sound.get_total_duration()

        try:
            point_process = call(sound, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling)

            local_shimmer = call(
                [sound, point_process],
                "Get shimmer (local)",
                self.start_time,
                self.end_time,
                self.shortest_period,
                self.longest_period,
                self.maximum_period_factor,
                self.maximum_amplitude,
            )
            localdb_shimmer = call(
                [sound, point_process],
                "Get shimmer (local_dB)",
                self.start_time,
                self.end_time,
                self.shortest_period,
                self.longest_period,
                self.maximum_period_factor,
                self.maximum_amplitude,
            )
            apq3_shimmer = call(
                [sound, point_process],
                "Get shimmer (apq3)",
                self.start_time,
                self.end_time,
                self.shortest_period,
                self.longest_period,
                self.maximum_period_factor,
                self.maximum_amplitude,
            )
            aqpq5_shimmer = call(
                [sound, point_process],
                "Get shimmer (apq5)",
                self.start_time,
                self.end_time,
                self.shortest_period,
                self.longest_period,
                self.maximum_period_factor,
                self.maximum_amplitude,
            )
            apq11_shimmer = call(
                [sound, point_process],
                "Get shimmer (apq11)",
                self.start_time,
                self.end_time,
                self.shortest_period,
                self.longest_period,
                self.maximum_period_factor,
                self.maximum_amplitude,
            )
            dda_shimmer = call(
                [sound, point_process],
                "Get shimmer (dda)",
                self.start_time,
                self.end_time,
                self.shortest_period,
                self.longest_period,
                self.maximum_period_factor,
                self.maximum_amplitude,
            )

            # Guardar en estado interno
            self.local_shimmer.append(local_shimmer)
            self.localdb_shimmer.append(localdb_shimmer)
            self.apq3_shimmer.append(apq3_shimmer)
            self.aqpq5_shimmer.append(aqpq5_shimmer)
            self.apq11_shimmer.append(apq11_shimmer)
            self.dda_shimmer.append(dda_shimmer)

            return {
                "local_shimmer": local_shimmer,
                "localdb_shimmer": localdb_shimmer,
                "apq3_shimmer": apq3_shimmer,
                "aqpq5_shimmer": aqpq5_shimmer,
                "apq11_shimmer": apq11_shimmer,
                "dda_shimmer": dda_shimmer,
            }

        except Exception as e:
            return {
                "local_shimmer": str(e),
                "localdb_shimmer": str(e),
                "apq3_shimmer": str(e),
                "aqpq5_shimmer": str(e),
                "apq11_shimmer": str(e),
                "dda_shimmer": str(e),
            }

    def shimmer_pca(self):
        try:
            data = pd.DataFrame({
                "localShimmer": self.local_shimmer,
                "localdbShimmer": self.localdb_shimmer,
                "apq3Shimmer": self.apq3_shimmer,
                "apq5Shimmer": self.aqpq5_shimmer,
                "apq11Shimmer": self.apq11_shimmer,
                "ddaShimmer": self.dda_shimmer,
            }).dropna()

            measures = data.columns.tolist()

            x = StandardScaler().fit_transform(data[measures].values)
            pca = PCA(n_components=1)
            principal_components = pca.fit_transform(x)

            return principal_components.flatten()
        except Exception as e:
            return str(e)
