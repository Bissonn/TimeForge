import os
import csv
import time
import math
import logging

logger = logging.getLogger("CalibrationLogger")


class CalibrationLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        # Dodajemy kolumnę 'metric_used', żebyś wiedział co zostało zmierzone
        self.headers = [
            "timestamp", "dataset", "window_size", "model", "lr",
            "fold", "epoch", "train_loss", "val_loss", "is_stable", "metric_used"
        ]
        self._initialize_file()

    def _initialize_file(self):
        directory = os.path.dirname(self.filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        if not os.path.exists(self.filepath):
            with open(self.filepath, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def _get_metric_values(self, history, prefix='val'):
        """
        Pomocnicza funkcja szukająca najlepszego klucza dla metryki.
        Szuka: val_loss, val_mae, val_mse, loss, mae, mse itp.
        """
        # Możliwe nazwy kluczy w kolejności priorytetu
        candidates = [
            f"{prefix}_loss",
            f"{prefix}_mae",
            f"{prefix}_mse",
            f"{prefix}_rmse",
            # Fallbacki bez prefixu (dla trainingu często jest po prostu 'loss')
            "loss" if prefix == 'train' else None,
            "mae" if prefix == 'train' else None,
            "mse" if prefix == 'train' else None
        ]
        candidates = [c for c in candidates if c]  # usuń None

        # Sprawdź co jest dostępne w historii
        available_keys = list(history.keys())

        selected_key = None
        values = []

        for key in candidates:
            if key in available_keys:
                selected_key = key
                values = history[key]
                break

        # Jeśli nic nie znaleziono, weź pierwszy dostępny klucz zawierający 'loss' lub 'val'
        if values is None or len(values) == 0:
            for key in available_keys:
                if prefix in key:
                    selected_key = key
                    values = history[key]
                    break

        return selected_key, values

    def log_history(self, dataset, window, model_name, lr, fold, history):
        """
        Loguje historię do CSV, dynamicznie znajdując odpowiednie metryki.
        """
        if not history:
            return

        # 1. Znajdź odpowiednie klucze dla Train i Val
        train_key, train_values = self._get_metric_values(history, prefix='train')
        val_key, val_values = self._get_metric_values(history, prefix='val')

        # Jeśli nie udało się znaleźć listy wartości, przerywamy
        if not train_values:
            # Fallback: spróbuj znaleźć cokolwiek co wygląda na loss
            train_key, train_values = self._get_metric_values(history, prefix='')

        # Ustal długość na podstawie dostępnych danych
        epochs = 0
        if train_values:
            epochs = len(train_values)
        elif val_values:
            epochs = len(val_values)

        if epochs == 0:
            logger.warning(f"Fold {fold}: History is empty or keys unrecognized. Keys found: {list(history.keys())}")
            return

        # Zapis do pliku
        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            for i in range(epochs):
                # Bezpieczne pobieranie wartości
                t_loss = train_values[i] if train_values and i < len(train_values) else None
                v_loss = val_values[i] if val_values and i < len(val_values) else None

                # Check stability
                is_stable = True
                if (t_loss is None or isinstance(t_loss, float) and (math.isnan(t_loss) or math.isinf(t_loss))) or \
                        (v_loss is None or isinstance(v_loss, float) and (math.isnan(v_loss) or math.isinf(v_loss))):
                    is_stable = False

                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    dataset,
                    window,
                    model_name,
                    f"{float(lr):.6f}",  # Upewniamy się że LR jest formatowany jako float
                    fold,
                    i + 1,
                    f"{t_loss:.6f}" if t_loss is not None else "NaN",
                    f"{v_loss:.6f}" if v_loss is not None else "NaN",
                    "YES" if is_stable else "NO",
                    val_key  # Informacja jaka metryka została użyta
                ])