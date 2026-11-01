from io import BytesIO

import numpy as np
import soundfile as sf


def waveform_to_wav(waveform: np.ndarray, sample_rate: int) -> bytes:
    if waveform.size == 0 or sample_rate <= 0:
        raise ValueError("Waveform and sample rate must be valid.")
    buffer = BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV")
    return buffer.getvalue()
