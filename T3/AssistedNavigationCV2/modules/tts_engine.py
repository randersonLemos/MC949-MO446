import simpleaudio as sa
from piper import PiperVoice
import wave


class TTSEngine:
    def __init__(self):
        # Mude o diretório se necessário
        self.voice = PiperVoice.load(
            "models/pt_BR-cadu-medium.onnx"
        )  # ajuste para o modelo desejado

    def speak(self, text):

        audio_path = "output.wav"

        chunks = list(self.voice.synthesize(text))
        wav_bytes = b"".join(chunk.audio_int16_bytes for chunk in chunks)
        # Parâmetros do áudio
        sample_rate = chunks[0].sample_rate
        sample_width = chunks[0].sample_width
        sample_channels = chunks[0].sample_channels
        # Salva como WAV válido
        with wave.open(audio_path, "wb") as wf:
            wf.setnchannels(sample_channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(wav_bytes)
        wave_obj = sa.WaveObject.from_wave_file(audio_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()


"""
Módulo de Text-to-Speech (TTS) para alertas sonoros                                                                                                                                                                     
"""
try:
    import pyttsx3

    TTS_AVAILABLE = True
except Exception as e:
    TTS_AVAILABLE = False
    print(f"Aviso: TTS não disponível - {e}")
    print("Para habilitar TTS no Linux, instale: sudo apt-get install espeak")
