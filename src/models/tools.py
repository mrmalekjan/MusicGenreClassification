import librosa
import tensorflow as tf
import numpy as np

def audio_loader(audio_path, sr, duration, mono =True, tensor_based = True):
  if tensor_based == False:
    audio = librosa.load(audio_path, sr=sr, duration=duration, mono =mono)[0] # duration is in seconds
  else:
    audio = tf.io.read_file(audio_path)
    if mono == True:
      audio, own_sample_rate = tf.audio.decode_wav(audio, desired_channels=1)
    else:
      audio, own_sample_rate = tf.audio.decode_wav(audio)
    print(own_sample_rate)
    number_of_samples = tf.shape(audio)[0]
    new_number_of_samples = tf.cast(number_of_samples * sr / own_sample_rate, tf.int32)
    audio = tf.signal.resample(audio, new_number_of_samples)

  if len(audio) < duration * sr:
    audio = reflect_pad(audio, int((duration * sr - len(audio)) / 2))
    if len(audio) < duration * sr:
      audio = np.concatenate((audio,[audio[-1]]))
  return audio

def reflect_pad(signal, pad_width):
    left_pad = signal[1:pad_width+1][::-1]
    right_pad = signal[-pad_width-1:-1][::-1]
    padded_signal = tf.concat((left_pad, signal, right_pad),0)
    #padded_signal = np.concatenate((left_pad, signal, right_pad))
    return padded_signal

def audio_nomalizer(audio):
  audio_max = np.max(audio)
  audio_min = np.min(audio)
  audio = (audio - audio_min)/(audio_max - audio_min)
  return audio, audio_min, audio_max

def audio_spectrogram_extractor(audio, n_fft=2048, hop_length=1024, expand_dims=False, normalize=False):
  audio_spectrogram = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
  audio_spectrogram = librosa.amplitude_to_db(np.abs(audio_spectrogram), ref=np.max)
  if expand_dims == True:
    audio_spectrogram = np.expand_dims(audio_spectrogram, axis=2)
  if normalize == True:
    audio_spectrogram = (audio_spectrogram - np.min(audio_spectrogram)) / (np.max(audio_spectrogram) - np.min(audio_spectrogram))
  return audio_spectrogram