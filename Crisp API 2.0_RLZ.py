import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import librosa.display
import pandas as pd

data_dir = 'batata frita/exp ingredion/01/todos'
audio_files = glob(data_dir + "/*.wav")

dc = {'Ad. Energia Sonora':[0],
      'Ad. dos Picos de Amplitude':[0],
      'C Sonoro':[0]}
df1 = pd.DataFrame(dc)

for file in range(0, len(audio_files), 1):
    y, sr = librosa.load(audio_files[file], duration=1.0)
    time = np.arange(0, len(y)) / sr

    # Ad. energia Sonora
    # Detect the silence regions
    threshold = 20  # threshold in dB
    silence_regions = librosa.effects.split(y, top_db=threshold)

    # Remove the silence regions from the audio signal
    y_trimmed = np.concatenate([y[start:end] for start, end in silence_regions])

    # Compute the energy of each frame
    frame_length = int(sr * 0.01)  # frame length in samples (10 ms frames)
    hop_length = int(sr * 0.005)  # hop length in samples (5 ms hops)
    energy = librosa.feature.rms(y=y_trimmed, frame_length=frame_length, hop_length=hop_length)

    # Compute the median energy
    avg_energy = np.mean(energy)*100

    #print('Ad. Energia Sonora = ', avg_energy)

    # Ad. dos picos de Amplitude
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512, aggregate=np.median)
    peaks = librosa.util.peak_pick(onset_env, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    times = librosa.times_like(onset_env, sr=sr, hop_length=512)

    # peaks é a localização dos picos no array, onset_env[peaks] pega os valores dos picos naquele momento
    # times[peaks] representa o tempo onde os picos acontecem
    # print(onset_env[peaks])

    in_sum = onset_env[peaks]
    suma = in_sum/max(in_sum)
    value = sum(suma)*len(suma)
    #print('Ad. dos Picos de Amplitude = ', value)
    ons = onset_env/max(onset_env)

    # Criar o Dataframe pro excel
    dc = {'Ad. Energia Sonora': [avg_energy], 'Ad. dos Picos de Amplitude': [value],
          'C Sonoro':[avg_energy*value]}
    df2 = pd.DataFrame(dc)
    df1 = df1.append(df2, ignore_index=True)

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 8))

    librosa.display.waveplot(y=y, sr=sr, ax=ax[0])
    ax[0].set_ylabel('Amplitude', fontsize=12)
    ax[0].set_xlabel('Time', fontsize=12)
    ax[0].set_xscale('linear')
    ax[0].set_xlim([0, 1])

    ax[1].plot(times, ons, alpha=0.8, label='Onset strength')
    ax[1].set_ylabel('Normalized Amplitude', fontsize= 12)
    ax[1].vlines(times[peaks], 0, ons.max(), color='r', alpha=0.8, label='Selected peaks')
    ax[1].legend(frameon=True, framealpha=0.8)
    ax[1].axis('tight')
    ax[1].set_xlabel('Time', fontsize=12)
    ax[1].set_xlim([0, 1])

    D = librosa.stft(y)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max), y_axis='mel', x_axis='time', ax=ax[2])
    ax[2].set_xscale('linear')
    ax[2].set_ylabel('Hz', fontsize=12)
    ax[2].set_xlabel('Time', fontsize=12)
    ax[2].set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(f'{file}.png') # salva a figura

    print(plt.show())

df1 = df1.drop(0, axis=0)  # Delete a linha para se ter um data frame limpo 1 linha x 225 colunas
df1.to_excel('dados.xlsx') #salva o excel

