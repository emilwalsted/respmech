#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 1, 12:38:56 2021

@author: Emil S. Walsted

------------------------------------------------------------------------ 
Automated EMG processing and calculationa
------------------------------------------------------------------------ 
(c) Copyright 2021 Emil Schwarz Walsted <emilwalsted@gmail.com>
The latest version of this code is available at GitHub: 
https://github.com/emilwalsted/respmech

This code processes EMG signals for subsequent analysis and is used from
respmech.py to analyse diaphragm EMG in humans. 

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
 
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>

Parts of the noise reduction code has been adapted, with permission, from code
written by Tim Sainburg (https://timsainburg.com).

Requires librosa: https://librosa.org/doc/latest/install.html

"""
import ntpath
import subprocess
import copy
import numpy as np
from numpy.core.fromnumeric import mean 
import scipy as sp
import scipy.io as sio  
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from collections import OrderedDict 
from scipy import signal
import warnings
warnings.filterwarnings("ignore")

plt.style.use('seaborn-white')

def calculate_rms(emgchannels, rms_s, samplingfrequency):
    rms = []
    intemg = []
    #intrms = []
    for emgch in emgchannels.T:
        
        #Root mean square
        rmsch = []
        rollingwindowsize = int(rms_s * samplingfrequency)
    
        rollstart = 0
        rollend = rollingwindowsize-1
        
        r = [] 
        try:
            for i in range(0+int(rollingwindowsize/2),len(emgch)-int(rollingwindowsize/2)):
                r += [np.sqrt(np.mean(np.square(emgch[i+rollstart:i+rollend])))]
            
            rmsch = np.max(r)
        except:
            #If breath is too small (i.e. invalid and should be ignored)
            rmsch = 0
            
        rms += [rmsch]

        #Integral
        intemgch = []
        intch = np.abs(emgch)
        xval = np.linspace(0, len(emgch)-1, len(emgch)) / samplingfrequency
        intemgch += [sp.integrate.simps(intch, xval)]

        intemg = np.append(intemg, intemgch)
    
    return rms + [np.max(rms), np.mean(rms)], list(intemg) + [np.max(intemg), np.mean(intemg)]
    
def timeshift(inputarray, shift):
    if (shift == 0) or (shift>=len(inputarray)):
        return inputarray
    else:
        ret = np.empty_like(inputarray)
        if shift >= 0:
            ret[:shift] = 0
            ret[shift:] = inputarray[:-shift]
        else:
            ret[shift:] = 0
            ret[:shift] = inputarray[-shift:]   
    return ret

def timeshift_average(ecgdata, avgdata, tmin, tmax):
    
    lowest = np.inf
    tlowest = 0
    tavg = None
    
    for t in range(tmin,tmax+1):
        shiftedavg = timeshift(avgdata, t)
        
        if type(ecgdata) is np.ndarray:
            ecg = ecgdata
        else:
            ecg = np.array(ecgdata)
             
        newl = np.sum(np.square(ecg - shiftedavg))
        if newl<lowest:
            lowest = newl
            tlowest=t     
            tavg = shiftedavg
            
    return tlowest, tavg

def amplitude_average(ecgdata, avgdata, rangefactor, steps):
    lowest = np.inf
    alowest = 0
    aavg = None
    amplitudes = np.linspace(-1 * rangefactor, rangefactor, steps)
    for a in amplitudes:
        ampavg = avgdata * a
        newa = np.sum(np.square(ecgdata-ampavg))
        if newa<lowest:
            lowest = newa
            alowest=a   
            aavg = ampavg
        
    return alowest, aavg

def subtractecg(ch, peaks, samplingfrequency, windowsize):

    emgecgchannels = list(np.array(ch).T)
    retch = []
    chno=0
    for emgecgch in emgecgchannels:
        chno += 1
        ecgwindows = []
              
        ecgwindowstart = int(windowsize * samplingfrequency/2)
        ecgwindowend = int(windowsize * samplingfrequency/2)
        shiftinterval = int(0.2 * samplingfrequency)

        #Create average ECG for the window
        for peak in peaks:

            if (peak-ecgwindowstart >= 0) and (peak+ecgwindowend <= len(emgecgch)): #only average whole windows
                ecgwindow = np.array(emgecgch[peak-ecgwindowstart:peak+ecgwindowend])

                #align averaging windows:
                if len(ecgwindows)>0:
                    _, ecgavgtime = timeshift_average(ecgwindows[0], ecgwindow, -shiftinterval, shiftinterval)
                    ecgwindows += [ecgavgtime]
                else:    
                    ecgwindows += [ecgwindow]

        ecgavg = np.mean(ecgwindows, axis=0)
   
        #Subtract average ECG from EMG
        peakno = 0
        retwindows = []
        for peak in peaks:
            peakno += 1

            partial = False
            if (peak-ecgwindowstart < 0) :
                ecgwindow = emgecgch[0:peak+ecgwindowend]
                emgecgch[0:peak+ecgwindowend] = list(np.array(ecgwindow) - np.array(ecgavg[len(ecgavg)-len(ecgwindow):len(ecgavg)]))
                retwindows += [[0, peak+ecgwindowend]]
                partial = True

            if (peak+ecgwindowend > len(emgecgch)):
                ecgwindow = emgecgch[peak-ecgwindowstart:len(emgecgch)-1]
                emgecgch[peak-ecgwindowstart:len(emgecgch)-1] = list(np.array(ecgwindow) - np.array(ecgavg[0:len(ecgwindow)]))
                retwindows += [[peak-ecgwindowstart, len(emgecgch)-1]]
                partial = True
                
            if partial == False:
                ecgwindow = np.array(emgecgch[peak-ecgwindowstart:peak+ecgwindowend])
                retwindows += [[peak-ecgwindowstart, peak+ecgwindowend]]
                
                _, ecgavgtime = timeshift_average(ecgwindow, ecgavg, -shiftinterval, shiftinterval)
                _, fittedavg = amplitude_average(ecgwindow, ecgavgtime, 1.25, 1000) 
                adjwindow = list(ecgwindow - fittedavg)
                
                #Diagnostics:
                if chno == 1:
                                  
                    plt.figure()
                    plt.plot(ecgwindow, label="Raw data", color='black', linewidth=0.5)
                    plt.plot(ecgavg, label="Average ECG", color='silver', linewidth=0.5)
                    plt.plot(ecgavgtime, label="Time shifted average ECG", color='orange', linewidth=0.5)
                    plt.plot(fittedavg, label="Amplitude adjusted average ECG", color='green', linewidth=0.5)
                    plt.plot(adjwindow, label="ECG reduced signal", color='r', linewidth=0.5)
                    plt.suptitle("Channel # " + str(chno) + ", ECG complex #" + str(peakno))   
                    plt.legend(loc="upper left")
                    plt.savefig("/Users/emilnielsen/Documents/Medicin/Forskning/RespMech demo/Output/diag/ch" + str(chno) + "_#" + str(peakno) + ".pdf")
                    plt.close()
                
                emgecgch[peak-ecgwindowstart:peak+ecgwindowend] =  adjwindow 
                
        retch += [emgecgch]
    return list(np.array(retch).T), retwindows

def remove_ecg(emgecgchannels, peakch, samplingfrequency, ecgminheight, ecgmindistance, ecgminwidth, windowsize):
    
    peaks, _ = signal.find_peaks(peakch, height=ecgminheight, distance=ecgmindistance*samplingfrequency, width=ecgminwidth*samplingfrequency)
    processedchannels, ecgwindows = subtractecg(emgecgchannels, peaks, samplingfrequency,  windowsize)

    return processedchannels, ecgwindows, peaks/samplingfrequency

    
def savesoundemg(emgchannel, wavpath):
    from scipy.io.wavfile import write
    
    scaled = np.int16(emgchannel/np.max(np.abs(emgchannel)) * 32767)
    write(wavpath, 4000, scaled)    


import time
from datetime import timedelta as td

def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y, n_fft, hop_length, win_length):
    ###y_pad = librosa.util.fix_length(y, len(y) + n_fft // 2)
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)


def plot_spectrogram(signal, title):
    fig, ax = plt.subplots(figsize=(20, 4))
    cax = ax.matshow(
        signal,
        origin="lower",
        aspect="auto",
        cmap="seismic",
        vmin=-1 * np.max(np.abs(signal)),
        vmax=np.max(np.abs(signal)),
    )
    fig.colorbar(cax)
    ax.set_title(title)
    plt.tight_layout()
    #plt.show()


def plot_statistics_and_filter(
    mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
):
    fig, ax = plt.subplots(ncols=2, figsize=(20, 4))
    ax[0].plot(mean_freq_noise, label="Mean power of noise")
    ax[0].plot(std_freq_noise, label="Std. power of noise")
    ax[0].plot(noise_thresh, label="Noise threshold (by frequency)")
    ax[0].set_title("Threshold for mask")
    ax[0].legend()
    cax = ax[1].matshow(smoothing_filter, origin="lower")
    fig.colorbar(cax)
    ax[1].set_title("Filter for smoothing Mask")
    #plt.show()


def removeNoise(
    audio_clip,
    noise_clip,
    n_grad_freq=0, #12
    n_grad_time=4, #4
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    n_std_thresh=2,
    prop_decrease=1,
    verbose=True,
    visual=True,
):
    """Remove noise from audio based upon a clip containing only noise

    Args:
        audio_clip (array): The first parameter.
        noise_clip (array): The second parameter.
        n_grad_freq (int): how many frequency channels to smooth over with the mask.
        n_grad_time (int): how many time channels to smooth over with the mask.
        n_fft (int): number audio of frames between STFT columns.
        win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
        hop_length (int):number audio of frames between STFT columns.
        n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
        prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none)
        visual (bool): Whether to plot the steps of the algorithm

    Returns:
        array: The recovered signal with noise subtracted

    """

    if verbose:
        start = time.time()
    # STFT over noise
    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  # convert to dB
    # Calculate statistics over noise
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    if verbose:
        print("STFT on noise:", td(seconds=time.time() - start))
        start = time.time()
    # STFT over signal
    if verbose:
        start = time.time()
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))
    if verbose:
        print("STFT on signal:", td(seconds=time.time() - start))
        start = time.time()
    # Calculate value to mask dB to
    mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
    print(noise_thresh, mask_gain_dB)
    # Create a smoothing filter for the mask in time and frequency
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    # calculate the threshold for each frequency/time bin
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T
    # mask if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh
    if verbose:
        print("Masking:", td(seconds=time.time() - start))
        start = time.time()
    # convolve the mask with a smoothing filter
    sig_mask = sp.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease
    if verbose:
        print("Mask convolution:", td(seconds=time.time() - start))
        start = time.time()
    # mask the signal
    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )  # mask real
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
        1j * sig_imag_masked
    )
    if verbose:
        print("Mask application:", td(seconds=time.time() - start))
        start = time.time()
    # recover the signal
    recovered_signal = _istft(sig_stft_amp, n_fft, hop_length, win_length)
    recovered_spec = _amp_to_db(
        np.abs(_stft(recovered_signal, n_fft, hop_length, win_length))
    )
    if verbose:
        print("Signal recovery:", td(seconds=time.time() - start))
    if visual:
        plot_spectrogram(noise_stft_db, title="Noise")
    if visual:
        plot_statistics_and_filter(
            mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
        )
    if visual:
        plot_spectrogram(sig_stft_db, title="Signal")
    if visual:
        plot_spectrogram(sig_mask, title="Mask applied")
    if visual:
        plot_spectrogram(sig_stft_db_masked, title="Masked signal")
    if visual:
        plot_spectrogram(recovered_spec, title="Recovered spectrogram")
    return recovered_signal

# from https://stackoverflow.com/questions/33933842/how-to-generate-noise-in-frequency-range-with-numpy
def fftnoise(f):
    f = np.array(f, dtype="complex")
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1 : Np + 1] *= phases
    f[-1 : -1 - Np : -1] = np.conj(f[1 : Np + 1])
    return np.fft.ifft(f).real


def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
    f = np.zeros(samples)
    f[np.logical_and(freqs >= min_freq, freqs <= max_freq)] = 1
    return fftnoise(f)

def reducenoise(emgchannel, noiseprofile, noiseprofilecolumn, samplingfrequency):
    #from scipy.io import wavfile
    #rate, data = wavfile.read(wavpath)
    #data = data / 32768    
    import sys, os
    sys.stdout = open(os.devnull, "w")

    if len(noiseprofilecolumn) == 0:
        noiseprofilecolumn = emgchannel

    noise = np.array(noiseprofilecolumn[int(noiseprofile[0]*samplingfrequency):int(noiseprofile[1]*samplingfrequency)])
    output = removeNoise(audio_clip=emgchannel, noise_clip=noise, verbose=False, visual=False, n_fft=len(noise)**2)

    sys.stdout = sys.__stdout__
    return np.array(output)
    
def saveemgplots(outpdf, breaths, timecol, rows, titles, ylabels, title, rms=[], rmsint=[], refsig=[], reflabel="Reference", ylim=[], ecgwindows=[], peaks=[]):   

    plt.ioff()
    norows = len(rows[0])
    fig, _ = plt.subplots(nrows=norows, ncols=1, sharex=True, figsize=(29.7, 21))
    fig.suptitle(ntpath.basename(outpdf), fontsize=48)    
    
    for i in range(0, norows):
        ax = plt.subplot(norows, 1, i+1)
                
        ax2 = ax.twinx()

        ax.plot(timecol, rows[:,i], 'b-', linewidth=0.25, label="EMG")
       
        ax.grid(True, which="major", color="black", linewidth='0.5', linestyle="-")
        ax.grid(True, which="minor", color="#D4D4D4", linewidth='0.25', linestyle="-")
        
        #Plot ECG removal windows
        yl = list(ax.get_ylim())
        for ecgwindow in ecgwindows:
            poly = Rectangle([ecgwindow[0], ylim[0]], ecgwindow[1]-ecgwindow[0], ylim[1]-ylim[0], alpha=0.05, color="#0000FF", fill=True)
            ax.add_patch(poly)         

        ax.scatter(peaks, np.full((len(peaks),1),ylim[1]*0.99), s=20, c='r', marker=11)


        #Plot reference signal
        if len(refsig)>0:
            ax2.plot(timecol, refsig, 'r-',  linewidth=0.5, alpha=1, color="#FF0000", label=reflabel)
            #ax2.set_ylabel(r'mcV $\cdot$ 0.5$s^{-1}$',fontweight="bold", size=16)
            ax2.legend(loc="upper right")

        #Plot RMS
        if len(rms)>0:
            ax.plot(timecol, rms[:,i], 'm-', linewidth=1, label=r'$\frac{\sum_{x-25ms}^{x+25ms}\sqrt{EMG^2}}{50ms}$')
        
        ax.set_ylim(ylim)
        ax.set_xlim([min(timecol), max(timecol)])
        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', bottom=True, direction='out', length=8)
        
        yl = list(ax.get_ylim())
        for breathno in breaths:
            blab = " #" + str(breathno)
            breath= breaths[breathno]
            if breath["ignored"]:
                blab += " (ignored)"
                poly = Rectangle([breath["time"][0], yl[0]],[breath["time"][len(breath["time"])-1]]-breath["time"][0], yl[1]-yl[0], alpha=0.1, color="#FF0000", fill=True)
                ax.add_patch(poly)
            
            ax.axvline(x=breath["time"][0], linewidth=0.5, linestyle="--", color="k")
            ax.text(breath["time"][0], yl[1]-((yl[1]-yl[0])*0.05), blab, fontsize=12)
        
        ax.set_title(titles[i],fontweight="bold", size=20)
        ax.set_ylabel(ylabels[i],fontweight="bold", size=16)
        
        ax.legend(loc="upper left")
        
    ax.set_xlabel(r'$time (seconds)$', size=16)

    fig.savefig(outpdf)
    plt.close(fig)
    

import librosa
import librosa.display as libdisp
from scipy.io import wavfile
import ntpath

def plotRawWave(plotTitle, sampleRate, samples, figWidth=14, figHeight=4):
    fig = plt.figure(figsize=(figWidth, figHeight))
    plt.plot(np.linspace(0, len(samples)/sampleRate, len(samples)), samples, linewidth=0.1)
    plt.title("Raw sound wave of " + plotTitle)
    plt.ylabel("Amplitude")
    plt.xlabel("Time [sec]")
    #plt.show()  # force display while in for loop
    fig.savefig(str.replace(plotTitle, ".wav", ".waveform.pdf"))
    return None

def computeLogSpectrogram(audio, sampleRate, windowSize=100, stepSize=10, epsilon=1e-10):
    nperseg  = int(round(windowSize * sampleRate / 1000))
    noverlap = int(round(stepSize   * sampleRate / 1000))
    freqs, times, spec = signal.spectrogram(audio,
                                            fs=sampleRate,
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + epsilon)

def plotLogSpectrogram(plotTitle, freqs, times, spectrogram, figWidth=14, figHeight=4):
    fig = plt.figure(figsize=(figWidth, figHeight))
    plt.imshow(spectrogram.T, aspect='auto', origin='lower', 
               cmap="inferno",   #  default was "viridis"  (perceptually uniform)
               extent=[times.min(), times.max(), freqs.min(), freqs.max()])
    plt.colorbar(pad=0.01)
    plt.title('Spectrogram of ' + ntpath.basename(plotTitle))
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    fig.tight_layout()
    #plt.show()  # force display while in for loop
    fig.savefig(str.replace(plotTitle, ".wav", ".spectrogram.pdf"))
    return None

def computeLogMelSpectrogram(samples, sampleRate, nMels=256):
    melSpectrum = librosa.feature.melspectrogram(samples.astype(float), sr=sampleRate, n_mels=nMels)
    
    # Convert to dB, which is a log scale.  Use peak power as reference.
    logMelSpectrogram = librosa.power_to_db(melSpectrum, ref=np.max)
    
    return logMelSpectrogram

def plotLogMelSpectrogram(plotTitle, sampleRate, logMelSpectrum, figWidth=14, figHeight=4):
    fig = plt.figure(figsize=(figWidth, figHeight))
    libdisp.specshow(logMelSpectrum, sr=sampleRate, x_axis='time', y_axis='mel')
    plt.title('Mel log-frequency power spectrogram: ' + ntpath.basename(plotTitle))
    plt.colorbar(pad=0.01, format='%+02.0f dB')
    plt.tight_layout()  
    #plt.show()  # force display while in for loop
    fig.savefig(str.replace(plotTitle, ".wav", ".log-freq spectogram.pdf"))
    return None

def computeMFCC(samples, sampleRate, nFFT=512, hopLength=16, nMFCC=50):
    mfcc = librosa.feature.mfcc(y=samples.astype(float), sr=sampleRate, n_fft=nFFT, hop_length=hopLength, n_mfcc=nMFCC)
    
    # Let's add on the first and second deltas  (what is this really doing?)
    #mfcc = librosa.feature.delta(mfcc, order=2)
    return mfcc

def plotMFCC(plotTitle, sampleRate, mfcc, figWidth=14, figHeight=4):
    fig = plt.figure(figsize=(figWidth, figHeight))
    librosa.display.specshow(mfcc, sr=sampleRate, x_axis='time', y_axis='mel')
    plt.colorbar(pad=0.01)
    plt.title("Mel-frequency cepstral coefficients (MFCC): " + ntpath.basename(plotTitle))
    plt.tight_layout()
    #plt.show()  # force display while in for loop
    fig.savefig(str.replace(plotTitle, ".wav", ".mel-freq cepstral coef.pdf"))
    return None

def showWavefile(filename):
    sampleRate, samples = wavfile.read(filename)  
    plotRawWave(filename, sampleRate, samples)
    
    freqs, times, logSpectrogram = computeLogSpectrogram(samples, sampleRate)
    plotLogSpectrogram(filename, freqs, times, logSpectrogram)
    
    logMelSpectrogram = computeLogMelSpectrogram(samples, sampleRate)
    plotLogMelSpectrogram(filename, sampleRate, logMelSpectrogram)
    
    mfcc = computeMFCC(samples, sampleRate)
    #print(mfcc.shape)
    plotMFCC(filename, sampleRate, mfcc)
    
    return sampleRate, samples, logSpectrogram, logMelSpectrogram, mfcc

