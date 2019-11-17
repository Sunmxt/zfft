# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import av
from scipy.fftpack import fft
from scipy.signal import firwin, fftconvolve, lfilter
import click
import _io
import time

import zfft


def cli_dump_media_info(av_container):
    print('Total duration: {}s\n'.format(av_container.duration/1000000))
    print('Audio tracks:')
    for track_id in range(len(av_container.streams.audio)):
        stream = av_container.streams.audio[track_id]
        show_meta = []
        for name, v in stream.metadata.items():
            show_meta.append('{}={}'.format(name, v))
        print('index {}: {}({}), {} channel, {}Hz ,{}'.format(track_id, stream.codec_context.codec.name, stream.codec_context.codec.long_name, stream.codec_context.channels, stream.codec_context.sample_rate, ', '.join(show_meta)))
        

def cli_open_av_container(file):
    if isinstance(file, _io.BufferedReader):
        filename = file.name
    elif isinstance(file, str):
        filename = file
    print('media file: {}'.format(filename))
    
    try:
        return av.open(file)
    except (av.AVError, FileNotFoundError) as e:
        print(e)
        return None
    
    
@click.group(help = 'spectrum analyze tool for audio and video')
@click.pass_context
def aspectrum(ctx):
    pass
    
@aspectrum.command(help = 'show metadata for media file.')
@click.pass_context
@click.argument('media_file', nargs = 1, type = click.File('rb'))
def probe(ctx, media_file):
    container = cli_open_av_container(media_file)
    if not container:
        return
    cli_dump_media_info(container)
    
    
@aspectrum.command(help = 'analyze audio sample for spectrum')
@click.pass_context
@click.argument('media_file', nargs = 1, type = click.File('rb'))
@click.option('-f', '--offset', default = 0, type = int, show_default = True, help = 'time offset in milliseconds.')
@click.option('-a', '--audio-track', default = 0, type = int, show_default = True, help = 'index of audio track')
@click.option('-c', '--channel', default=0, type = int, show_default = True, help = 'audio channel to analyze')
@click.option('-l', '--length', default=50, type = int, show_default = True, help = 'sample length in milliseconds')
@click.option('-b', '--zoom-begin-freq', default=0, type = int, show_default = True, help = 'starting frequency point to inspect')
@click.option('-e', '--zoom-end-freq', default=0, type = int, show_default = True, help = 'stopping frequency point to inspect')
def audio(ctx, media_file, offset, audio_track, channel, length, zoom_begin_freq, zoom_end_freq):
    container = cli_open_av_container(media_file)
    if not container:
        return
    if audio_track >= len(container.streams.audio):
        print('track {} not found.'.format(audio_track))
        return
    track = container.streams.audio[0]
    if offset > 0:
        begin_time_idx = int(offset * track.time_base.denominator / 1000 / track.time_base.numerator)
        container.seek(begin_time_idx)
    if channel >= track.codec_context.channels:
        print('channel {} not found.'.format(channel))
        return
    if zoom_end_freq < 1:
        zooming = False
        zoom_end_freq = track.codec_context.sample_rate // 2
    if zoom_begin_freq >= zoom_end_freq:
        print('starting zoom frequency should be greater then stopping zoom frequency.')
        return
    frames = container.decode(audio=audio_track)
    audio_nd = None
    concat_audio_frame = lambda nd: np.concatenate((audio_nd, nd)) if audio_nd is not None else nd
    num_of_pt = float(length) * track.codec_context.sample_rate / 1000
    for frame in frames:
        nd = frame.to_ndarray()[channel]
        audio_nd = concat_audio_frame(nd)
        num_of_pt -= len(nd)
        if num_of_pt <= 0:
            break
    t = np.arange(0, 1024000, 1) / track.codec_context.sample_rate
    audio_nd = np.sin(2*np.pi*199* t) + np.sin(2*np.pi*200.5* t)
    num_of_axs = 3#2 if zooming else 3
    fig, axs = plt.subplots(num_of_axs,1)
    x = np.arange(0, len(audio_nd), 1)#/ track.codec_context.sample_rate
    f_nd = fft(audio_nd)
    axs[0].plot(x/track.codec_context.sample_rate, audio_nd, linewidth = 0.2)
    axs[1].stem(x[:len(f_nd)//2]*track.codec_context.sample_rate/len(f_nd), np.abs(f_nd[:len(f_nd)//2]), markerfmt=" ", use_line_collection=True)
    freq, zf_nd = zfft.zoomfft(audio_nd, 195, 205, fs = track.codec_context.sample_rate)
    axs[2].stem(freq, np.abs(zf_nd), markerfmt = " ", use_line_collection=True)
    plt.pause(0)
    

#timescale = 1
#N = 2000
#freq = 100
#dt = timescale/N
#omega_base = 2*np.pi/timescale/N
#
#x = np.arange(0, N, 1)
#time = dt*x
#td = np.sin(omega_base*freq*x)
#fd = fft(td)
#
#fig, axs = plt.subplots(2,1)
#axs[0].plot(time, td)
#axs[1].stem(x, np.abs(fd))
#fig.show()
#
#plt.pause(0)

if __name__ == '__main__':
    aspectrum(obj={})
