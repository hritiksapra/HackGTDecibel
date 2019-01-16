from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from flask import url_for
from flask import flash
from werkzeug.utils import secure_filename
import sys
import os
from pydub import AudioSegment
import librosa
from google.cloud import texttospeech
import json
import numpy as np
import wave
import math
import contextlib
import librosa.display
from dtw import dtw
from numpy.linalg import norm
from pygame import mixer
from mutagen.mp3 import MP3
import random

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "yougo.json"
UPLOAD_FOLDER = './outputaudio'
ALLOWED_EXTENSIONS = set(['mp3'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key = "#khlkjh@"
# if __name__ == "__main__":
#     app.run(host='0.0.0.0')

languagePicked = False
varsToSend = []
fileRead = []
randomSentence = ""
lang_code = ""
textSpoken = ""

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@app.route("/language", methods = ['POST', 'GET'])
def index():
    global languagePicked
    return render_template('PickLang.html')


def readFile(lang):
    global randomSentence
    # global fileRead, randomSentence
    # file = open("./Corpora/" + languagePicked + '.txt', 'r')
    # for line in file.readlines():
    #     fileRead.append(line)
    #
    # randomSentence = random.choice(fileRead)
    # randomSentence = str(randomSentence)
    # file.close()
    array = ["Can I have some juice to drink?", "Où sont tes enfants ?", "Der Frosch sprang und landete im Teich.", "Stare con le mani in mano.", "Cuando llevamos nosotros máscaras?"]

    if languagePicked == "French":
        array = [array[1]]
    elif languagePicked == "English":
        array = [array[0]]
    elif languagePicked == "German":
        array = [array[2]]
    elif languagePicked == "Italian":
        array = [array[3]]
    elif languagePicked == "Spanish":
        array = [array[4]]
    randomSentence = array[0][0]

@app.route("/parseInput", methods = ['POST'])
def parseInput():
    global languagePicked
    global lang_code
    if request.method == 'POST':
        if request.form['submit_button'] == 'English':
            languagePicked = "English"
            lang_code = 'en-US'
        elif request.form['submit_button'] == 'Spanish':
            languagePicked = "Spanish"
            lang_code = 'es-ES'
        elif request.form['submit_button'] == 'French':
            languagePicked = "French"
            lang_code = 'fr-FR'
        elif request.form['submit_button'] == 'Japanese':
            languagePicked = "Japanese"
            lang_code = 'ja-JP'
        elif request.form['submit_button'] == 'German':
            languagePicked = "German"
            lang_code = 'de-DE'
        elif request.form['submit_button'] == 'Portuguese':
            languagePicked = "Portuguese"
            lang_code = 'pt-BR'
        elif request.form['submit_button'] == 'Korean':
            languagePicked = "Korean"
            lang_code = 'ko-KR'
        elif request.form['submit_button'] == 'Italian':
            languagePicked = "Italian"
            lang_code = 'it-IT'
        elif request.form['submit_button'] == 'Swedish':
            languagePicked = "Swedish"
            lang_code = 'sv-SE'
        elif request.form['submit_button'] == 'Turkish':
            languagePicked = "Turkish"
            lang_code = 'tr-TR'
        readFile(languagePicked)
    print(languagePicked, file=sys.stdout)
    return redirect(url_for('showText'))

@app.route("/test", methods=['POST'])
@app.route("/test/<varsToSend>", methods = ['POST'])
def test(varsToSend=None):
    array = ["Hello, World"]

    # Instantiates a client
    client = texttospeech.TextToSpeechClient()

    # Set the text input to be synthesized
    i = 0
    for x in array:
        synthesis_input = texttospeech.types.SynthesisInput(text=x)
        # Build the voice request, select the language code ("en-US") and the ssml
        # voice gender ("neutral")
        voice = texttospeech.types.VoiceSelectionParams(
        language_code='en-US',
        ssml_gender=texttospeech.enums.SsmlVoiceGender.FEMALE)

        # Select the type of audio file you want returned
        audio_config = texttospeech.types.AudioConfig(
        audio_encoding=texttospeech.enums.AudioEncoding.MP3)

        # Perform the text-to-speech request on the text input with the selected
        # voice parameters and audio file type
        response = client.synthesize_speech(synthesis_input, voice, audio_config)

        # The response's audio_content is binary.
        with open('outputaudio/output.mp3', 'wb') as out:
            out.write(response.audio_content)
            print('Audio content written to file output.mp3')

    sound = AudioSegment.from_mp3("outputaudio/output.mp3")
    sound.export("outputaudio/output.wav", format="wav")
    fname = 'outputaudio/output.wav'
    outname = 'outputaudio/filtered.wav'
    cutOffFrequency = 400.0

    # from http://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    def running_mean(x, windowSize):
      cumsum = np.cumsum(np.insert(x, 0, 0))
      return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize

    # from http://stackoverflow.com/questions/2226853/interpreting-wav-data/2227174#2227174
    def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved = True):

        if sample_width == 1:
            dtype = np.uint8 # unsigned char
        elif sample_width == 2:
            dtype = np.int16 # signed 2-byte short
        else:
            raise ValueError("Only supports 8 and 16 bit audio formats.")

        channels = np.fromstring(raw_bytes, dtype=dtype)

        if interleaved:
            # channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
            channels.shape = (n_frames, n_channels)
            channels = channels.T
        else:
            # channels are not interleaved. All samples from channel M occur before all samples from channel M-1
            channels.shape = (n_channels, n_frames)

        return channels

    with contextlib.closing(wave.open(fname,'rb')) as spf:
        sampleRate = spf.getframerate()
        ampWidth = spf.getsampwidth()
        nChannels = spf.getnchannels()
        nFrames = spf.getnframes()

        # Extract Raw Audio from multi-channel Wav File
        signal = spf.readframes(nFrames*nChannels)
        spf.close()
        channels = interpret_wav(signal, nFrames, nChannels, ampWidth, True)

        # get window size
        # from http://dsp.stackexchange.com/questions/9966/what-is-the-cut-off-frequency-of-a-moving-average-filter
        freqRatio = (cutOffFrequency/sampleRate)
        N = int(math.sqrt(0.196196 + freqRatio**2)/freqRatio)

        # Use moviung average (only on first channel)
        filtered = running_mean(channels[0], N).astype(channels.dtype)

        wav_file = wave.open(outname, "w")
        wav_file.setparams((1, ampWidth, sampleRate, nFrames, spf.getcomptype(), spf.getcompname()))
        wav_file.writeframes(filtered.tobytes('C'))
        wav_file.close()

    sound = AudioSegment.from_mp3("outputaudio/input.mp3")
    sound.export("outputaudio/input.wav", format="wav")
    fname = 'outputaudio/input.wav'
    outname = 'outputaudio/filteredinput.wav'
    cutOffFrequency = 400.0

    # from http://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    def running_mean(x, windowSize):
      cumsum = np.cumsum(np.insert(x, 0, 0))
      return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize

    # from http://stackoverflow.com/questions/2226853/interpreting-wav-data/2227174#2227174
    def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved = True):

        if sample_width == 1:
            dtype = np.uint8 # unsigned char
        elif sample_width == 2:
            dtype = np.int16 # signed 2-byte short
        else:
            raise ValueError("Only supports 8 and 16 bit audio formats.")

        channels = np.fromstring(raw_bytes, dtype=dtype)

        if interleaved:
            # channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
            channels.shape = (n_frames, n_channels)
            channels = channels.T
        else:
            # channels are not interleaved. All samples from channel M occur before all samples from channel M-1
            channels.shape = (n_channels, n_frames)

        return channels

    with contextlib.closing(wave.open(fname,'rb')) as spf:
        sampleRate = spf.getframerate()
        ampWidth = spf.getsampwidth()
        nChannels = spf.getnchannels()
        nFrames = spf.getnframes()

        # Extract Raw Audio from multi-channel Wav File
        signal = spf.readframes(nFrames*nChannels)
        spf.close()
        channels = interpret_wav(signal, nFrames, nChannels, ampWidth, True)

        # get window size
        # from http://dsp.stackexchange.com/questions/9966/what-is-the-cut-off-frequency-of-a-moving-average-filter
        freqRatio = (cutOffFrequency/sampleRate)
        N = int(math.sqrt(0.196196 + freqRatio**2)/freqRatio)

        # Use moviung average (only on first channel)
        filtered = running_mean(channels[0], N).astype(channels.dtype)

        wav_file = wave.open(outname, "w")
        wav_file.setparams((1, ampWidth, sampleRate, nFrames, spf.getcomptype(), spf.getcompname()))
        wav_file.writeframes(filtered.tobytes('C'))
        wav_file.close()

    #Loading audio files
    y1, sr1 = librosa.load('outputaudio/filtered.wav')
    y2, sr2 = librosa.load('outputaudio/filteredinput.wav')

    #Showing multiple plots using subplot
    mfcc1 = librosa.feature.mfcc(y1,sr1)   #Computing MFCC values

    mfcc2 = librosa.feature.mfcc(y2, sr2)

    dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
    print("The normalized distance between the two : ",dist)   # 0 for similar audios

    percent = (1000 - dist)/1000
    percentage = percent*100
    print("Percentage Similarity : "+str(percentage)+"%")
    percentageSimularity = percentage
    # mixer.init()
    # mixer.music.load("./outputaudio/output.mp3")
    # mixer.music.play()
    audioToCheck = MP3("./outputaudio/output.mp3")
    audioLength = audioToCheck.info.length;

    global textSpoken

    textSpoken = array[0]
    durationEachCharOut = len(textSpoken)/audioLength
    durationEachCharIn = len(textSpoken)/audioLength
    if ',' in textSpoken or '.' in textSpoken:
        durationEachCharOut += 30;
    print(durationEachCharOut, file=sys.stdout)
    varsToSend = [percentageSimularity, audioLength, durationEachCharOut, durationEachCharIn, textSpoken]
    print(textSpoken, file=sys.stdout)

    return render_template('test.html', varsToSend = varsToSend)

@app.route("/playFile", methods=['GET', 'POST'])
@app.route("/playFile<varsToSend>", methods=['GET', 'POST'])
def playFile(varsToSend=None):
    array = ["Can I have some juice to drink?", "Où sont tes enfants ?", "Der Frosch sprang und landete im Teich.", "Stare con le mani in mano.", "Cuando llevamos nosotros máscaras?"]

    if languagePicked == "French":
        array = [array[1]]
    elif languagePicked == "English":
        array = [array[0]]
    elif languagePicked == "German":
        array = [array[2]]
    elif languagePicked == "Italian":
        array = [array[3]]
    elif languagePicked == "Spanish":
        array = [array[4]]

    print(array, file=sys.stdout)
    # Instantiates a client
    client = texttospeech.TextToSpeechClient()
    # Set the text input to be synthesized
    i = 0
    for x in array:
        synthesis_input = texttospeech.types.SynthesisInput(text=x)
        # Build the voice request, select the language code ("en-US") and the ssml
        # voice gender ("neutral")
        voice = texttospeech.types.VoiceSelectionParams(
            language_code=lang_code,
            ssml_gender=texttospeech.enums.SsmlVoiceGender.FEMALE)
        # Select the type of audio file you want returned
        audio_config = texttospeech.types.AudioConfig(
            audio_encoding=texttospeech.enums.AudioEncoding.MP3)
        # Perform the text-to-speech request on the text input with the selected
        # voice parameters and audio file type
        response = client.synthesize_speech(synthesis_input, voice, audio_config)
        # The response's audio_content is binary.
        with open('./outputaudio/output.mp3', 'wb') as out:
            out.write(response.audio_content)
            print('Audio content written to file output.mp3')
            out.close()
    mixer.init()
    mixer.music.load("./outputaudio/output.mp3")
    audioToCheck = MP3("./outputaudio/output.mp3")
    audioLength = audioToCheck.info.length
    textSpoken = array[0]
    durationEachCharOut = len(textSpoken) / audioLength
    durationEachCharIn = len(textSpoken) / audioLength
    if ',' in textSpoken or '.' in textSpoken:
        durationEachCharOut += 30
    print(durationEachCharOut, file=sys.stdout)
    varsToSend = [0, audioLength, durationEachCharOut, durationEachCharIn, textSpoken]
    mixer.music.play()
    return render_template('showText.html', varsToSend=varsToSend)


@app.route("/showText", methods=['GET', 'POST'])
@app.route("/showText<varsToSend>", methods=['GET', 'POST'])
def showText(varsToSend=None):
    return render_template("showText.html", varsToSend=varsToSend)


@app.route("/uploadFile", methods=['GET', 'POST'])
@app.route("/uploadFile<textToSpeak>", methods=['POST'])
def uploadFile(textToSpeak=None):
    textToSpeak = randomSentence
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            print("No file part")
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            ext = filename[-1:-4:-1]
            ext = ext[::-1]
            filename = "input." + ext
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploadFile'))
    return render_template('uploadFile.html', textToSpeak=textToSpeak)

@app.route("/analyze")
def analyze():
    return render_template("analyze.html")
