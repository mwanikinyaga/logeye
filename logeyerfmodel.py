# In[1]:


import numpy as np
import pydub
import librosa

class Clip:
    """A single 5-sec long recording. 
    Single channel, Ogg Vorbis compressed @ 192 kbit/s"""

    
    RATE = 44100   # All recordings in ESC are 44.1 kHz
    FRAME = 512    # Frame size in samples

 #  A helper class to compute baseline features   
    class Audio:
        """The actual audio data of the clip.
        
            Uses a context manager to load/unload the raw audio data. This way clips
            can be processed sequentially with reasonable memory usage.
        """
        
        def __init__(self, path):
            self.path = path
        
        def __enter__(self):
            # Actual recordings are sometimes not frame accurate, so we trim/overlay to exactly 5 seconds
            self.data = pydub.AudioSegment.silent(duration=5000)
            self.data = self.data.overlay(pydub.AudioSegment.from_file(self.path)[0:5000])
            self.raw = (np.frombuffer(self.data._data, dtype="int16") + 0.5) / (0x7FFF + 0.5)   # convert to float
            return(self)
        
        def __exit__(self, exception_type, exception_value, traceback):
            if exception_type is not None:
                print (exception_type, exception_value, traceback)
            del self.data
            del self.raw
        
    def __init__(self, filename):
        self.filename = os.path.basename(filename)
        self.path = os.path.abspath(filename)        
        self.directory = os.path.dirname(self.path)
        self.category = self.directory.split('/')[-1]
        
        self.audio = Clip.Audio(self.path)
        
        with self.audio as audio:
            self._compute_mfcc(audio)    
            self._compute_zcr(audio)
            
    def _compute_mfcc(self, audio):
        # MFCC computation with default settings (2048 FFT window length, 512 hop length, 128 bands)
        self.melspectrogram = librosa.feature.melspectrogram(audio.raw, sr=Clip.RATE, hop_length=Clip.FRAME)
        self.amplitude_to_db = librosa.amplitude_to_db(self.melspectrogram)
        self.mfcc = librosa.feature.mfcc(S=self.amplitude_to_db, n_mfcc=13).transpose()
            
    def _compute_zcr(self, audio):
        # Zero-crossing rate
        self.zcr = []
        frames = int(np.ceil(len(audio.data) / 1000.0 * Clip.RATE / Clip.FRAME))
        
        for i in range(0, frames):
            frame = Clip._get_frame(audio, i)
            self.zcr.append(np.mean(0.5 * np.abs(np.diff(np.sign(frame)))))

        self.zcr = np.asarray(self.zcr)
            
    @classmethod
    def _get_frame(cls, audio, index):
        if index < 0:
            return None
        return audio.raw[(index * Clip.FRAME):(index+1) * Clip.FRAME]
    
    def __repr__(self):
        return '<{0}/{1}>'.format(self.category, self.filename)


# Looking at some specific recordings
# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sb
sb.set(style="white", palette="muted")

import pandas as pd


# In[3]:


import random
random.seed(20150420)


# In[4]:


# Reload this cell to get a different clip at every try
import glob
import os
import IPython.display
import librosa.display

# ESC-50 is a labeled set of 2 000 environmental recordings with 50 classes and 40 clips per class
all_recordings = glob.glob('ESC-50/*/*.ogg')
clip = Clip(all_recordings[random.randint(0, len(all_recordings) - 1)])    

with clip.audio as audio:
    plt.subplot(2, 1, 1)
    plt.title('{0} : {1}'.format(clip.category, clip.filename))
    plt.plot(np.arange(0, len(audio.raw)) / 44100.0, audio.raw)
   
    plt.subplot(2, 1, 2)
    librosa.display.specshow(clip.amplitude_to_db, sr=44100, x_axis='frames', y_axis='linear', cmap='RdBu_r')
    
IPython.display.Audio(filename=clip.path, rate=Clip.RATE)   


# Loads all recordings for 5 classes
"""Crackling Fire, Chainsaw, Car horn, Engine, Hand saw"""
# In[15]:


import os

def load_dataset(name):
    """Load all dataset recordings into a nested list."""
    clips = []

    included_prefix = ['203','502','504','505','510']

    for directory in sorted(os.listdir('{0}/'.format(name))):
        if any(directory.startswith(pre) for pre in included_prefix):
            directory = '{0}/{1}'.format(name, directory)
            if os.path.isdir(directory) and os.path.basename(directory)[0:3].isdigit():
                print('Parsing ' + directory)
                category = []
                for clip in sorted(os.listdir(directory)):
                    if clip[-3:] == 'ogg':
                        category.append(Clip('{0}/{1}'.format(directory, clip)))
                clips.append(category)
                
    IPython.display.clear_output()
    print('All {0} recordings loaded.'.format(name))            
    
    return clips

clips_50 = load_dataset('ESC-50')

# An overview of waveforms on corresponding mel-spectrograms
# In[16]:


def add_subplot_axes(ax, position):
    box = ax.get_position()
    
    position_display = ax.transAxes.transform(position[0:2])
    position_fig = plt.gcf().transFigure.inverted().transform(position_display)
    x = position_fig[0]
    y = position_fig[1]
    
    return plt.gcf().add_axes([x, y, box.width * position[2], box.height * position[3]], facecolor='w')


# In[17]:


def plot_clip_overview(clip, ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax_waveform = add_subplot_axes(ax, [0.0, 0.7, 1.0, 0.3])
    ax_spectrogram = add_subplot_axes(ax, [0.0, 0.0, 1.0, 0.7])
    
    with clip.audio as audio:
        ax_waveform.plot(np.arange(0, len(audio.raw)) / float(Clip.RATE), audio.raw)
        ax_waveform.get_xaxis().set_visible(False)
        ax_waveform.get_yaxis().set_visible(False)
        ax_waveform.set_title('{0} \n {1}'.format(clip.category, clip.filename), {'fontsize': 8}, y=1.03)
        
        librosa.display.specshow(clip.amplitude_to_db, sr=Clip.RATE, x_axis='time', y_axis='mel', cmap='RdBu_r')
        ax_spectrogram.get_xaxis().set_visible(False)
        ax_spectrogram.get_yaxis().set_visible(False)

categories = 5
clips_shown = 7
f, axes = plt.subplots(categories, clips_shown, figsize=(clips_shown * 2, categories * 2), sharex=True, sharey=True)
f.subplots_adjust(hspace = 0.35)

for c in range(0, categories):
    for i in range(0, clips_shown):
        plot_clip_overview(clips_10[c][i], axes[c, i])

# Look into distribution of computed baseline features
#  Single clip / All features perspective
# In[39]:


def plot_single_clip(clip):
    col_names = list('MFCC_{}'.format(i) for i in range(np.shape(clip.mfcc)[1]))
    MFCC = pd.DataFrame(clip.mfcc[:, :], columns=col_names)

    f = plt.figure(figsize=(10, 6))
    ax = f.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    ax_mfcc = add_subplot_axes(ax, [0.0, 0.0, 1.0, 0.75])
    ax_mfcc.set_xlim(-400, 400)
    ax_zcr = add_subplot_axes(ax, [0.0, 0.85, 1.0, 0.05])
    ax_zcr.set_xlim(0.0, 1.0)

    plt.title('Feature distribution across frames of a single clip ({0} : {1})'.format(clip.category, clip.filename), y=1.5)
    sb.boxplot(MFCC, order=list(reversed(MFCC.columns)), ax=ax_mfcc)
    sb.boxplot(pd.DataFrame(clip.zcr, columns=['ZCR']), ax=ax_zcr)
    
plot_single_clip(clips_50[4][2])


# All clips / Single feature
# In[51]:


def plot_single_feature_one_clip(feature, title, ax):
    sb.despine()
    ax.set_title(title, y=1.10)
    sb.distplot(feature, bins=20, hist=True, rug=False,
                hist_kws={"histtype": "stepfilled", "alpha": 0.5},
                kde_kws={"shade": False},
                color=sb.color_palette("muted", 4)[2], ax=ax)

def plot_single_feature_all_clips(feature, title, ax):
    sb.despine()
    ax.set_title(title, y=1.03)
    sb.boxplot(feature, order=list(reversed(feature.columns)), ax=ax)

def plot_single_feature_aggregate(feature, title, ax):    
    sb.despine()
    ax.set_title(title, y=1.03)
    sb.distplot(feature, bins=20, hist=True, rug=False,
                hist_kws={"histtype": "stepfilled", "alpha": 0.5},
                kde_kws={"shade": False},
                color=sb.color_palette("muted", 4)[1], ax=ax)

def generate_feature_summary(category, clip, coefficient):
    title = "{0} : {1}".format(clips_50[category][clip].category, clips_50[category][clip].filename)
    MFCC = pd.DataFrame()
    aggregate = []
    for i in range(0, len(clips_50[category])):
        MFCC[i] = clips_50[category][i].mfcc[:, coefficient]
        aggregate = np.concatenate([aggregate, clips_50[category][i].mfcc[:, coefficient]])    

    f = plt.figure(figsize=(14, 12))
    f.subplots_adjust(hspace=0.6, wspace=0.3)

    ax1 = plt.subplot2grid((3, 3), (0, 0))
    ax2 = plt.subplot2grid((3, 3), (1, 0))
    ax3 = plt.subplot2grid((3, 3), (0, 1), rowspan=2)
    ax4 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)

    ax1.set_xlim(0.0, 0.5)
    ax2.set_xlim(-100, 250)
    ax4.set_xlim(-100, 250)
    
    plot_single_feature_one_clip(clips_50[category][clip].zcr, 'ZCR distribution across frames\n{0}'.format(title), ax1)
    plot_single_feature_one_clip(clips_50[category][clip].mfcc[:, coefficient], 'MFCC_{0} distribution across frames\n{1}'.format(coefficient, title), ax2)

    plot_single_feature_all_clips(MFCC, 'Differences in MFCC_{0} distribution\nbetween clips of {1}'.format(coefficient, clips_50[category][clip].category), ax3)

    plot_single_feature_aggregate(aggregate, 'Aggregate MFCC_{0} distribution\n(bag-of-frames across all clips\nof {1})'.format(coefficient, clips_50[category][clip].category), ax4)
    
generate_feature_summary(0, 0, 1)    


# In[61]:


# All clips / All features

def plot_all_features_aggregate(clips, ax):
    ax_mfcc = add_subplot_axes(ax, [0.0, 0.0, 0.85, 1.0])
    ax_zcr = add_subplot_axes(ax, [0.9, 0.0, 0.1, 1.0])
    
    sb.set_style('ticks')
    
    col_names = list('MFCC_{}'.format(i) for i in range(np.shape(clips[0].mfcc)[1]))
    aggregated_mfcc = pd.DataFrame(clips[0].mfcc[:, :], columns=col_names)

    for i in range(1, len(clips)):
        aggregated_mfcc = aggregated_mfcc.append(pd.DataFrame(clips[i].mfcc[:, :], columns=col_names))
        
    aggregated_zcr = pd.DataFrame(clips[0].zcr, columns=['ZCR']) 
    for i in range(1, len(clips)):
        aggregated_zcr = aggregated_zcr.append(pd.DataFrame(clips[i].zcr, columns=['ZCR']))
    
    sb.despine(ax=ax_mfcc)
    ax.set_title('Aggregate distribution: {0}'.format(clips[0].category), y=1.10, fontsize=10)
    sb.boxplot(aggregated_mfcc, order=aggregated_mfcc.columns, ax=ax_mfcc)
    ax_mfcc.set_xticklabels(range(0, 13), rotation=90, fontsize=8)
    ax_mfcc.set_xlabel('MFCC', fontsize=8)
    ax_mfcc.set_ylim(-150, 200)
    ax_mfcc.set_yticks((-150, -100, -50, 0, 50, 100, 150, 200))
    ax_mfcc.set_yticklabels(('-150', '', '', '0', '', '', '', '200'))
    
    sb.despine(ax=ax_zcr, right=False, left=True)
    sb.boxplot(aggregated_zcr, order=aggregated_zcr.columns, ax=ax_zcr)
    ax_zcr.set_ylim(0.0, 0.5)
    ax_zcr.set_yticks((0.0, 0.25, 0.5))
    ax_zcr.set_yticklabels(('0.0', '', '0.5'))

categories = 5
    
f, axes = plt.subplots(int(np.ceil(categories / 3.0)), 3, figsize=(16, categories * 1))
f.subplots_adjust(hspace=0.8, wspace=0.4)

map(lambda ax: ax.get_xaxis().set_visible(False), axes.flat)
map(lambda ax: ax.get_yaxis().set_visible(False), axes.flat)
map(lambda ax: ax.set_frame_on(False), axes.flat)

for c in range(0, categories):
    plot_all_features_aggregate(clips_50[c], axes.flat[c])


# Classification using random forest with 500 estimators
# Features MFCCs & ZCR at clip level
# In[70]:


def create_set(clips):
    cases = pd.DataFrame()

    for c in range(0, len(clips)):
        for i in range(0, len(clips[c])):
            case = pd.DataFrame([clips[c][i].filename], columns=['filename'])
            case['category'] = c
            case['category_name'] = clips[c][i].category
            case['fold'] = clips[c][i].filename[0]
            
            mfcc_mean = pd.DataFrame(np.mean(clips[c][i].mfcc[:, :], axis=0)[1:]).T
            mfcc_mean.columns = list('MFCC_{} mean'.format(i) for i in range(np.shape(clips[c][i].mfcc)[1]))[1:]
            mfcc_std = pd.DataFrame(np.std(clips[c][i].mfcc[:, :], axis=0)[1:]).T
            mfcc_std.columns = list('MFCC_{} std dev'.format(i) for i in range(np.shape(clips[c][i].mfcc)[1]))[1:]
            case = case.join(mfcc_mean)
            case = case.join(mfcc_std)
            
            case['ZCR mean'] = np.mean(clips[c][i].zcr)
            case['ZCR std dev'] = np.std(clips[c][i].zcr)

            cases = cases.append(case)
    
    cases[['category', 'fold']] = cases[['category', 'fold']].astype(int)
    return cases

cases_50 = create_set(clips_50)


# In[125]:

import sklearn as sk
import sklearn.ensemble

features_start = 'MFCC_1 mean'
features_end = 'ZCR std dev'

def to_percentage(number):
    return int(number * 1000) / 10.0

def classify(cases, classifier='rf', PCA=False, debug=False):
    results = []
    class_count = len(cases['category'].unique())
    #print(class_count)
    confusion = np.zeros((class_count, class_count), dtype=int)
    
    for fold in range(1, 6):
        train = cases[cases['fold'] != fold].copy()
        test = cases[cases['fold'] == fold].copy()
        classifier_name = ''

        if PCA:
            pca = sk.decomposition.PCA()
            pca.fit(train.loc[:, features_start:features_end])
            train.loc[:, features_start:features_end] = pca.transform(train.loc[:, features_start:features_end])
            test.loc[:, features_start:features_end] = pca.transform(test.loc[:, features_start:features_end])

        if classifier == 'rf':
            classifier_name = 'Random Forest'
            rf = sk.ensemble.RandomForestClassifier(n_estimators=500, random_state=20150420)
            rf.fit(train.loc[:, features_start:features_end], train['category'])
            test.loc[:, 'prediction'] = rf.predict(test.loc[:, features_start:features_end])
            
        accuracy = np.sum(test['category'] == test['prediction']) / float(len(test['category']))
        results.append(accuracy)
        confusion_current = sk.metrics.confusion_matrix(test['category'], test['prediction'])
        confusion = confusion + confusion_current
        
        print ('Classifying fold {0} with {1} classifier. Accuracy: {2}%'.format(fold, classifier_name, to_percentage(accuracy)))
        if debug:
            print ('Confusion matrix:\n', confusion_current, '\n')
        
    print ('Average accuracy: {0}%\n'.format(to_percentage(np.mean(results))))
    return confusion, results

def pretty_confusion(confusion_matrix, cases, mode='recall', css_classes=['diagonal', 'cell_right'], raw=False):
    if mode == 'recall':
        confusion_matrix = confusion_matrix * 1000 / np.sum(confusion_matrix, axis=1) / 10.0
        confusion_matrix = np.vectorize(lambda x: '{0}%'.format(x))(confusion_matrix)

    show_headers = False if 'draggable' in css_classes else True
        
    categories = sorted(cases['category'].unique())
    print(categories)
    labels = map(lambda c: cases[cases['category'] == c]['category_name'][0:1][0][6:], categories)
    confusion_matrix = pd.DataFrame(confusion_matrix, index=labels)
    
    if raw:
        return confusion_matrix    
    else:
        return IPython.display.HTML(confusion_matrix.to_html(classes=css_classes, header=show_headers))


confusion_10_rf, accuracy_10_rf = classify(cases_50, 'rf')

pretty_confusion(confusion_10_rf, cases_50, 'recall')

