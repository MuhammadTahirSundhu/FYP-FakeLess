** FakeLess - Audio Deepfake Prevention

Step 1 
Clone into your system using
```
git clone https://www.github.com/MuhammadTahirSundhu/FYP-Fakeless.git
```
or use Github CLI (you would have to install it beforehand)

``` 
gh repo clone MuhammadTahirSundhu/FYP-FakeLess
```
Step 2
Move to Code folder and run prepare_librispeech.py
``` 
cd Code
python prepare_librispeech.py --root /path/to/datasetsdownloads --output ../data/librispeech_prepared
```
If run correctly you will have successfully decompressed flac audio to wav files
But you have flac files remaining which you have to remove

Go to main/data/librispeech_prepared

Check for the folder LibriSpeech and delete it


Step 3 

Before running defense_training.py 
you will need to install the required modules 
torch,torchaudio,speechbrain and some others

locate where speechbrain is installed this may vary according to your system/install directory

part 1

\path\to\python\Lib\site-packages\speechbrain\lobes\models\ECAPA_TDNN.py

find (around line 480)
```
x = layer(x, lengths=lengths)
```
and replace with 
```
try:
  x = layer(x, lengths=lengths)
except TypeError:
  x = layer(x)
```

** DISCLAIMER dont edit if try expect block is already there




part 2

\path\to\python\Lib\site-packages\speechbrain\nnet\CNN.py

find (around line 476)
```
x = F.pad(x, padding, mode=self.padding_mode)
```

replace with 
```
x = F.pad(x, padding, mode='constant', value=0)
```
** DISCLAIMER dont edit if syntax is same


run defense_training.py

```
python defense_training.py train_universal
```

