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



