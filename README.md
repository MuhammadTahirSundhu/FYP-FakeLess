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


## ⚙️ Optional: Manual Android SDK Setup (Without Android Studio)

If you’re not using **Android Studio** and instead want to install and configure the Android SDK manually (for use with **Flutter** and **VS Code**), follow these steps.

---

### 🪜 1. Download the Command-Line SDK Tools

Download the latest Android command-line tools from the official Android developer site:  
👉 [https://developer.android.com/studio#command-tools](https://developer.android.com/studio#command-tools)

Extract the contents to a convenient directory, for example:
```
C:\Android\sdk
```

---

### 🪜 2. Set Environment Variables

#### Add a new system variable:
| Variable | Value |
|-----------|--------|
| `ANDROID_HOME` | `C:\Android\sdk` |

#### Then edit your system **Path** variable and add the following entries:
```
%ANDROID_HOME%\cmdline-tools\latest\bin
%ANDROID_HOME%\platform-tools
%ANDROID_HOME%\tools\bin
```

> 💡 *If `tools\bin` doesn’t exist (only available in older SDK versions), you can skip it.*

---

### 🪜 3. Verify Installation

Open a **new Command Prompt** (important) and run:
```bash
where sdkmanager
```

You should see a path like:
```
C:\Android\sdk\cmdline-tools\latest\bin\sdkmanager.bat
```

Then check:
```bash
sdkmanager --list
```

If it lists available packages, your SDK tools are working correctly.

---

### 🪜 4. Accept Android Licenses and Check Flutter Setup

Run the following commands to make Flutter recognize your Android SDK:
```bash
flutter doctor --android-licenses
flutter doctor
```

Make sure all entries show ✅ (especially **Android toolchain**).

---

### 🪜 5. Connect Your Android Device

1. Enable **Developer Options** and **USB Debugging** on your phone.  
2. Connect it via USB.  
3. Run:
   ```bash
   flutter devices
   ```
   If your phone appears in the list, you’re ready to build and run the app.

---

### 🧩 Troubleshooting

- **`sdkmanager not recognized` in VS Code but works in CMD?**  
  Restart VS Code or your computer — the PATH changes sometimes don’t apply until you do.
- **`No connected devices found`?**  
  Ensure USB drivers are installed and debugging is enabled on your device.
- **Using PowerShell in VS Code?**  
  Try switching the terminal shell to CMD (`Ctrl + Shift + P` → *Select Default Profile* → *Command Prompt*).

---

✅ *That’s it! You can now run your Flutter app on a connected Android device without installing Android Studio.*
