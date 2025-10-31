import 'dart:io';
// import 'dart:math';
// import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:just_audio/just_audio.dart';
// import 'package:mobile_app_fakeless/utils/delta_loader.dart';
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
// import 'package:wav/wav.dart';
import 'package:mobile_app_fakeless/utils/wav_helper.dart';
import 'package:serious_python/serious_python.dart';
// import 'package:http/http.dart' as http;
// import 'package:flutter/services.dart';




class AudioPage extends StatefulWidget {
  const AudioPage({super.key});

  @override
  AudioPageState createState() => AudioPageState();
}

class AudioPageState extends State<AudioPage> {
  final AudioRecorder _audioRecorder = AudioRecorder();
  final AudioPlayer _audioPlayer = AudioPlayer();
  bool isRecording = false;
  bool isPlaying = false;
  String? recordingPath;
  String? protectedPath;
  
  @override
  void initState() {
    super.initState();

    // Listen for when the audio finishes playing
    _audioPlayer.playerStateStream.listen((state) {
      if (state.processingState == ProcessingState.completed) {
        if(!mounted) return;
        setState(() {
          isPlaying = false;
        });
      }
    });
  }


  Widget _recordButton(){
    return FloatingActionButton(onPressed: () async {
        if(isRecording){
          String? filePath = await _audioRecorder.stop();
          if (filePath != null){
          if(!mounted) return;
            setState(() {
              isRecording = false;
              recordingPath = filePath;
              });
        }
        }else{
          if (await _audioRecorder.hasPermission()) {
            final Directory appDocDir = await getApplicationDocumentsDirectory();
            final String filePath = p.join(appDocDir.path, 'audio.wav');//${DateTime.now().millisecondsSinceEpoch}
            await _audioRecorder.start(
              RecordConfig(encoder: AudioEncoder.wav,
                bitRate: 16000,
                sampleRate: 16000,), 
              path: filePath);
              if(!mounted) return;
              setState(() {
                isRecording = true;
                recordingPath = null;
                });
          }
        }
      },
      heroTag: 'recordButton', 
      child: Icon(isRecording ? Icons.stop : Icons.mic)
    );
  }

  Widget _playButton(){
    if(isPlaying){
      return FloatingActionButton(onPressed: () async {
        await _audioPlayer.stop();
        if(!mounted) return;
        setState(() {
          isPlaying = false;
        });
      }, 
      heroTag: 'stopButton',
      child: const Icon(Icons.stop)
      );
    }
    else{
      return FloatingActionButton(onPressed: () async {
        if (recordingPath != null){
          await _audioPlayer.setFilePath(recordingPath!);
          _audioPlayer.play();
          if(!mounted) return;
          setState(() {
            isPlaying = true;
          });
        }
      }, 
      heroTag: 'playButton',
      child: const Icon(Icons.play_arrow)
      );
    }
  }

  Widget _applyButton() {
    return FloatingActionButton(
      onPressed: () async {
        if (recordingPath == null) return;

        ScaffoldMessenger.of(context)
            .showSnackBar(const SnackBar(content: Text("Applying perturbation...")));

        final protectedPath = await applyDeltaWithPython(recordingPath!);

        if (protectedPath != null && protectedPath.contains('.wav')) {
          setState(() {
            this.protectedPath = protectedPath;
          });
          ScaffoldMessenger.of(context)
              .showSnackBar(const SnackBar(content: Text("Protected audio ready!")));
        } else {
          ScaffoldMessenger.of(context)
              .showSnackBar(SnackBar(content: Text("Error: $protectedPath")));
        }
      },
      heroTag: 'applyButton',
      child: const Icon(Icons.cyclone_rounded),
    );
  }

  Future<String?> applyDeltaWithPython(String inputFilePath) async {
    try {
      final pythonZipAsset = "assets/python/Code.zip";

      // where Python will write outputs
      final dir = await getApplicationDocumentsDirectory();
      final outputDir = p.join(dir.path, "demo_outputs");
      await Directory(outputDir).create(recursive: true);

      // Pass values via environment variables (serious_python supports environmentVariables)
      final env = <String, String>{
        "INPUT_PATH": inputFilePath,
        "DELTA_PATH": "checkpoints_phase1/universal_delta_epoch25.npy",
        "OUT_DIR": outputDir,
      };

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Starting Python script..."))
      );

      final stdout = await SeriousPython.run(
        pythonZipAsset,
        appFileName: "apply_defense.py",
        environmentVariables: env,
        // you can use sync: true if you want to block until the script finishes,
        // but be careful: it can block the UI thread. The default is async/background.
      );

      debugPrint("SeriousPython stdout:\n$stdout");

      final protPath = p.join(outputDir, "prot_universal.wav");
      final protFile = File(protPath);
      if (await protFile.exists()) {
        return protPath;
      } else {
        final msg = (stdout != null && stdout.isNotEmpty) ? stdout : "Protected file not found at $protPath";
        return "ERROR: $msg";
      }
    } catch (e, st) {
      debugPrint("applyDeltaWithPython error: $e\n$st");
      return "ERROR: $e";
    }
  }



  Widget _playProtectedButton(){
    if(isPlaying){
      return FloatingActionButton(onPressed: () async {
        await _audioPlayer.stop();
        if(!mounted) return;
        setState(() {
          isPlaying = false;
        });
      }, 
      heroTag: 'stopProtectedButton',
      child: const Icon(Icons.stop)
      );
    }
    else{
      return FloatingActionButton(onPressed: () async {
        if (protectedPath != null){
          await _audioPlayer.setFilePath(protectedPath!);
          _audioPlayer.play();
          if(!mounted) return;
          setState(() {
            isPlaying = true;
          });
        }
      }, 
      heroTag: 'playProtectedButton',
      child: const Icon(Icons.play_arrow)
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    final screenHeight = MediaQuery.of(context).size.height;
    return Scaffold(
      appBar: AppBar(title: const Text("Audio Page")),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          crossAxisAlignment: CrossAxisAlignment.center,
          spacing: 10,
          children: <Widget>[
            recordingPath != null ? _playButton() : Text("No recording available"),
            _recordButton(),
            recordingPath != null ? 
            SizedBox(
              height: 0.6 * screenHeight,
              width: 1 * screenWidth, 
              child: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.end,
                  crossAxisAlignment: CrossAxisAlignment.center,
                  spacing: 30,
                  children: <Widget>[
                    protectedPath != null ? _playProtectedButton() : Text("No protected audio available"),
                    _applyButton()
                  ]
                )
              )
            ) :
            SizedBox.shrink(),

          ],
        )
      ),
    );
  }

  @override
  void dispose() {
    if (isPlaying) {
      _audioPlayer.stop();
    }
    if (isRecording) {
      _audioRecorder.stop();
    }
    _audioPlayer.dispose();
    _audioRecorder.dispose();
    super.dispose();
  }

}
