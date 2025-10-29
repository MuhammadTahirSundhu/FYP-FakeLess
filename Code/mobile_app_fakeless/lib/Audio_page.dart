import 'dart:io';

import 'package:flutter/material.dart';
import 'package:just_audio/just_audio.dart';
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

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
              RecordConfig(), 
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

  @override
  Widget build(BuildContext context) {
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
            Text(
              'This is the audio recording page.',
              style: TextStyle(fontSize: 24),
            ),
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
