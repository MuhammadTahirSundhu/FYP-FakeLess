import 'dart:io';
// import 'dart:math';
import 'dart:typed_data';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:just_audio/just_audio.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:http/http.dart' as http;
import 'package:encrypt/encrypt.dart' as encrypt;
// import 'package:mobile_app_fakeless/utils/delta_loader.dart';
// import 'package:serious_python/serious_python.dart';
// import 'package:wav/wav.dart';
// wav_helper currently not used by on-device flow; keep helpers in this file instead.
// import 'package:tflite_flutter/tflite_flutter.dart';
// import 'package:flutter/services.dart';

final _plainKeyString = '0123456789abcdefghijklmnopqrstuv'; // 32 bytes

// Derive the base64 (urlsafe) key string to match Python's
final _b64KeyString = base64Url.encode(utf8.encode(_plainKeyString));
final _fernetKey = encrypt.Key.fromBase64(_b64KeyString);
final _fernet = encrypt.Fernet(_fernetKey);
final encrypter = encrypt.Encrypter(_fernet);

// Helpers that operate on base64 tokens for transport (ASCII-safe):
String encryptFileToBase64(Uint8List fileBytes) {
  final encrypted = encrypter.encryptBytes(fileBytes);
  return encrypted.base64; // token as base64 string
}

Uint8List decryptBase64ToBytes(String base64Token) {
  final encrypted = encrypt.Encrypted.fromBase64(base64Token);
  final decrypted = encrypter.decryptBytes(encrypted);
  return Uint8List.fromList(decrypted);
}

class AudioPage extends StatefulWidget {
  const AudioPage({super.key});

  @override
  AudioPageState createState() => AudioPageState();
}

class AudioPageState extends State<AudioPage> {
  final AudioRecorder _audioRecorder = AudioRecorder();
  final AudioPlayer _audioPlayerreco = AudioPlayer();
  final AudioPlayer _audioPlayerproc = AudioPlayer();
  bool isRecording = false;
  // bool isPlayingOriginal = false;
  // bool isPlayingProtected = false;
  String? recordingPath;
  String? protectedPath;
  bool audioPlayerrecoPath = false;
  bool audioPlayerprocPath = false;

  Duration positionreco = Duration.zero;
  Duration durationreco = Duration.zero;
  Duration positionproc = Duration.zero;
  Duration durationproc = Duration.zero;

  // Future<String?> applyDeltaWithPython(String inputFilePath) async {
  //   try {
  //     final pythonZipAsset = "assets/python/Code.zip";

  //     // where Python will write outputs
  //     final dir = await getApplicationDocumentsDirectory();
  //     final outputDir = p.join(dir.path, "demo_outputs");
  //     await Directory(outputDir).create(recursive: true);

  //     // Pass values via environment variables (serious_python supports environmentVariables)
  //     final env = <String, String>{
  //       "INPUT_PATH": inputFilePath,
  //       "DELTA_PATH": "checkpoints_phase1/universal_delta_epoch25.npy",
  //       "OUT_DIR": outputDir,
  //     };

  //     ScaffoldMessenger.of(context).showSnackBar(
  //       const SnackBar(content: Text("Starting Python script...")),
  //     );

  //     final stdout = await SeriousPython.run(
  //       pythonZipAsset,
  //       appFileName: "apply_defense.py",
  //       environmentVariables: env,
  //       // you can use sync: true if you want to block until the script finishes,
  //       // but be careful: it can block the UI thread. The default is async/background.
  //     );

  //     debugPrint("SeriousPython stdout:\n$stdout");

  //     final protPath = p.join(outputDir, "prot_universal.wav");
  //     final protFile = File(protPath);
  //     if (await protFile.exists()) {
  //       return protPath;
  //     } else {
  //       final msg = (stdout != null && stdout.isNotEmpty)
  //           ? stdout
  //           : "Protected file not found at $protPath";
  //       return "ERROR: $msg";
  //     }
  //   } catch (e, st) {
  //     debugPrint("applyDeltaWithPython error: $e\n$st");
  //     return "ERROR: $e";
  //   }
  // }

  /// Apply the universal delta using the on-device TFLite model.
  ///
  /// This is a best-effort scaffold that:
  ///  - Loads the TFLite interpreter from app assets (`assets/models/delta_universal.tflite`).
  ///  - Reads the WAV file (assumes PCM16 mono) into a normalized sample list.
  ///  - (TODO) Computes the log-mel spectrogram from the waveform.
  ///  - Runs the model to obtain a mel-domain delta.
  ///  - (TODO) Reconstructs waveform from the perturbed mel (e.g., Griffin-Lim or vocoder).
  ///
  /// Right now this function runs the interpreter with a placeholder (zeros)
  /// input so you can validate the interpreter load on-device. Replace the
  /// mel-preprocessing and reconstruction TODOs with actual implementations.
  // Future<String?> applyDeltaWithTFLite(String inputFilePath) async {
  //   try {
  //     // 1) Load interpreter from assets (ensure model is listed in pubspec.yaml)
  //     final interpreter = await Interpreter.fromAsset(
  //       'models/delta_universal.tflite',
  //     );

  //     // 2) Inspect model input shape to create a compatible placeholder input
  //     final inputTensors = interpreter.getInputTensors();
  //     if (inputTensors.isEmpty) {
  //       return 'ERROR: model has no inputs';
  //     }
  //     final shape = inputTensors.first.shape; // e.g. [1, n_mels, T]
  //     final batch = shape[0];
  //     final nMels = shape[1];
  //     final tFrames = shape[2];

  //     // 3) Read WAV file (PCM16 mono) into List<double> normalized samples
  //     // Note: mel-spectrogram computation is not implemented here yet.
  //     final _samples = await _readWavPcm16(inputFilePath);

  //     // TODO: compute log-mel spectrogram from `samples` (shape [n_mels, T_var]).
  //     // For now we create a zeros input matching the model shape so the
  //     // interpreter can be exercised on-device.
  //     final input = List.generate(
  //       batch,
  //       (_) => List.generate(nMels, (_) => List.filled(tFrames, 0.0)),
  //     );

  //     // 4) Prepare output buffer with same shape
  //     final output = List.generate(
  //       batch,
  //       (_) => List.generate(nMels, (_) => List.filled(tFrames, 0.0)),
  //     );

  //     // 5) Run interpreter (synchronous)
  //     interpreter.run(input, output);

  //     // 6) TODO: apply `output` delta to mel spectrogram, reconstruct waveform.
  //     // For now, just write a copy of the input file to demo_outputs to signal success.
  //     final dir = await getApplicationDocumentsDirectory();
  //     final outputDir = p.join(dir.path, "demo_outputs");
  //     await Directory(outputDir).create(recursive: true);
  //     final protPath = p.join(outputDir, "prot_universal.wav");
  //     await File(inputFilePath).copy(protPath);

  //     return protPath;
  //   } catch (e, st) {
  //     debugPrint('applyDeltaWithTFLite error: $e\n$st');
  //     return 'ERROR: $e';
  //   }
  // }

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
            recordingPath != null
                ? SizedBox(
                 child: Center(
                   child: Row(
                    children: <Widget>[
                      _playButton(),
                      Slider(
                        min: 0.0,
                        max: durationreco.inSeconds.toDouble(),
                        value: positionreco.inSeconds.toDouble(),
                        onChanged: (double value){
                          _audioPlayerreco.seek(Duration(seconds: value.toInt()));
                        },
                      ),
                      Text(formatDuration(positionreco) + "/" + formatDuration(durationreco)),
                    ]
                   )
                 )
                )
                : Text("No recording available"),
            _recordButton(),
            recordingPath != null
                ? SizedBox(
                    height: 0.6 * screenHeight,
                    width: 1 * screenWidth,
                    child: Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.end,
                        crossAxisAlignment: CrossAxisAlignment.center,
                        spacing: 30,
                        children: <Widget>[
                          protectedPath != null
                              ? SizedBox( 
                                child: Row(
                                  children: <Widget>[
                                  _playProtectedButton(),
                                  Slider(
                                    min: 0.0,
                                    max: durationproc.inSeconds.toDouble(),
                                    value: positionproc.inSeconds.toDouble(),
                                    onChanged: ( double value){
                                      _audioPlayerproc.seek(Duration(seconds: value.toInt()));
                                    },
                                    ),
                                  Text(formatDuration(positionproc) + "/" + formatDuration(durationproc)),
                                ],
                                )
                              )
                              : Text("No protected audio available"),
                          _applyButton(),
                          _applyButtonCloud(),
                        ],
                      ),
                    ),
                  )
                : SizedBox.shrink(),
          ],
        ),
      ),
    );
  }

  // @override
  // void dispose() {
  //   if (isPlayingOriginal) {
  //     _audioPlayer.stop();
  //   }
  //   if (isRecording) {
  //     _audioRecorder.stop();
  //   }
  //   _audioPlayer.dispose();
  //   _audioRecorder.dispose();
  //   super.dispose();
  // }

  

  String formatDuration(Duration d){
    String twoDigits(int n) => n.toString().padLeft(2, '0');
    final minutes = twoDigits(d.inMinutes.remainder(60));
    final seconds = twoDigits(d.inSeconds.remainder(60));
    return '$minutes:$seconds';
  }


  @override
  void initState() {
    super.initState();

    //functionality for original audio player
    _audioPlayerreco.playerStateStream.listen((state) {
      if (state.processingState == ProcessingState.completed) {
        if (!mounted) return;
        setState(() {
          positionreco = Duration.zero;
        });
        _audioPlayerreco.stop();
        _audioPlayerreco.seek(positionreco);
      }
    });

    _audioPlayerreco.positionStream.listen((pos) {
      if (!mounted) return;
      setState(() {
        positionreco = pos;
      });
    });

    _audioPlayerreco.durationStream.listen((dur) {
      if (!mounted) return;
      setState(() {
        durationreco = dur!;
      });
    });

    //functionality for protected audio player
    _audioPlayerproc.playerStateStream.listen((state) {
      if (state.processingState == ProcessingState.completed) {
        if (!mounted) return;
        setState(() {
          positionproc = Duration.zero;
        });
        _audioPlayerproc.stop();
        _audioPlayerproc.seek(positionproc);
      }
    });

    _audioPlayerproc.positionStream.listen((pos) {
      if (!mounted) return;
      setState(() {
        positionproc = pos;
      });
    });

    _audioPlayerproc.durationStream.listen((dur) {
      if (!mounted) return;
      setState(() {
        durationproc = dur!;
      });
    });
  }

  void handleSeek(double value, AudioPlayer audioPlayer){
    audioPlayer.seek(Duration(seconds: value.toInt()));
  }

  Widget _applyButton() {
    return FloatingActionButton(
      onPressed: () async {
        // NOTE: previous implementation used SeriousPython to call a Python
        // script that applied the universal delta. That integration is
        // commented out below. Replace this placeholder with an on-device
        // TFLite invocation (or other platform-specific implementation).

        if (recordingPath == null) return;

        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text("Applying perturbation locally (TODO)..."),
          ),
        );

        // --- previous SeriousPython implementation (commented out) ---
        /*
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
        */

        // TODO: implement applyDeltaWithTFLite(recordingPath) and set
        // `this.protectedPath` once the local model inference is implemented.
      },
      heroTag: 'applyButton',
      child: const Icon(Icons.cyclone_rounded),
    );
  }

  Widget _applyButtonCloud() {
    return FloatingActionButton(
      onPressed: () async {
        if (recordingPath == null) return;

        final bytes = File(recordingPath!).readAsBytesSync();
        final encBase64 = encryptFileToBase64(bytes);
        final dec = decryptBase64ToBytes(encBase64);

        debugPrint(bytes.length.toString());
        debugPrint('Cloud encrypt/decrypt test:');
        debugPrint(dec.length.toString());

        final uri = Uri.parse('https://aldo-cushiony-drily.ngrok-free.dev/protect'); // Replace with your server IP
        final request = http.MultipartRequest('POST', uri);

        // Encrypt the audio file before upload and write base64 token to file
        final fileBytes = await File(recordingPath!).readAsBytes();
        final encryptedBase64 = encryptFileToBase64(fileBytes);
        final tempDir = await getTemporaryDirectory();
        final encryptedPath = p.join(tempDir.path, 'encrypted_audio.b64');
        await File(encryptedPath).writeAsString(encryptedBase64);
        request.files.add(await http.MultipartFile.fromPath('audio', encryptedPath));

        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Uploading encrypted audio to cloud for protection...")),
        );

        try {
          final streamedResponse = await request.send();
          final statusCode = streamedResponse.statusCode;
          final responseBytes = await streamedResponse.stream.toBytes();
          if (statusCode == 200) {
            // Response contains a base64 Fernet token (ASCII) — decode and decrypt
            final responseBase64 = utf8.decode(responseBytes);
            final decryptedBytes = decryptBase64ToBytes(responseBase64);
            final dir = await getApplicationDocumentsDirectory();
            final protPath = p.join(dir.path, "protected.wav");
            final file = File(protPath);
            await file.writeAsBytes(decryptedBytes);
            setState(() {
              protectedPath = protPath;
            });
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text("Protected audio ready!")),
            );
          } else {
            // Try to decode error message if present
            String errorMsg = "Error: $statusCode";
            try {
              final errorStr = String.fromCharCodes(responseBytes);
              if (errorStr.isNotEmpty) errorMsg = errorStr;
            } catch (_) {}
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text("Error uploading file: $errorMsg")),
            );
          }
        } catch (e) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text("Error: $e")),
          );
        }
      },
      heroTag: 'applyButtonCloud',
      child: const Icon(Icons.cloud_circle),
    );
  }

  Widget _playButton() {
    if (_audioPlayerreco.playing) {
      return FloatingActionButton(
        onPressed: () async {
          await _audioPlayerreco.pause();
          if (!mounted) return; // ensure widget is loaded
        },
        heroTag: 'stopButton',
        child: const Icon(Icons.stop),
      );
    } else {
      return FloatingActionButton(
        onPressed: () async {
          if (recordingPath != null) {
            if (audioPlayerrecoPath != true){
              await _audioPlayerreco.setFilePath(recordingPath!);
            }
            _audioPlayerreco.play();
            if (!mounted) return; // ensure widget is loaded
            setState(() {
              audioPlayerrecoPath = true;
            });
          }
        },
        heroTag: 'playButton',
        child: const Icon(Icons.play_arrow),
      );
    }
  }

  Widget _playProtectedButton() {
    if (_audioPlayerproc.playing) {
      return FloatingActionButton(
        onPressed: () async {
          await _audioPlayerproc.pause();
          if (!mounted) return; // ensure widget is loaded
        },
        heroTag: 'stopProtectedButton',
        child: const Icon(Icons.stop),
      );
    } else {
      return FloatingActionButton(
        onPressed: () async {
          if (protectedPath != null) {
            if (audioPlayerprocPath != true){
              await _audioPlayerproc.setFilePath(protectedPath!);
            }
            _audioPlayerproc.play();
            if (!mounted) return; // ensure widget is loaded
            setState(() {
              audioPlayerprocPath = true;
            });
          }
        },
        heroTag: 'playProtectedButton',
        child: const Icon(Icons.play_arrow),
      );
    }
  }

  // Helper: read PCM16 mono WAV samples and return normalized [-1..1] doubles.
  Future<List<double>> _readWavPcm16(String path) async {
    final bytes = await File(path).readAsBytes();
    // minimal WAV parsing: find 'data' chunk
    int idx = 12; // skip RIFF header
    while (idx + 8 < bytes.length) {
      final chunkId = String.fromCharCodes(bytes.sublist(idx, idx + 4));
      final chunkSize = bytes.buffer.asByteData().getUint32(
        idx + 4,
        Endian.little,
      );
      if (chunkId == 'data') {
        final dataStart = idx + 8;
        final dataBytes = bytes.sublist(dataStart, dataStart + chunkSize);
        final out = <double>[];
        final bd = ByteData.sublistView(Uint8List.fromList(dataBytes));
        for (int i = 0; i + 1 < bd.lengthInBytes; i += 2) {
          final s = bd.getInt16(i, Endian.little);
          out.add(s / 32768.0);
        }
        return out;
      }
      idx += 8 + chunkSize;
    }
    return <double>[];
  }

  Widget _recordButton() {
    return FloatingActionButton(
      onPressed: () async {
        if (isRecording) {
          String? filePath = await _audioRecorder.stop();
          if (filePath != null) {
            if (!mounted) return;
            setState(() {
              isRecording = false;
              recordingPath = filePath;
            });
          }
        } else {
          if (await _audioRecorder.hasPermission()) {
            final Directory appDocDir =
                await getApplicationDocumentsDirectory();
            final String filePath = p.join(
              appDocDir.path,
              'audio.wav',
            ); //${DateTime.now().millisecondsSinceEpoch}
            await _audioRecorder.start(
              RecordConfig(
                encoder: AudioEncoder.wav,
                bitRate: 16000,
                sampleRate: 16000,
              ),
              path: filePath,
            );
            if (!mounted) return;
            setState(() {
              isRecording = true;
              recordingPath = null;
              protectedPath = null;
            });
          }
        }
      },
      heroTag: 'recordButton',
      child: Icon(isRecording ? Icons.stop : Icons.mic),
    );
  }
}
