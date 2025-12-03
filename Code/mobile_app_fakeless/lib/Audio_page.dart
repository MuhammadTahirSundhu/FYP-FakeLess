import 'dart:io';
// import 'dart:math';
import 'dart:typed_data';

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

final key = encrypt.Key.fromUtf8('0123456789abcdefghijklmnopqrstuv'); // 32 chars for AES-256
final iv = encrypt.IV.fromLength(16); // 16 bytes IV for AES

Uint8List encryptFile(Uint8List fileBytes) {
  final encrypter = encrypt.Encrypter(encrypt.AES(key));
  final encrypted = encrypter.encryptBytes(fileBytes, iv: iv);
  return encrypted.bytes;
}

Uint8List decryptFile(Uint8List encryptedBytes) {
  final encrypter = encrypt.Encrypter(encrypt.AES(key));
  final decrypted = encrypter.decryptBytes(encrypt.Encrypted(encryptedBytes), iv: iv);
  return Uint8List.fromList(decrypted);
}

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
                ? _playButton()
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
                              ? _playProtectedButton()
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

  @override
  void initState() {
    super.initState();

    // Listen for when the audio finishes playing
    _audioPlayer.playerStateStream.listen((state) {
      if (state.processingState == ProcessingState.completed) {
        if (!mounted) return;
        setState(() {
          isPlaying = false;
        });
      }
    });
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

        final uri = Uri.parse('https://aldo-cushiony-drily.ngrok-free.dev/protect'); // Replace with your server IP
        final request = http.MultipartRequest('POST', uri);
        request.files.add(await http.MultipartFile.fromPath('audio', recordingPath!));

        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Uploading to cloud for protection...")),
        );

        try {
          final streamedResponse = await request.send();
          final statusCode = streamedResponse.statusCode;
          final responseBytes = await streamedResponse.stream.toBytes();
          if (statusCode == 200) {
            // Save the received bytes as a .wav file
            final dir = await getApplicationDocumentsDirectory();
            final protPath = p.join(dir.path, "protected.wav");
            final file = File(protPath);
            await file.writeAsBytes(responseBytes);
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
              SnackBar(content: Text("Error uplaoding file")),
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
    if (isPlaying) {
      return FloatingActionButton(
        onPressed: () async {
          await _audioPlayer.stop();
          if (!mounted) return;
          setState(() {
            isPlaying = false;
          });
        },
        heroTag: 'stopButton',
        child: const Icon(Icons.stop),
      );
    } else {
      return FloatingActionButton(
        onPressed: () async {
          if (recordingPath != null) {
            await _audioPlayer.setFilePath(recordingPath!);
            _audioPlayer.play();
            if (!mounted) return;
            setState(() {
              isPlaying = true;
            });
          }
        },
        heroTag: 'playButton',
        child: const Icon(Icons.play_arrow),
      );
    }
  }

  Widget _playProtectedButton() {
    if (isPlaying) {
      return FloatingActionButton(
        onPressed: () async {
          await _audioPlayer.stop();
          if (!mounted) return;
          setState(() {
            isPlaying = false;
          });
        },
        heroTag: 'stopProtectedButton',
        child: const Icon(Icons.stop),
      );
    } else {
      return FloatingActionButton(
        onPressed: () async {
          if (protectedPath != null) {
            await _audioPlayer.setFilePath(protectedPath!);
            _audioPlayer.play();
            if (!mounted) return;
            setState(() {
              isPlaying = true;
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
            });
          }
        }
      },
      heroTag: 'recordButton',
      child: Icon(isRecording ? Icons.stop : Icons.mic),
    );
  }
}
