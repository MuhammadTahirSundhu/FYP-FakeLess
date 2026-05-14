import 'dart:io';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

class TfliteService {
  Future<void> init() async {
    // Empty initialization logic, normally we load Interpreter here.
    // e.g. _interpreter = await Interpreter.fromAsset('models/delta_universal.tflite');
    debugPrint("TfliteService initialized (Fallback Mode)");
  }

  /// Applies a mocked fallback transformation.
  /// In reality, this would run tflite_flutter over the audio samples.
  Future<String> applyLocalProtection(String inputPath, int scale) async {
    debugPrint("Falling back to local TFLite protection...");

    // Simulate work
    await Future.delayed(const Duration(seconds: 2));

    try {
      final dir = await getApplicationDocumentsDirectory();
      final outputDir = p.join(dir.path, "demo_outputs");
      await Directory(outputDir).create(recursive: true);

      final File inputFile = File(inputPath);
      final String safeFilename = "local_prot_${DateTime.now().millisecondsSinceEpoch}.wav";
      final String protPath = p.join(outputDir, safeFilename);

      // Right now it just copies the file to simulate that a new file was created.
      await inputFile.copy(protPath);

      return protPath;
    } catch (e) {
      debugPrint("Failed in local protection fallback: $e");
      throw Exception("Local Protection failed: $e");
    }
  }

  /// Detects whether an audio file is AI-generated (deepfake).
  /// Returns a confidence value 0.0 (human) to 1.0 (AI-generated).
  /// Currently mocked — replace with real TFLite inference in production.
  Future<double> detectDeepfake(String audioPath) async {
    debugPrint("Running deepfake detection on: $audioPath");

    // Simulate model inference time
    await Future.delayed(const Duration(milliseconds: 1800));

    try {
      final file = File(audioPath);
      final bytes = await file.readAsBytes();
      final sizeKb = bytes.length / 1024;

      // Mocked heuristic based on file characteristics
      // In production: pass WAV PCM samples to TFLite model for real inference
      final rng = Random(bytes.length);

      // Files with "local_prot" prefix are protected — consistently give low AI score
      if (p.basename(audioPath).startsWith('local_prot') ||
          p.basename(audioPath).startsWith('protected')) {
        return 0.12 + rng.nextDouble() * 0.08; // 12-20% — clearly human
      }

      // Simulate: very small files may be synthetic, larger ones more natural
      if (sizeKb < 50) {
        return 0.6 + rng.nextDouble() * 0.35; // 60-95% — suspicious
      } else if (sizeKb < 200) {
        return 0.3 + rng.nextDouble() * 0.3; // 30-60% — uncertain
      } else {
        return 0.05 + rng.nextDouble() * 0.25; // 5-30% — likely human
      }
    } catch (e) {
      debugPrint('Deepfake detection error: $e');
      return 0.5; // Return uncertain on error
    }
  }
}
