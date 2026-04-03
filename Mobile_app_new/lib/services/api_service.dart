import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:flutter/foundation.dart';

class OfflineException implements Exception {
  final String message;
  OfflineException(this.message);
  @override String toString() => "OfflineException: $message";
}

class ApiException implements Exception {
  final String message;
  final int? statusCode;
  ApiException(this.message, [this.statusCode]);
  @override String toString() => "ApiException: $message, Status code: $statusCode";
}

class ApiService {
  // filter_strength: 1 = lowest protection, 0 = highest protection
  // We map our 1-5 scale to this: scale 1 → filter 1.0, scale 5 → filter 0.0
  static double _scaleToFilterStrength(int scale) {
    // scale 1→1.0, scale 2→0.75, scale 3→0.5, scale 4→0.25, scale 5→0.0
    return ((5 - scale) / 4.0);
  }

  String get endpoint =>
      dotenv.env['PROTECTION_API_ENDPOINT'] ?? 'https://fakeless-api.onrender.com/protect';

  Future<Map<String, dynamic>> protectAudio(String recordingPath, int scale) async {
    // 1. Check Connectivity
    final connectivityResult = await Connectivity().checkConnectivity();
    if (connectivityResult.contains(ConnectivityResult.none)) {
      throw OfflineException("No active internet connection.");
    }

    try {
      final filterStrength = _scaleToFilterStrength(scale);
      debugPrint("ApiService: Sending to $endpoint with filter_strength=$filterStrength");

      // 2. Build multipart request — matches server contract:
      //    -F "file=@audio.wav" -F "filter_strength=0.5"
      final uri = Uri.parse(endpoint);
      final request = http.MultipartRequest('POST', uri);
      request.files.add(
        await http.MultipartFile.fromPath('file', recordingPath,
            filename: p.basename(recordingPath)),
      );
      request.fields['filter_strength'] = filterStrength.toStringAsFixed(2);

      // 3. Send with a generous timeout (Render cold-starts can be slow)
      final streamedResponse = await request.send().timeout(
        const Duration(seconds: 120),
        onTimeout: () => throw ApiException("Server timed out. Please try again."),
      );

      final responseBytes = await streamedResponse.stream.toBytes();

      if (streamedResponse.statusCode == 200) {
        // Server returns raw WAV bytes — save directly to disk
        final dir = await getApplicationDocumentsDirectory();
        final filename = "protected_${DateTime.now().millisecondsSinceEpoch}.wav";
        final protPath = p.join(dir.path, filename);
        await File(protPath).writeAsBytes(responseBytes);

        debugPrint("ApiService: Protected audio saved to $protPath");

        return {
          'path': protPath,
          'pesq': null, // Server doesn't return metrics; handled by TFLite locally
          'stoi': null,
        };
      } else {
        String errorMsg = "Server error (${streamedResponse.statusCode})";
        try {
          final errorStr = String.fromCharCodes(responseBytes);
          if (errorStr.isNotEmpty && errorStr.length < 500) errorMsg = errorStr;
        } catch (_) {}
        throw ApiException(errorMsg, streamedResponse.statusCode);
      }
    } on SocketException catch (e) {
      throw OfflineException("Failed to reach protection server: ${e.message}");
    }
  }
}
