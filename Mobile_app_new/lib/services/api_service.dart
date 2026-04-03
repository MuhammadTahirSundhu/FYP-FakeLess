import 'dart:io';
import 'dart:convert';
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
    switch (scale) {
      case 1: return 0.1;
      case 2: return 0.3;
      case 3: return 0.6;
      case 4: return 0.8;
      case 5: return 1.0;
      default: return 0.3;
    }
  }

  String get endpoint =>
      dotenv.env['PROTECTION_API_ENDPOINT'] ?? 'https://fakeless-api.onrender.com/protect';

  Future<Map<String, dynamic>> protectAudio(String recordingPath, int scale) async {
    final connectivityResult = await Connectivity().checkConnectivity();
    if (connectivityResult.contains(ConnectivityResult.none)) {
      throw OfflineException("No active internet connection.");
    }

    try {
      final filterStrength = _scaleToFilterStrength(scale);
      debugPrint("ApiService: Sending to $endpoint with filter_strength=$filterStrength");

      final uri = Uri.parse(endpoint);
      final request = http.MultipartRequest('POST', uri);
      request.files.add(
        await http.MultipartFile.fromPath('file', recordingPath, filename: p.basename(recordingPath)),
      );
      request.fields['filter_strength'] = filterStrength.toStringAsFixed(2);

      final streamedResponse = await request.send().timeout(
        const Duration(seconds: 120),
        onTimeout: () => throw ApiException("Server timed out. Please try again."),
      );
      
      // Parse custom scores out of the response headers returned by the server
      double? pesqScore;
      double? stoiScore;
      if (streamedResponse.headers.containsKey('x-pesq-score')) {
        pesqScore = double.tryParse(streamedResponse.headers['x-pesq-score'] ?? '');
      }
      if (streamedResponse.headers.containsKey('x-stoi-score')) {
        stoiScore = double.tryParse(streamedResponse.headers['x-stoi-score'] ?? '');
      }

      final responseBytes = await streamedResponse.stream.toBytes();

      if (streamedResponse.statusCode == 200) {
        late Uint8List audioBytesToSave;

        // The backend developer ("Sameed") switched the endpoint to return JSON with base64 audio and metrics
        // So we gracefully try to decode as JSON first.
        try {
          final decodedText = utf8.decode(responseBytes);
          final jsonMap = jsonDecode(decodedText);
          
          debugPrint("ApiService JSON received keys: ${jsonMap.keys.toString()}");
          
          final base64String = jsonMap['audio'] ?? jsonMap['protected_audio'] ?? jsonMap['audio_base64'] ?? jsonMap['audio_data'] ?? jsonMap['file_data'];
          if (base64String != null) {
            audioBytesToSave = base64Decode(base64String);
          } else {
             audioBytesToSave = responseBytes;
          }

          // Also pull metrics if they're in the JSON body
          if (jsonMap.containsKey('pesq_score')) pesqScore = (jsonMap['pesq_score'] as num).toDouble();
          else if (jsonMap.containsKey('pesq')) pesqScore = (jsonMap['pesq'] as num).toDouble();

          if (jsonMap.containsKey('stoi_score')) stoiScore = (jsonMap['stoi_score'] as num).toDouble();
          else if (jsonMap.containsKey('stoi')) stoiScore = (jsonMap['stoi'] as num).toDouble();
        } catch (e) {
          // If it fails to parse as JSON, assume it's the raw WAV binary
          audioBytesToSave = responseBytes;
        }

        final dir = await getApplicationDocumentsDirectory();
        final filename = "protected_${DateTime.now().millisecondsSinceEpoch}.wav";
        final protPath = p.join(dir.path, filename);
        await File(protPath).writeAsBytes(audioBytesToSave);

        debugPrint("ApiService: Protected audio saved to $protPath. PESQ=$pesqScore, STOI=$stoiScore");

        return {
          'path': protPath,
          'pesq': pesqScore, 
          'stoi': stoiScore,
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

  String get cloneEndpoint =>
      dotenv.env['ATTACKER_API_ENDPOINT'] ?? 'https://fakeless-api.onrender.com/clone';

  Future<String> cloneVoice(String recordingPath, String textParam) async {
    final connectivityResult = await Connectivity().checkConnectivity();
    if (connectivityResult.contains(ConnectivityResult.none)) {
      throw OfflineException("No active internet connection.");
    }

    try {
      debugPrint("ApiService: Sending clone request to $cloneEndpoint with text='$textParam'");

      final uri = Uri.parse(cloneEndpoint);
      final request = http.MultipartRequest('POST', uri);
      request.files.add(
        await http.MultipartFile.fromPath('speaker', recordingPath, filename: p.basename(recordingPath)),
      );
      request.fields['text'] = textParam;

      final streamedResponse = await request.send().timeout(
        const Duration(seconds: 120),
        onTimeout: () => throw ApiException("Server timed out. Please try again."),
      );

      final responseBytes = await streamedResponse.stream.toBytes();

      if (streamedResponse.statusCode == 200) {
        late Uint8List audioBytesToSave;

        try {
          final decodedText = utf8.decode(responseBytes);
          final jsonMap = jsonDecode(decodedText);
          
          final base64String = jsonMap['audio'] ?? jsonMap['cloned_audio'] ?? jsonMap['audio_base64'] ?? jsonMap['audio_data'] ?? jsonMap['file_data'];
          if (base64String != null) {
            audioBytesToSave = base64Decode(base64String);
          } else {
             audioBytesToSave = responseBytes;
          }
        } catch (e) {
          audioBytesToSave = responseBytes;
        }

        final dir = await getApplicationDocumentsDirectory();
        final filename = "cloned_${DateTime.now().millisecondsSinceEpoch}.wav";
        final clonePath = p.join(dir.path, filename);
        await File(clonePath).writeAsBytes(audioBytesToSave);

        debugPrint("ApiService: Cloned audio saved to $clonePath");
        return clonePath;
      } else {
        String errorMsg = "Clone Server error (${streamedResponse.statusCode})";
        try {
          final errorStr = String.fromCharCodes(responseBytes);
          if (errorStr.isNotEmpty && errorStr.length < 500) errorMsg = errorStr;
        } catch (_) {}
        throw ApiException(errorMsg, streamedResponse.statusCode);
      }
    } on SocketException catch (e) {
      throw OfflineException("Failed to reach attacker API: ${e.message}");
    }
  }
}
