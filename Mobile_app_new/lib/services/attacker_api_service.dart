import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'dart:io';

class AttackerApiService {
  String get endpoint => dotenv.env['ATTACKER_API_ENDPOINT'] ?? 'https://fakeless-api.onrender.com/clone';

  Future<String?> simulateAttack({required String audioPath, required String targetText}) async {
    // Check Connectivity
    final connectivityResult = await (Connectivity().checkConnectivity());
    if (connectivityResult.contains(ConnectivityResult.none)) {
      throw Exception("No internet connection to reach attacker model.");
    }

    try {
      debugPrint("AttackerApiService: Sending clone request to $endpoint");
      debugPrint("AttackerApiService: Fields - text: $targetText, speaker: ${p.basename(audioPath)}");

      final uri = Uri.parse(endpoint);
      final request = http.MultipartRequest('POST', uri);
      
      request.files.add(await http.MultipartFile.fromPath('speaker', audioPath));
      request.fields['text'] = targetText;

      final streamedResponse = await request.send();
      final responseBytes = await streamedResponse.stream.toBytes();

      if (streamedResponse.statusCode == 200) {
        late Uint8List audioBytesToSave;

        // Support both raw WAV and JSON/Base64
        try {
          final decodedText = utf8.decode(responseBytes);
          final jsonMap = jsonDecode(decodedText);
          
          debugPrint("AttackerApiService received JSON keys: ${jsonMap.keys.toString()}");
          
          final base64String = jsonMap['audio'] ?? jsonMap['cloned_audio'] ?? jsonMap['audio_base64'] ?? jsonMap['audio_data'] ?? jsonMap['file_data'];
          if (base64String != null) {
            audioBytesToSave = base64Decode(base64String);
          } else {
             audioBytesToSave = responseBytes;
          }
        } catch (e) {
          // fallback to raw bytes
          audioBytesToSave = responseBytes;
        }

        final dir = await getApplicationDocumentsDirectory();
        final safeFilename = "cloned_${DateTime.now().millisecondsSinceEpoch}.wav";
        final clonedPath = p.join(dir.path, safeFilename);
        final file = File(clonedPath);
        await file.writeAsBytes(audioBytesToSave);

        debugPrint("AttackerApiService: Cloned audio saved to $clonedPath");
        return clonedPath;
      } else {
        String errorMsg = "Attacker Model Failed (${streamedResponse.statusCode})";
        try {
          final errorStr = String.fromCharCodes(responseBytes);
          if (errorStr.isNotEmpty && errorStr.length < 500) errorMsg = errorStr;
        } catch (_) {}
        throw Exception(errorMsg);
      }
    } on SocketException catch (e) {
      throw Exception("Failed to reach attacker model: ${e.message}");
    }
  }
}
