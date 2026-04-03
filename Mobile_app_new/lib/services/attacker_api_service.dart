import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'dart:io';

class AttackerApiService {
  String get endpoint => dotenv.env['ATTACKER_API_ENDPOINT'] ?? 'http://128.0.0.1:8001/clone_voice';

  Future<String?> simulateAttack({required String audioPath, required String targetText}) async {
    // Check Connectivity
    final connectivityResult = await (Connectivity().checkConnectivity());
    if (connectivityResult.contains(ConnectivityResult.none)) {
      throw Exception("No internet connection to reach attacker model.");
    }

    try {
      final uri = Uri.parse(endpoint);
      final request = http.MultipartRequest('POST', uri);
      
      // The API structure as requested from .env config payload rules
      request.files.add(await http.MultipartFile.fromPath('audio', audioPath));
      request.fields['text'] = targetText;

      final streamedResponse = await request.send();
      final responseBytes = await streamedResponse.stream.toBytes();

      if (streamedResponse.statusCode == 200) {
        String decodedText = utf8.decode(responseBytes);
        String responseBase64;
        
        try {
          final jsonMap = jsonDecode(decodedText);
          responseBase64 = jsonMap['audio'] ?? jsonMap['audio_base64'] ?? jsonMap['cloned_audio'];
        } catch (_) {
          // Fallback if raw base64 string
          responseBase64 = decodedText;
        }

        // Save cloned audio to temp file
        final decodedAudioBytes = base64Decode(responseBase64);
        final dir = await getApplicationDocumentsDirectory();
        final safeFilename = "cloned_${DateTime.now().millisecondsSinceEpoch}.wav";
        final clonedPath = p.join(dir.path, safeFilename);
        final file = File(clonedPath);
        await file.writeAsBytes(decodedAudioBytes);

        return clonedPath;
      } else {
        String errorMsg = "Attacker Model Failed";
        try {
          final errorStr = String.fromCharCodes(responseBytes);
          if (errorStr.isNotEmpty) errorMsg = errorStr;
        } catch (_) {}
        throw Exception(errorMsg);
      }
    } on SocketException catch (e) {
      throw Exception("Failed to reach attacker model: ${e.message}");
    }
  }
}
