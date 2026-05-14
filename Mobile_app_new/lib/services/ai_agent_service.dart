import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:http/http.dart' as http;

class AiAgentResponse {
  final String message;
  final int actionScale;

  AiAgentResponse({required this.message, required this.actionScale});

  factory AiAgentResponse.fromJson(Map<String, dynamic> json) {
    return AiAgentResponse(
      message: json['message'] as String? ?? "I didn't quite get that.",
      actionScale: (json['actionScale'] as num?)?.toInt() ?? 0,
    );
  }
}

class AiAgentService {
  static const String _systemInstruction = '''
You are the FAKeless Assistant, an intelligent and helpful conversational agent dedicated to championing the FAKeless brand.

Here is everything you know about the FAKeless app:
1. Purpose: FAKeless stops AI voice cloners and deepfake tools from stealing or mimicking the speaker\'s voice.
2. Tradeoff (Quality vs. Protection): Adding more protection adds more noise (perturbations) to the audio file.
   - Low protection (Scale 1-2): Preserves pristine audio quality but provides less defense against highly advanced cloners.
   - High protection (Scale 4-5): Provides powerful, bank-grade defense but might make the speech sound less crisp.
3. Metrics (PESQ & STOI):
   - PESQ (Perceptual Evaluation of Speech Quality): Measures how clear the speech sounds. Higher is better!
   - STOI (Short-Time Objective Intelligibility): Measures how easy it is to understand the words. Higher is better!

Your strict instructions:
1. Answer in a very simple, concise, non-technical way. Speak directly to normal everyday users.
2. IMPORTANT: If the user asks something NOT related to FAKeless, audio protection, or voice cloning, reply ONLY with: "FAKeless can tell you right now only information related to this app."
3. If the user wants to protect/secure an audio file, determine the appropriate protection scale (1 to 5).

You MUST always return a valid JSON object with exactly two keys:
- "message": your conversational response as a string.
- "actionScale": an integer from 1-5 if the user requested audio protection, or 0 otherwise.
''';

  String get _apiKey => dotenv.env['GEMINI_API_KEY'] ?? '';

  // Use the v1 (stable) REST endpoint directly — avoids SDK v1beta issues
  static const String _baseUrl = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent';
  Future<AiAgentResponse> sendMessage(String text) async {
    if (_apiKey.isEmpty) {
      debugPrint("AiAgentService: GEMINI_API_KEY is not set in .env");
      return AiAgentResponse(
          message: "Error: No API key configured. Please add GEMINI_API_KEY to .env",
          actionScale: 0);
    }

    try {
      final url = Uri.parse('$_baseUrl?key=$_apiKey');

      final body = jsonEncode({
        "systemInstruction": {
          "parts": [{"text": _systemInstruction}]
        },
        "contents": [
          {
            "role": "user",
            "parts": [{"text": text}]
          }
        ],
        "generationConfig": {
          "responseMimeType": "application/json",
          "temperature": 0.4,
          "maxOutputTokens": 512,
        }
      });

      final response = await http
          .post(
            url,
            headers: {'Content-Type': 'application/json'},
            body: body,
          )
          .timeout(const Duration(seconds: 30));

      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body) as Map<String, dynamic>;
        final candidates = decoded['candidates'] as List<dynamic>?;
        final rawText = candidates?.first['content']['parts']?.first['text'] as String? ?? '{}';

        final cleanText = _stripJsonFences(rawText);
        final jsonResponse = jsonDecode(cleanText) as Map<String, dynamic>;
        return AiAgentResponse.fromJson(jsonResponse);
      } else {
        debugPrint("AiAgentService HTTP Error ${response.statusCode}: ${response.body}");
        return AiAgentResponse(
            message: "Server error (${response.statusCode}). Please try again.",
            actionScale: 0);
      }
    } catch (e) {
      debugPrint("AiAgentService Exception: $e");
      return AiAgentResponse(
          message: "Sorry, I'm having trouble connecting right now. Please try again.",
          actionScale: 0);
    }
  }

  String _stripJsonFences(String raw) {
    String s = raw.trim();
    if (s.startsWith('```json')) s = s.substring(7);
    if (s.startsWith('```')) s = s.substring(3);
    if (s.endsWith('```')) s = s.substring(0, s.length - 3);
    return s.trim();
  }
}
