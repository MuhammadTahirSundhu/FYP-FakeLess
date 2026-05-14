import 'package:flutter/foundation.dart';
import 'package:google_generative_ai/google_generative_ai.dart';

class AiGuardService {
  // IMPORTANT: For production, load API Key from flutter_dotenv or CI secrets.
  // Using an empty key placeholder for safety. Replace with an actual key if it's available.
  static const String _geminiApiKey = 'AIzaSyD4F-452TVuZoVmVj6iSH9X5NIfz1vRvZA';

  Future<int> determineScaleFromIntent(String naturalLanguageInput) async {
    if (_geminiApiKey.isEmpty) {
      debugPrint("AiGuardService: No API Key provided, returning mocked calculation.");
      // MOCK BEHAVIOR
      await Future.delayed(const Duration(milliseconds: 1500));
      if (naturalLanguageInput.toLowerCase().contains('high') || naturalLanguageInput.toLowerCase().contains('bank')) return 5;
      if (naturalLanguageInput.toLowerCase().contains('low')) return 1;
      return 3; 
    }

    try {
      final model = GenerativeModel(
        model: 'gemini-2.5-flash',
        apiKey: _geminiApiKey,
      );

      final prompt = """
You are FAKeless AiGuard, a strict text-processing bot. 
The user is providing an instruction for protecting an audio recording. 
Interpret the instructions and output ONLY a single integer from 1 to 5.
1 = Very light protection, 5 = Bank-grade/High intensity protection.
Instruction: "$naturalLanguageInput"
Output single integer:
""";

      final content = [Content.text(prompt)];
      final response = await model.generateContent(content);
      
      final textOutput = response.text?.trim() ?? '3';
      final parsed = int.tryParse(textOutput);
      
      if (parsed != null && parsed >= 1 && parsed <= 5) {
        return parsed;
      }
      return 3; // Default fallback scale
    } catch (e) {
      debugPrint("AiGuardService Failed: $e");
      return 3; // Graceful degradation
    }
  }
}
