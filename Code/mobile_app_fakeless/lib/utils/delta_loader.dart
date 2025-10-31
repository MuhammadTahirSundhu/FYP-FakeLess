import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;

class DeltaLoader {
  static Future<List<double>> loadDelta1D(String assetPath) async {
    final jsonString = await rootBundle.loadString(assetPath);
    final List<dynamic> rawData = json.decode(jsonString);
    // Flatten if nested list
    final flattened = rawData.expand((e) => e is List ? e : [e]).toList();
    return flattened.map((e) => (e as num).toDouble()).toList();
  }
}
