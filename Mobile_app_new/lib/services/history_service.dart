import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

class ProtectionSession {
  final String id;
  final DateTime date;
  final int scale;
  final double? pesq;
  final double? stoi;
  final String audioPath;
  final bool isLocalMode;

  ProtectionSession({
    required this.id,
    required this.date,
    required this.scale,
    this.pesq,
    this.stoi,
    required this.audioPath,
    this.isLocalMode = false,
  });

  Map<String, dynamic> toJson() => {
    'id': id,
    'date': date.toIso8601String(),
    'scale': scale,
    'pesq': pesq,
    'stoi': stoi,
    'audioPath': audioPath,
    'isLocalMode': isLocalMode,
  };

  factory ProtectionSession.fromJson(Map<String, dynamic> json) {
    return ProtectionSession(
      id: json['id'] as String,
      date: DateTime.parse(json['date'] as String),
      scale: json['scale'] as int,
      pesq: json['pesq'] != null ? (json['pesq'] as num).toDouble() : null,
      stoi: json['stoi'] != null ? (json['stoi'] as num).toDouble() : null,
      audioPath: json['audioPath'] as String,
      isLocalMode: json['isLocalMode'] as bool? ?? false,
    );
  }

  String get scaleLabel {
    switch (scale) {
      case 1: return 'Minimal';
      case 2: return 'Light';
      case 3: return 'Standard';
      case 4: return 'Strong';
      case 5: return 'Maximum';
      default: return 'Standard';
    }
  }
}

class HistoryService {
  static const _storageKey = 'fakeless_session_history';

  Future<List<ProtectionSession>> loadAll() async {
    final prefs = await SharedPreferences.getInstance();
    final rawList = prefs.getStringList(_storageKey) ?? [];
    return rawList.map((raw) {
      final map = jsonDecode(raw) as Map<String, dynamic>;
      return ProtectionSession.fromJson(map);
    }).toList()
      ..sort((a, b) => b.date.compareTo(a.date)); // Newest first
  }

  Future<void> save(ProtectionSession session) async {
    final prefs = await SharedPreferences.getInstance();
    final rawList = prefs.getStringList(_storageKey) ?? [];
    rawList.add(jsonEncode(session.toJson()));
    // Keep last 50 sessions to avoid unbounded growth
    final trimmed = rawList.length > 50 ? rawList.sublist(rawList.length - 50) : rawList;
    await prefs.setStringList(_storageKey, trimmed);
  }

  Future<void> delete(String id) async {
    final prefs = await SharedPreferences.getInstance();
    final rawList = prefs.getStringList(_storageKey) ?? [];
    rawList.removeWhere((raw) {
      final map = jsonDecode(raw) as Map<String, dynamic>;
      return map['id'] == id;
    });
    await prefs.setStringList(_storageKey, rawList);
  }

  Future<void> clearAll() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_storageKey);
  }
}
