import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
  // Core palette
  static const Color background  = Color(0xFF0A0E1A); // Deep navy black
  static const Color surface     = Color(0xFF111827); // Card surface
  static const Color surfaceHigh = Color(0xFF1C2333); // Elevated surface

  static const Color cyanAccent   = Color(0xFF00E5FF); // Electric cyan
  static const Color cyanDim      = Color(0xFF00B8CC); // Muted cyan
  static const Color orangeAccent = Color(0xFFFF8C00); // Safety orange
  static const Color purpleAccent = Color(0xFF7C3AED); // Deep violet
  static const Color successGreen = Color(0xFF10B981); // Emerald green
  static const Color errorRed     = Color(0xFFEF4444); // Clear red

  static const Color textMain      = Color(0xFFF1F5F9); // Near white
  static const Color textSecondary = Color(0xFF94A3B8); // Slate grey

  // Gradients
  static const LinearGradient heroBg = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [Color(0xFF0A0E1A), Color(0xFF0D1B2E), Color(0xFF0A1628)],
  );

  static const LinearGradient cyanGradient = LinearGradient(
    colors: [Color(0xFF00E5FF), Color(0xFF0090A8)],
  );

  static const LinearGradient orangeGradient = LinearGradient(
    colors: [Color(0xFFFF8C00), Color(0xFFFF5722)],
  );

  static const LinearGradient cardGradient = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [Color(0xFF1C2333), Color(0xFF111827)],
  );

  static ThemeData get genericTheme {
    final base = ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: background,
      colorScheme: const ColorScheme.dark(
        background: background,
        primary: cyanAccent,
        secondary: orangeAccent,
        surface: surface,
        onPrimary: Colors.black,
        onSecondary: Colors.black,
        onBackground: textMain,
        onSurface: textMain,
      ),
      appBarTheme: AppBarTheme(
        backgroundColor: background,
        foregroundColor: cyanAccent,
        elevation: 0,
        centerTitle: true,
        titleTextStyle: GoogleFonts.inter(
          color: cyanAccent,
          fontWeight: FontWeight.w700,
          fontSize: 18,
          letterSpacing: 1.5,
        ),
      ),
      floatingActionButtonTheme: const FloatingActionButtonThemeData(
        backgroundColor: cyanAccent,
        foregroundColor: Colors.black,
      ),
      cardTheme: CardThemeData(
        color: surface,
        elevation: 0,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      ),
      bottomNavigationBarTheme: const BottomNavigationBarThemeData(
        backgroundColor: surface,
        selectedItemColor: cyanAccent,
        unselectedItemColor: textSecondary,
        type: BottomNavigationBarType.fixed,
        elevation: 16,
      ),
    );

    return base.copyWith(
      textTheme: GoogleFonts.interTextTheme(base.textTheme).copyWith(
        headlineLarge: GoogleFonts.inter(color: textMain, fontWeight: FontWeight.w800, fontSize: 28),
        headlineMedium: GoogleFonts.inter(color: textMain, fontWeight: FontWeight.w700, fontSize: 22),
        titleLarge: GoogleFonts.inter(color: textMain, fontWeight: FontWeight.w600, fontSize: 18),
        titleMedium: GoogleFonts.inter(color: textMain, fontWeight: FontWeight.w500, fontSize: 16),
        bodyLarge: GoogleFonts.inter(color: textMain, fontSize: 15),
        bodyMedium: GoogleFonts.inter(color: textSecondary, fontSize: 13),
        labelSmall: GoogleFonts.inter(color: textSecondary, fontSize: 11, letterSpacing: 0.8),
      ),
    );
  }
}
