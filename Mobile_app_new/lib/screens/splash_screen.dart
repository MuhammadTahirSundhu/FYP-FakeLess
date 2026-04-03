import 'dart:math';
import 'package:flutter/material.dart';
import '../core/theme.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'main_shell.dart';
import 'onboarding_screen.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _morphAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1400),
    );

    // Morphs from 0.0 (Waves) to 1.0 (Shield)
    _morphAnimation = CurvedAnimation(
      parent: _controller,
      curve: const Interval(0.4, 1.0, curve: Curves.easeInOutCubic),
    );

    // Load prefs concurrently with the animation so there's no gap after
    bool hasSeenOnboarding = false;
    SharedPreferences.getInstance().then((prefs) {
      hasSeenOnboarding = prefs.getBool('has_seen_onboarding') ?? false;
    });

    _controller.forward().then((_) {
      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        PageRouteBuilder(
          transitionDuration: const Duration(milliseconds: 400),
          pageBuilder: (_, __, ___) => hasSeenOnboarding ? const MainShell() : const OnboardingScreen(),
          transitionsBuilder: (_, animation, __, child) => FadeTransition(
            opacity: animation,
            child: child,
          ),
        ),
      );
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.background,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            AnimatedBuilder(
              animation: _controller,
              builder: (context, child) {
                return SizedBox(
                  width: 200,
                  height: 200,
                  child: CustomPaint(
                    painter: WaveShieldPainter(
                      progress: _controller.value,
                      morphProgress: _morphAnimation.value,
                    ),
                  ),
                );
              },
            ),
            const SizedBox(height: 40),
            const Text(
              "FAKeless",
              style: TextStyle(
                color: AppTheme.cyanAccent,
                fontSize: 32,
                fontWeight: FontWeight.bold,
                letterSpacing: 8,
              ),
            ).animate().fade(duration: 1000.ms).slideY(begin: 0.5, end: 0),
            const SizedBox(height: 8),
            const Text(
              "AI Audio Protection",
              style: TextStyle(
                color: AppTheme.textSecondary,
                fontSize: 16,
                letterSpacing: 2,
              ),
            ).animate(delay: 400.ms).fade(duration: 800.ms),
          ],
        ),
      ),
    );
  }
}

class WaveShieldPainter extends CustomPainter {
  final double progress; // 0.0 to 1.0 overall time
  final double morphProgress; // 0.0 to 1.0 (0=waves, 1=shield)

  WaveShieldPainter({
    required this.progress,
    required this.morphProgress,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4.0
      ..strokeCap = StrokeCap.round;

    // Interpolate color from Cyan (Waves) to Orange (Shield)
    paint.color = Color.lerp(AppTheme.cyanAccent, AppTheme.orangeAccent, morphProgress)!;

    final path = Path();

    // Parameters for shapes
    final width = size.width * 0.8;
    final numWaves = 5;
    final waveSpacing = width / (numWaves - 1);
    final startX = center.dx - width / 2;

    for (int i = 0; i < numWaves; i++) {
      // 1. Calculate Wave points
      double x = startX + (i * waveSpacing);
      // Pulsating sine wave effect
      double waveAmplitude = 40.0 * sin((progress * pi * 4) + i);
      
      // Enforce zero amplitude at edges to make it look contained
      if (i == 0 || i == numWaves - 1) waveAmplitude = 10.0 * sin(progress * pi * 8);

      final waveTop = Offset(x, center.dy - waveAmplitude);
      final waveBottom = Offset(x, center.dy + waveAmplitude);

      // 2. Calculate Shield points
      // We shape the 5 lines into a classic shield boundary
      double shieldYTop;
      double shieldYBottom;

      if (i == 0 || i == numWaves - 1) {
        // Outer edges of shield
        shieldYTop = center.dy - 30;
        shieldYBottom = center.dy + 10;
      } else if (i == 1 || i == numWaves - 2) {
        shieldYTop = center.dy - 40;
        shieldYBottom = center.dy + 40;
      } else {
        // Center of shield (pointy bottom)
        shieldYTop = center.dy - 45;
        shieldYBottom = center.dy + 70;
      }

      final targetTop = Offset(x, shieldYTop);
      final targetBottom = Offset(x, shieldYBottom);

      // 3. Morph between Wave and Shield
      final currentTop = Offset.lerp(waveTop, targetTop, morphProgress)!;
      final currentBottom = Offset.lerp(waveBottom, targetBottom, morphProgress)!;

      path.moveTo(currentTop.dx, currentTop.dy);
      path.lineTo(currentBottom.dx, currentBottom.dy);
    }

    // Draw the morphing lines
    canvas.drawPath(path, paint);

    // Draw an outline boundary for the shield as it fully forms
    if (morphProgress > 0) {
      final outlinePaint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0
        ..color = AppTheme.orangeAccent.withOpacity(morphProgress);
      
      final outlinePath = Path();
      outlinePath.moveTo(startX, center.dy - 30);
      outlinePath.lineTo(center.dx, center.dy - 50); // Top point
      outlinePath.lineTo(startX + width, center.dy - 30);
      outlinePath.quadraticBezierTo(startX + width, center.dy + 40, center.dx, center.dy + 80); // Bottom point
      outlinePath.quadraticBezierTo(startX, center.dy + 40, startX, center.dy - 30);
      
      canvas.drawPath(outlinePath, outlinePaint);
    }
  }

  @override
  bool shouldRepaint(covariant WaveShieldPainter oldDelegate) {
    return oldDelegate.progress != progress || oldDelegate.morphProgress != morphProgress;
  }
}
