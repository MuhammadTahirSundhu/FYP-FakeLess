import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../core/theme.dart';

/// Animated microphone level meter.
/// Pass [amplitude] in dBFS (from record package) — typically -60 to 0.
/// When [isActive] is false, the widget hides itself.
class LevelMeter extends StatefulWidget {
  final double amplitude; // dBFS value (negative, 0 = max)
  final bool isActive;
  final int barCount;

  const LevelMeter({
    super.key,
    required this.amplitude,
    required this.isActive,
    this.barCount = 24,
  });

  @override
  State<LevelMeter> createState() => _LevelMeterState();
}

class _LevelMeterState extends State<LevelMeter> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  final _rng = Random();

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this, duration: const Duration(milliseconds: 100))
      ..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!widget.isActive) return const SizedBox.shrink();

    return AnimatedBuilder(
      animation: _controller,
      builder: (_, __) {
        // Normalise amplitude: dBFS range -60..0 → 0.0..1.0
        final normalised = ((widget.amplitude + 60) / 60).clamp(0.0, 1.0);
        return Container(
          height: 52,
          padding: const EdgeInsets.symmetric(horizontal: 20),
          child: CustomPaint(
            painter: _LevelPainter(
              normalised: normalised,
              barCount: widget.barCount,
              rng: _rng,
            ),
            child: const SizedBox.expand(),
          ),
        );
      },
    ).animate().fadeIn(duration: 300.ms);
  }
}

class _LevelPainter extends CustomPainter {
  final double normalised; // 0.0 – 1.0
  final int barCount;
  final Random rng;

  _LevelPainter({required this.normalised, required this.barCount, required this.rng});

  @override
  void paint(Canvas canvas, Size size) {
    final barWidth = (size.width / barCount) * 0.6;
    final gap = (size.width / barCount) * 0.4;
    final maxBarHeight = size.height;

    for (int i = 0; i < barCount; i++) {
      // Centre-weighted distribution: bars near centre are taller
      final centreFactor = 1.0 - (((i - barCount / 2).abs()) / (barCount / 2)) * 0.4;
      final micro = rng.nextDouble() * 0.15; // slight random jitter
      final barHeight = ((normalised * centreFactor + micro) * maxBarHeight).clamp(3.0, maxBarHeight);

      final x = i * (barWidth + gap);
      final y = (maxBarHeight - barHeight) / 2;
      final rect = RRect.fromRectAndRadius(
        Rect.fromLTWH(x, y, barWidth, barHeight),
        const Radius.circular(3),
      );

      // Colour gradient: low = cyan, high = orange/red
      final fraction = barHeight / maxBarHeight;
      final color = Color.lerp(AppTheme.cyanAccent, AppTheme.errorRed, fraction * fraction)!;
      canvas.drawRRect(rect, Paint()..color = color);
    }
  }

  @override
  bool shouldRepaint(covariant _LevelPainter old) => true;
}
