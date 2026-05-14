import 'package:flutter/material.dart';
import 'package:percent_indicator/percent_indicator.dart';
import '../core/theme.dart';
import 'package:flutter_animate/flutter_animate.dart';

class ResultsDisplay extends StatelessWidget {
  final double? pesqResult;
  final double? stoiResult;

  const ResultsDisplay({
    super.key,
    this.pesqResult,
    this.stoiResult,
  });

  @override
  Widget build(BuildContext context) {
    // PESQ max is roughly 4.5
    final pesqPercent = (pesqResult?.clamp(0.0, 4.5) ?? 0.0) / 4.5;
    // STOI max is 1.0
    final stoiPercent = stoiResult?.clamp(0.0, 1.0) ?? 0.0;

    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        _buildIndicator(
          title: "PESQ Score",
          subtitle: "(Audio Quality)",
          value: pesqResult,
          percent: pesqPercent,
          color: AppTheme.cyanAccent,
        ),
        _buildIndicator(
          title: "STOI Score",
          subtitle: "(Intelligibility)",
          value: stoiResult,
          percent: stoiPercent,
          color: AppTheme.orangeAccent,
        ),
      ],
    ).animate().fadeIn(duration: 600.ms).slideY(begin: 0.2, end: 0, curve: Curves.easeOutQuart);
  }

  Widget _buildIndicator({
    required String title,
    required String subtitle,
    required double? value,
    required double percent,
    required Color color,
  }) {
    return Column(
      children: [
        CircularPercentIndicator(
          radius: 40.0,
          lineWidth: 8.0,
          animation: true,
          animationDuration: 1200,
          percent: percent,
          center: Text(
            value != null ? value.toStringAsFixed(2) : 'N/A',
            style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16.0),
          ),
          circularStrokeCap: CircularStrokeCap.round,
          progressColor: color,
          backgroundColor: Colors.white12,
        ),
        const SizedBox(height: 12),
        Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
        Text(subtitle, style: const TextStyle(fontSize: 10, color: Colors.grey)),
      ],
    );
  }
}
