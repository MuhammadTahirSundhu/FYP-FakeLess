import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import '../core/theme.dart';

/// Reads a WAV file and renders a waveform using CustomPainter.
/// Shows a skeleton while loading.
class WaveformVisualizer extends StatefulWidget {
  final String audioPath;
  final Color color;
  final String label;
  final double height;

  const WaveformVisualizer({
    super.key,
    required this.audioPath,
    required this.color,
    required this.label,
    this.height = 72,
  });

  @override
  State<WaveformVisualizer> createState() => _WaveformVisualizerState();
}

class _WaveformVisualizerState extends State<WaveformVisualizer> {
  List<double> _samples = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void didUpdateWidget(WaveformVisualizer old) {
    super.didUpdateWidget(old);
    if (old.audioPath != widget.audioPath) _load();
  }

  Future<void> _load() async {
    setState(() => _loading = true);
    try {
      final samples = await _extractSamples(widget.audioPath, targetPoints: 180);
      if (!mounted) return;
      setState(() {
        _samples = samples;
        _loading = false;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() => _loading = false);
    }
  }

  /// Reads raw WAV PCM bytes and down-samples to [targetPoints] normalised values.
  static Future<List<double>> _extractSamples(String path, {int targetPoints = 180}) async {
    final bytes = await File(path).readAsBytes();
    if (bytes.length < 44) return []; // WAV header is 44 bytes

    // WAV header offset 34: bits per sample (2 bytes, little-endian)
    final bitsPerSample = bytes[34] | (bytes[35] << 8);
    final dataStart = 44; // Standard PCM WAV

    final data = bytes.sublist(dataStart);
    if (data.isEmpty) return [];

    final bytesPerSample = (bitsPerSample / 8).ceil();
    final sampleCount = data.length ~/ bytesPerSample;
    if (sampleCount == 0) return [];

    final step = max(1, sampleCount ~/ targetPoints);
    final result = <double>[];

    for (int i = 0; i < sampleCount; i += step) {
      final offset = i * bytesPerSample;
      if (offset + bytesPerSample > data.length) break;

      double value;
      if (bitsPerSample == 16) {
        final raw = data[offset] | (data[offset + 1] << 8);
        // Convert unsigned to signed
        final signed = raw > 32767 ? raw - 65536 : raw;
        value = signed / 32768.0;
      } else if (bitsPerSample == 8) {
        value = (data[offset] - 128) / 128.0;
      } else {
        // 32-bit float PCM
        final byteData = ByteData.sublistView(Uint8List.fromList(data.sublist(offset, offset + 4)));
        value = byteData.getFloat32(0, Endian.little).clamp(-1.0, 1.0);
      }
      result.add(value.abs()); // Use absolute value for display
    }

    // Normalise to max amplitude
    final maxVal = result.isEmpty ? 1.0 : result.reduce(max);
    return maxVal > 0 ? result.map((v) => v / maxVal).toList() : result;
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(bottom: 6),
          child: Row(children: [
            Container(width: 8, height: 8, decoration: BoxDecoration(shape: BoxShape.circle, color: widget.color)),
            const SizedBox(width: 6),
            Text(widget.label, style: GoogleFonts.inter(color: widget.color, fontSize: 11, fontWeight: FontWeight.w600)),
          ]),
        ),
        Container(
          height: widget.height,
          decoration: BoxDecoration(
            color: widget.color.withOpacity(0.05),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: widget.color.withOpacity(0.2)),
          ),
          clipBehavior: Clip.hardEdge,
          child: _loading
              ? _Skeleton(color: widget.color)
              : _samples.isEmpty
                  ? Center(child: Text('No data', style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 11)))
                  : CustomPaint(
                      painter: _WaveformPainter(samples: _samples, color: widget.color),
                      child: const SizedBox.expand(),
                    ),
        ),
      ],
    ).animate().fadeIn(duration: 400.ms).slideY(begin: 0.1);
  }
}

class _WaveformPainter extends CustomPainter {
  final List<double> samples;
  final Color color;
  _WaveformPainter({required this.samples, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = 1.5
      ..strokeCap = StrokeCap.round;

    final midY = size.height / 2;
    final stepX = size.width / samples.length;

    for (int i = 0; i < samples.length; i++) {
      final x = i * stepX;
      final amp = samples[i] * midY * 0.9;
      canvas.drawLine(
        Offset(x, midY - amp),
        Offset(x, midY + amp),
        paint,
      );
    }

    // Centre line
    canvas.drawLine(
      Offset(0, midY),
      Offset(size.width, midY),
      Paint()
        ..color = color.withOpacity(0.15)
        ..strokeWidth = 0.5,
    );
  }

  @override
  bool shouldRepaint(covariant _WaveformPainter old) =>
      old.samples != samples || old.color != color;
}

class _Skeleton extends StatelessWidget {
  final Color color;
  const _Skeleton({required this.color});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: SizedBox(
        width: 20,
        height: 20,
        child: CircularProgressIndicator(strokeWidth: 2, color: color),
      ),
    );
  }
}
