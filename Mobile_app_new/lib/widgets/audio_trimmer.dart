import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:syncfusion_flutter_sliders/sliders.dart';
import '../core/theme.dart';

/// Audio trimmer widget. Shows the total audio duration and a range slider
/// for the user to select a start/end window. Calls [onTrimChanged] with
/// the selected range so the parent can pass it to the protection pipeline.
class AudioTrimmer extends StatefulWidget {
  final Duration totalDuration;
  final void Function(Duration start, Duration end) onTrimChanged;

  const AudioTrimmer({
    super.key,
    required this.totalDuration,
    required this.onTrimChanged,
  });

  @override
  State<AudioTrimmer> createState() => _AudioTrimmerState();
}

class _AudioTrimmerState extends State<AudioTrimmer> {
  late SfRangeValues _values;

  @override
  void initState() {
    super.initState();
    _values = SfRangeValues(0.0, widget.totalDuration.inMilliseconds.toDouble());
  }

  @override
  void didUpdateWidget(AudioTrimmer old) {
    super.didUpdateWidget(old);
    if (old.totalDuration != widget.totalDuration) {
      _values = SfRangeValues(0.0, widget.totalDuration.inMilliseconds.toDouble());
    }
  }

  String _fmtMs(double ms) {
    final d = Duration(milliseconds: ms.round());
    final m = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final s = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    return '$m:$s';
  }

  @override
  Widget build(BuildContext context) {
    final maxMs = widget.totalDuration.inMilliseconds.toDouble();
    if (maxMs <= 0) return const SizedBox.shrink();

    final startLbl = _fmtMs(_values.start as double);
    final endLbl   = _fmtMs(_values.end as double);
    final dur = Duration(milliseconds: ((_values.end as double) - (_values.start as double)).round());
    final durLbl = '${dur.inMinutes.remainder(60).toString().padLeft(2, '0')}:${dur.inSeconds.remainder(60).toString().padLeft(2, '0')}';

    return Container(
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 10),
      decoration: BoxDecoration(
        color: AppTheme.purpleAccent.withOpacity(0.07),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.purpleAccent.withOpacity(0.25)),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          const Icon(Icons.content_cut, color: AppTheme.purpleAccent, size: 15),
          const SizedBox(width: 6),
          Text('Trim Audio (optional)',
              style: GoogleFonts.inter(color: AppTheme.purpleAccent, fontWeight: FontWeight.w700, fontSize: 13)),
          const Spacer(),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: AppTheme.purpleAccent.withOpacity(0.12),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text('Selection: $durLbl',
                style: GoogleFonts.inter(color: AppTheme.purpleAccent, fontWeight: FontWeight.w600, fontSize: 11)),
          ),
        ]),
        const SizedBox(height: 4),

        SfRangeSlider(
          min: 0.0,
          max: maxMs,
          values: _values,
          activeColor: AppTheme.purpleAccent,
          inactiveColor: AppTheme.surfaceHigh,
          showTicks: false,
          showLabels: false,
          onChanged: (SfRangeValues v) {
            setState(() => _values = v);
            widget.onTrimChanged(
              Duration(milliseconds: (v.start as double).round()),
              Duration(milliseconds: (v.end as double).round()),
            );
          },
        ),

        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 4),
          child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
            Text(startLbl, style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 11)),
            Text('Total: ${_fmtMs(maxMs)}',
                style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 11)),
            Text(endLbl, style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 11)),
          ]),
        ),
      ]),
    ).animate().fadeIn().slideY(begin: 0.08);
  }
}

/// Trims a WAV file to [start]..[end] duration by slicing raw PCM bytes.
/// Returns the path to the newly written trimmed file.
Future<String> trimWavFile({
  required String sourcePath,
  required Duration start,
  required Duration end,
  required String outputPath,
}) async {
  final bytes = await File(sourcePath).readAsBytes();
  if (bytes.length < 44) return sourcePath; // too short to trim

  // Parse WAV header fields we need
  final numChannels    = bytes[22] | (bytes[23] << 8);
  final sampleRate     = bytes[24] | (bytes[25] << 8) | (bytes[26] << 16) | (bytes[27] << 24);
  final bitsPerSample  = bytes[34] | (bytes[35] << 8);
  final bytesPerSample = (bitsPerSample ~/ 8) * numChannels;

  final startByte = (start.inMilliseconds / 1000.0 * sampleRate * bytesPerSample).round();
  final endByte   = (end.inMilliseconds   / 1000.0 * sampleRate * bytesPerSample).round();

  // Data chunk starts at byte 44 for standard PCM WAV
  const dataOffset = 44;
  final dataLength = bytes.length - dataOffset;

  final clampedStart = startByte.clamp(0, dataLength);
  final clampedEnd   = endByte.clamp(clampedStart, dataLength);

  final trimmedData = bytes.sublist(dataOffset + clampedStart, dataOffset + clampedEnd);
  final newDataSize = trimmedData.length;
  final newFileSize = 44 + newDataSize;

  // Build a new valid WAV header
  final header = ByteData(44);
  // RIFF chunk
  header.setUint8(0, 0x52); header.setUint8(1, 0x49); // 'R','I'
  header.setUint8(2, 0x46); header.setUint8(3, 0x46); // 'F','F'
  header.setUint32(4, newFileSize - 8, Endian.little);
  header.setUint8(8, 0x57); header.setUint8(9, 0x41); // 'W','A'
  header.setUint8(10, 0x56); header.setUint8(11, 0x45); // 'V','E'
  // fmt sub-chunk
  header.setUint8(12, 0x66); header.setUint8(13, 0x6D); // 'f','m'
  header.setUint8(14, 0x74); header.setUint8(15, 0x20); // 't',' '
  header.setUint32(16, 16, Endian.little); // chunk size
  header.setUint16(20, 1, Endian.little);  // PCM format
  header.setUint16(22, numChannels, Endian.little);
  header.setUint32(24, sampleRate, Endian.little);
  header.setUint32(28, sampleRate * bytesPerSample, Endian.little); // byte rate
  header.setUint16(32, bytesPerSample, Endian.little);    // block align
  header.setUint16(34, bitsPerSample, Endian.little);
  // data sub-chunk
  header.setUint8(36, 0x64); header.setUint8(37, 0x61); // 'd','a'
  header.setUint8(38, 0x74); header.setUint8(39, 0x61); // 't','a'
  header.setUint32(40, newDataSize, Endian.little);

  final out = File(outputPath);
  final sink = out.openWrite();
  sink.add(header.buffer.asUint8List());
  sink.add(trimmedData);
  await sink.flush();
  await sink.close();

  return outputPath;
}
