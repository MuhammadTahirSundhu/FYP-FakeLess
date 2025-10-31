import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';

Future<String?> applyDeltaWithPython(String inputPath) async {
  const platform = MethodChannel('com.example.mobile_app_fakeless/python');
  try {
    final result = await platform.invokeMethod('applyDelta', {'path': inputPath});
    return result;
  } on PlatformException catch (e) {
    print("Python error: ${e.message}");
    return null;
  }
}


Future<void> writeWavPcm16(File outFile, List<double> samples, int sampleRate) async {
  const numChannels = 1;
  const bitsPerSample = 16;
  final byteRate = sampleRate * numChannels * bitsPerSample ~/ 8;
  final blockAlign = numChannels * bitsPerSample ~/ 8;
  final subChunk2Size = samples.length * numChannels * bitsPerSample ~/ 8;
  final chunkSize = 4 + (8 + 16) + (8 + subChunk2Size);

  final bytes = BytesBuilder();

  bytes.add(_ascii('RIFF'));
  bytes.add(_u32(chunkSize));
  bytes.add(_ascii('WAVE'));

  bytes.add(_ascii('fmt '));
  bytes.add(_u32(16));
  bytes.add(_u16(1)); // PCM
  bytes.add(_u16(numChannels));
  bytes.add(_u32(sampleRate));
  bytes.add(_u32(byteRate));
  bytes.add(_u16(blockAlign));
  bytes.add(_u16(bitsPerSample));

  bytes.add(_ascii('data'));
  bytes.add(_u32(subChunk2Size));

  final bd = ByteData(2);
  for (final s in samples) {
    final v = (s.clamp(-1.0, 1.0) * 32767).round();
    bd.setInt16(0, v, Endian.little);
    bytes.add(bd.buffer.asUint8List());
  }

  await outFile.writeAsBytes(bytes.toBytes());
}

Uint8List _ascii(String s) => Uint8List.fromList(s.codeUnits);
Uint8List _u16(int v) {
  final b = ByteData(2)..setUint16(0, v, Endian.little);
  return b.buffer.asUint8List();
}
Uint8List _u32(int v) {
  final b = ByteData(4)..setUint32(0, v, Endian.little);
  return b.buffer.asUint8List();
}
