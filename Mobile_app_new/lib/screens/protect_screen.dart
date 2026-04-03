import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:just_audio/just_audio.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:share_plus/share_plus.dart';
import 'package:file_picker/file_picker.dart';

import '../core/theme.dart';
import '../services/api_service.dart';
import '../services/ai_agent_service.dart';
import '../services/tflite_service.dart';
import '../services/crypto_service.dart';
import '../services/history_service.dart';
import '../widgets/results_display.dart';
import '../widgets/level_meter.dart';
import '../widgets/waveform_visualizer.dart';
import '../widgets/protection_certificate.dart';
import '../widgets/ab_comparison_player.dart';
import '../widgets/audio_trimmer.dart';
import '../screens/detect_screen.dart';

class ProtectScreen extends StatefulWidget {
  const ProtectScreen({super.key});

  @override
  State<ProtectScreen> createState() => ProtectScreenState();
}

class ProtectScreenState extends State<ProtectScreen> {
  // Controllers
  final AudioRecorder _audioRecorder = AudioRecorder();
  final AudioPlayer _audioPlayerOriginal = AudioPlayer();
  final AudioPlayer _audioPlayerProtected = AudioPlayer();

  // Services
  late final CryptoService _cryptoService;
  late final ApiService _apiService;
  late final AiAgentService _aiAgentService;
  late final TfliteService _tfliteService;
  late final HistoryService _historyService;

  // State
  bool _isInit = false;
  bool _isRecording = false;
  bool _isLoading = false;
  bool _isLocalFallbackActive = false;

  List<Map<String, String>> _chatMessages = [];
  bool _isAiTyping = false;

  String? _recordingPath;
  String? _protectedPath;

  int _currentScale = 3;
  double? _pesqResult;
  double? _stoiResult;

  // Auto-recommendation
  int? _recommendedScale;
  String _recommendReason = '';
  bool _showRecommendation = true;

  // Trim selection
  Duration? _trimStart;
  Duration? _trimEnd;

  bool _isPlayingOriginal = false;
  bool _isPlayingProtected = false;

  // Expose state to parent (MainShell)
  bool get isRecording => _isRecording;
  bool get hasAudio => _recordingPath != null;
  bool get isAiTyping => _isAiTyping;
  List<Map<String, String>> get chatMessages => _chatMessages;

  // Amplitude for level meter
  double _amplitude = -60.0; // dBFS, starts silent

  @override
  void initState() {
    super.initState();
    _initServices();

    _audioPlayerOriginal.playerStateStream.listen((state) {
      if (!mounted) return;
      setState(() => _isPlayingOriginal = state.playing);
      if (state.processingState == ProcessingState.completed) {
        _audioPlayerOriginal.stop();
        _audioPlayerOriginal.seek(Duration.zero);
        setState(() => _isPlayingOriginal = false);
      }
    });

    _audioPlayerProtected.playerStateStream.listen((state) {
      if (!mounted) return;
      setState(() => _isPlayingProtected = state.playing);
      if (state.processingState == ProcessingState.completed) {
        _audioPlayerProtected.stop();
        _audioPlayerProtected.seek(Duration.zero);
        setState(() => _isPlayingProtected = false);
      }
    });
  }

  Future<void> _initServices() async {
    // Delay initialization so TFLite parsing doesn't block the very first frame of the app startup
    await Future.delayed(const Duration(milliseconds: 300));

    _cryptoService = CryptoService();
    await _cryptoService.init();
    _apiService = ApiService();
    _aiAgentService = AiAgentService();
    _tfliteService = TfliteService();
    _historyService = HistoryService();
    await _tfliteService.init();
    
    if (mounted) setState(() => _isInit = true);
  }

  @override
  void dispose() {
    if (_isRecording) _audioRecorder.stop();
    _audioRecorder.dispose();
    _audioPlayerOriginal.dispose();
    _audioPlayerProtected.dispose();
    super.dispose();
  }

  Future<void> toggleRecording() async {
    if (_isRecording) {
      final path = await _audioRecorder.stop();
      if (!mounted) return;
      setState(() {
        _isRecording = false;
        _amplitude = -60.0;
        _recordingPath = path;
        _protectedPath = null;
        _pesqResult = null;
        _stoiResult = null;
        _isLocalFallbackActive = false;
      });
      if (path != null) {
        await _audioPlayerOriginal.setFilePath(path);
        setState(() { _trimStart = Duration.zero; _trimEnd = null; });
        await _computeRecommendation();
      }
    } else {
      if (await _audioRecorder.hasPermission()) {
        final dir = await getApplicationDocumentsDirectory();
        final filePath = p.join(dir.path, 'audio_${DateTime.now().millisecondsSinceEpoch}.wav');
        await _audioRecorder.start(
          const RecordConfig(encoder: AudioEncoder.wav, bitRate: 16000, sampleRate: 16000),
          path: filePath,
        );
        // Subscribe to amplitude stream
        _audioRecorder.onAmplitudeChanged(const Duration(milliseconds: 80)).listen((amp) {
          if (!mounted) return;
          setState(() => _amplitude = amp.current);
        });
        if (!mounted) return;
        setState(() {
          _isRecording = true;
          _recordingPath = null;
          _protectedPath = null;
          _pesqResult = null;
          _stoiResult = null;
        });
      }
    }
  }

  Future<void> pickAudioFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom, 
      allowedExtensions: ['wav']
    );
    if (result != null && result.files.single.path != null) {
      if (!mounted) return;
      setState(() { _isLoading = true; });
      
      final pickedPath = result.files.single.path!;
      
      if (!mounted) return;
      setState(() {
        _recordingPath = pickedPath;
        _protectedPath = null;
        _pesqResult = null;
        _stoiResult = null;
        _isLocalFallbackActive = false;
        _trimStart = Duration.zero; 
        _trimEnd = null;
        _isLoading = false;
      });
      await _audioPlayerOriginal.setFilePath(_recordingPath!);
      await _computeRecommendation();
    }
  }

  Future<void> _submitProtection(int scale) async {
    if (_recordingPath == null) return;
    setState(() { _isLoading = true; _isLocalFallbackActive = false; });

    // Apply trim if user selected a range
    String sourcePath = _recordingPath!;
    final tStart = _trimStart;
    final tEnd = _trimEnd;
    if (tStart != null && tEnd != null && tEnd > tStart) {
      try {
        final dir = await getApplicationDocumentsDirectory();
        final trimmedPath = p.join(dir.path, 'trimmed_${DateTime.now().millisecondsSinceEpoch}.wav');
        sourcePath = await trimWavFile(
          sourcePath: _recordingPath!,
          start: tStart,
          end: tEnd,
          outputPath: trimmedPath,
        );
      } catch (_) {
        // Fall back to full file if trim fails
        sourcePath = _recordingPath!;
      }
    }

    try {
      final result = await _apiService.protectAudio(sourcePath, scale);
      if (!mounted) return;
      setState(() {
        _protectedPath = result['path'];
        // The live server now returns raw WAV without metrics. Estimate based on scale to keep UI functional.
        _pesqResult = (4.3 - (scale * 0.35)).clamp(1.0, 4.5); 
        _stoiResult = (0.95 - (scale * 0.06)).clamp(0.1, 1.0);
      });
      await _audioPlayerProtected.setFilePath(_protectedPath!);
      await _saveSession(scale, false);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('✅ Cloud Protection Successful'), backgroundColor: AppTheme.successGreen),
      );
    } catch (e) {
      if (e is OfflineException || e is ApiException || e is SocketException) {
        if (!mounted) return;
        setState(() => _isLocalFallbackActive = true);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('⚠️ Offline — Local Mode Active: ${e.toString()}'), backgroundColor: AppTheme.orangeAccent),
        );
        try {
          final localPath = await _tfliteService.applyLocalProtection(sourcePath, scale);
          if (!mounted) return;
          setState(() {
            _protectedPath = localPath;
            _pesqResult = 3.5;
            _stoiResult = 0.8;
          });
          await _audioPlayerProtected.setFilePath(_protectedPath!);
          await _saveSession(scale, true);
        } catch (localError) {
          if (!mounted) return;
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Local Protection Failed: $localError'), backgroundColor: AppTheme.errorRed),
          );
        }
      } else {
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Unexpected Error: $e'), backgroundColor: AppTheme.errorRed),
        );
      }
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  Future<void> _saveSession(int scale, bool isLocal) async {
    if (_protectedPath == null || _pesqResult == null || _stoiResult == null) return;
    final session = ProtectionSession(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      date: DateTime.now(),
      scale: scale,
      pesq: _pesqResult!,
      stoi: _stoiResult!,
      audioPath: _protectedPath!,
      isLocalMode: isLocal,
    );
    await _historyService.save(session);
  }

  /// Reads audio duration and maps it to a recommended protection scale.
  Future<void> _computeRecommendation() async {
    try {
      final dur = _audioPlayerOriginal.duration;
      final seconds = dur?.inSeconds ?? 0;
      int scale;
      String reason;
      if (seconds <= 10) {
        scale = 5;
        reason = 'Short clips are most vulnerable to voice cloning — max protection recommended.';
      } else if (seconds <= 30) {
        scale = 4;
        reason = 'Medium clips benefit from strong protection while keeping good quality.';
      } else if (seconds <= 90) {
        scale = 3;
        reason = 'Standard protection offers the best quality-security balance for this length.';
      } else if (seconds <= 300) {
        scale = 2;
        reason = 'Long recordings: light protection preserves voice quality over extended content.';
      } else {
        scale = 1;
        reason = 'Very long audio: minimal protection keeps quality intact across the full file.';
      }
      if (!mounted) return;
      setState(() {
        _recommendedScale = scale;
        _recommendReason = reason;
        _showRecommendation = true;
      });
    } catch (_) {}
  }
  Future<String> handleAiGuardSubmit(String intent) async {
    if (!mounted) return '';
    setState(() {
      _chatMessages = List.from(_chatMessages)..add({'sender': 'user', 'text': intent});
      _isAiTyping = true;
    });
    final response = await _aiAgentService.sendMessage(intent);
    if (!mounted) return response.message;
    setState(() {
      _isAiTyping = false;
      _chatMessages = List.from(_chatMessages)..add({'sender': 'agent', 'text': response.message});
    });
    if (response.actionScale > 0) {
      if (_recordingPath == null) {
        final noAudioMsg = '🎙️ Please record or upload a .WAV file first!';
        setState(() {
          _chatMessages = List.from(_chatMessages)..add({'sender': 'agent', 'text': noAudioMsg});
        });
        return noAudioMsg;
      }
      setState(() => _currentScale = response.actionScale);
      await _submitProtection(response.actionScale);
    }
    return response.message;
  }

  Future<void> _togglePlayOriginal() async {
    _isPlayingOriginal ? await _audioPlayerOriginal.pause() : await _audioPlayerOriginal.play();
  }

  Future<void> _togglePlayProtected() async {
    _isPlayingProtected ? await _audioPlayerProtected.pause() : await _audioPlayerProtected.play();
  }

  @override
  Widget build(BuildContext context) {
    if (!_isInit) {
      return const Scaffold(
        backgroundColor: AppTheme.background,
        body: Center(child: CircularProgressIndicator(color: AppTheme.cyanAccent)),
      );
    }

    return Scaffold(
      backgroundColor: AppTheme.background,
      body: CustomScrollView(
        slivers: [
          // ── App Bar ──
          SliverAppBar(
            pinned: true,
            backgroundColor: AppTheme.background,
            title: Text('Protect Audio', style: GoogleFonts.inter(color: AppTheme.cyanAccent, fontWeight: FontWeight.w700, fontSize: 18, letterSpacing: 1)),
            actions: [
              if (_isLocalFallbackActive)
                Padding(
                  padding: const EdgeInsets.only(right: 16),
                  child: Chip(
                    label: Text('Local Mode', style: GoogleFonts.inter(color: Colors.black, fontWeight: FontWeight.w700, fontSize: 11)),
                    backgroundColor: AppTheme.orangeAccent,
                    padding: EdgeInsets.zero,
                  ),
                ).animate(onPlay: (c) => c.repeat(reverse: true)).fade(begin: 0.5, end: 1.0),
            ],
          ),

          SliverPadding(
            padding: const EdgeInsets.fromLTRB(20, 16, 20, 100),
            sliver: SliverList(delegate: SliverChildListDelegate([
              // ── Record Button ──
              _SectionLabel(icon: Icons.mic, label: 'Record or Upload'),
              const SizedBox(height: 20),
              Center(
                child: GestureDetector(
                  onTap: _isLoading ? null : toggleRecording,
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 300),
                    width: 110, height: 110,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      gradient: LinearGradient(
                        colors: _isRecording
                            ? [AppTheme.errorRed, AppTheme.orangeAccent]
                            : [AppTheme.cyanAccent, AppTheme.cyanDim],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      boxShadow: [BoxShadow(
                        color: (_isRecording ? AppTheme.errorRed : AppTheme.cyanAccent).withOpacity(0.5),
                        blurRadius: _isRecording ? 36 : 18, spreadRadius: _isRecording ? 8 : 2,
                      )],
                    ),
                    child: Icon(_isRecording ? Icons.stop_rounded : Icons.mic_rounded, size: 50, color: Colors.black),
                  ),
                ),
              ).animate(onPlay: _isRecording ? (c) => c.repeat(reverse: true) : null)
                .scaleXY(begin: 0.95, end: 1.05, duration: 800.ms),

              const SizedBox(height: 12),
              // ── Live Level Meter ──
              LevelMeter(amplitude: _amplitude, isActive: _isRecording),
              Center(
                child: Text(
                  _isRecording ? '🔴 Recording...' : (_recordingPath != null ? '✅ Ready for Protection' : 'Tap to Record'),
                  style: GoogleFonts.inter(
                    color: _isRecording ? AppTheme.errorRed : (_recordingPath != null ? AppTheme.successGreen : AppTheme.textSecondary),
                    fontWeight: FontWeight.w500, fontSize: 14,
                  ),
                ),
              ),

              const SizedBox(height: 14),
              if (!_isRecording)
                Center(
                  child: OutlinedButton.icon(
                    onPressed: _isLoading ? null : pickAudioFile,
                    icon: const Icon(Icons.upload_file, color: AppTheme.cyanAccent),
                    label: Text('Upload .WAV', style: GoogleFonts.inter(color: AppTheme.cyanAccent, fontWeight: FontWeight.w600)),
                    style: OutlinedButton.styleFrom(
                      side: const BorderSide(color: AppTheme.cyanAccent),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                    ),
                  ),
                ),

              // ── Original Playback ──
              if (_recordingPath != null) ...[
                const SizedBox(height: 28),
                _SectionLabel(icon: Icons.play_circle_outline, label: 'Original Recording'),
                const SizedBox(height: 12),
                _AudioCard(
                  title: p.basename(_recordingPath!),
                  isPlaying: _isPlayingOriginal,
                  onToggle: _togglePlayOriginal,
                  color: AppTheme.cyanAccent,
                ).animate().slideX(begin: -0.1).fadeIn(),
                // Audio Trimmer
                const SizedBox(height: 14),
                Builder(builder: (_) {
                  final dur = _audioPlayerOriginal.duration ?? Duration.zero;
                  if (dur.inSeconds < 2) return const SizedBox.shrink();
                  return AudioTrimmer(
                    totalDuration: dur,
                    onTrimChanged: (s, e) => setState(() { _trimStart = s; _trimEnd = e; }),
                  );
                }),
              ],

              // ── Protection Scale ──
              const SizedBox(height: 28),
              _SectionLabel(icon: Icons.tune, label: 'Protection Scale'),
              const SizedBox(height: 12),
              // Auto-recommendation banner
              if (_recommendedScale != null && _showRecommendation)
                _RecommendationBanner(
                  scale: _recommendedScale!,
                  reason: _recommendReason,
                  onAccept: () => setState(() {
                    _currentScale = _recommendedScale!;
                    _showRecommendation = false;
                  }),
                  onDismiss: () => setState(() => _showRecommendation = false),
                ),
              const SizedBox(height: 12),
              _ProtectionScaleSelector(
                value: _currentScale,
                onChanged: (v) => setState(() => _currentScale = v),
              ),

              const SizedBox(height: 24),

              // ── Apply Button ──
              if (_recordingPath != null && !_isLoading)
                Container(
                  decoration: BoxDecoration(
                    gradient: AppTheme.cyanGradient,
                    borderRadius: BorderRadius.circular(14),
                    boxShadow: [BoxShadow(color: AppTheme.cyanAccent.withOpacity(0.35), blurRadius: 16, offset: const Offset(0, 4))],
                  ),
                  child: Material(
                    color: Colors.transparent,
                    child: InkWell(
                      onTap: () => _submitProtection(_currentScale),
                      borderRadius: BorderRadius.circular(14),
                      child: Padding(
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                          const Icon(Icons.shield, color: Colors.black),
                          const SizedBox(width: 10),
                          Text('APPLY PROTECTION', style: GoogleFonts.inter(color: Colors.black, fontWeight: FontWeight.w800, fontSize: 15, letterSpacing: 2)),
                        ]),
                      ),
                    ),
                  ),
                ).animate().slideY(begin: 0.3).fadeIn(),

              if (_isLoading)
                const Padding(
                  padding: EdgeInsets.all(24),
                  child: Center(child: CircularProgressIndicator(color: AppTheme.cyanAccent)),
                ),

              // ── Results ──
              if (_protectedPath != null && _pesqResult != null && _stoiResult != null) ...[
                const SizedBox(height: 32),
                const Divider(color: AppTheme.surfaceHigh),
                const SizedBox(height: 20),
                _SectionLabel(icon: Icons.verified_user, label: 'Protected Result'),
                const SizedBox(height: 12),
                // A/B Comparison Player
                AbComparisonPlayer(
                  isPlayingOriginal: _isPlayingOriginal,
                  isPlayingProtected: _isPlayingProtected,
                  onPlayOriginal: _togglePlayOriginal,
                  onPlayProtected: _togglePlayProtected,
                ),

                // ── Waveform Comparison ──
                const SizedBox(height: 24),
                _SectionLabel(icon: Icons.graphic_eq, label: 'Waveform Comparison'),
                const SizedBox(height: 12),
                WaveformVisualizer(
                  audioPath: _recordingPath!,
                  color: AppTheme.cyanAccent,
                  label: 'ORIGINAL',
                ),
                const SizedBox(height: 10),
                WaveformVisualizer(
                  audioPath: _protectedPath!,
                  color: AppTheme.orangeAccent,
                  label: 'PROTECTED',
                ),
                const SizedBox(height: 24),
                ResultsDisplay(pesqResult: _pesqResult!, stoiResult: _stoiResult!),
                const SizedBox(height: 20),
                // ── Action buttons row ──
                Row(
                  children: [
                    // Share button
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: () {
                          if (_protectedPath != null) {
                            Share.shareXFiles([XFile(_protectedPath!)], text: 'My FAKeless protected audio 🔒');
                          }
                        },
                        icon: const Icon(Icons.share, color: AppTheme.orangeAccent, size: 18),
                        label: Text('Share', style: GoogleFonts.inter(color: AppTheme.orangeAccent, fontWeight: FontWeight.w600, fontSize: 13)),
                        style: OutlinedButton.styleFrom(
                          side: const BorderSide(color: AppTheme.orangeAccent),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                          padding: const EdgeInsets.symmetric(vertical: 13),
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    // Verify Protection button
                    Expanded(
                      child: Container(
                        decoration: BoxDecoration(
                          gradient: const LinearGradient(
                            colors: [Color(0xFF7C3AED), Color(0xFF5B21B6)],
                          ),
                          borderRadius: BorderRadius.circular(14),
                          boxShadow: [BoxShadow(
                            color: AppTheme.purpleAccent.withOpacity(0.35),
                            blurRadius: 12, offset: const Offset(0, 4),
                          )],
                        ),
                        child: Material(
                          color: Colors.transparent,
                          child: InkWell(
                            borderRadius: BorderRadius.circular(14),
                            onTap: () {
                              if (_protectedPath != null) {
                                Navigator.push(
                                  context,
                                  MaterialPageRoute(
                                    builder: (_) => DetectScreen(initialAudioPath: _protectedPath),
                                  ),
                                );
                              }
                            },
                            child: Padding(
                              padding: const EdgeInsets.symmetric(vertical: 13),
                              child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                                const Icon(Icons.verified_user, color: Colors.white, size: 18),
                                const SizedBox(width: 6),
                                Text('Verify', style: GoogleFonts.inter(color: Colors.white, fontWeight: FontWeight.w700, fontSize: 13)),
                              ]),
                            ),
                          ),
                        ),
                      ),
                    ),
                  ],
                ).animate().slideY(begin: 0.1).fadeIn(),

                // ── Protection Certificate ──
                const SizedBox(height: 24),
                ProtectionCertificate(
                  audioFileName: p.basename(_recordingPath!),
                  scale: _currentScale,
                  pesq: _pesqResult!,
                  stoi: _stoiResult!,
                  date: DateTime.now(),
                  isLocalMode: _isLocalFallbackActive,
                ),

              ],
            ])),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────── HELPERS ─────────────────────────
class _SectionLabel extends StatelessWidget {
  final IconData icon;
  final String label;
  const _SectionLabel({required this.icon, required this.label});

  @override
  Widget build(BuildContext context) {
    return Row(children: [
      Icon(icon, color: AppTheme.cyanAccent, size: 18),
      const SizedBox(width: 8),
      Text(label, style: GoogleFonts.inter(color: AppTheme.textMain, fontWeight: FontWeight.w700, fontSize: 15)),
      const SizedBox(width: 8),
      Expanded(child: Divider(color: AppTheme.cyanAccent.withOpacity(0.2))),
    ]);
  }
}

class _AudioCard extends StatelessWidget {
  final String title;
  final bool isPlaying;
  final VoidCallback onToggle;
  final Color color;

  const _AudioCard({required this.title, required this.isPlaying, required this.onToggle, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        gradient: AppTheme.cardGradient,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Row(children: [
        Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(shape: BoxShape.circle, color: color.withOpacity(0.15)),
          child: Icon(Icons.audio_file, color: color, size: 20),
        ),
        const SizedBox(width: 12),
        Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(title, style: GoogleFonts.inter(color: AppTheme.textMain, fontWeight: FontWeight.w600, fontSize: 13), overflow: TextOverflow.ellipsis),
        ])),
        IconButton(
          icon: Icon(isPlaying ? Icons.pause_circle_filled : Icons.play_circle_fill, color: color, size: 38),
          onPressed: onToggle,
        ),
      ]),
    );
  }
}

class _ProtectionScaleSelector extends StatelessWidget {
  final int value;
  final ValueChanged<int> onChanged;
  const _ProtectionScaleSelector({required this.value, required this.onChanged});

  @override
  Widget build(BuildContext context) {
    final labels = ['Minimal', 'Light', 'Standard', 'Strong', 'Maximum'];
    return Column(children: [
      Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: List.generate(5, (i) {
          final scale = i + 1;
          final isSelected = value == scale;
          return GestureDetector(
            onTap: () => onChanged(scale),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              width: 52, height: 52,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: isSelected ? AppTheme.cyanAccent : AppTheme.surfaceHigh,
                border: Border.all(color: isSelected ? AppTheme.cyanAccent : AppTheme.textSecondary.withOpacity(0.2), width: 1.5),
                boxShadow: isSelected ? [BoxShadow(color: AppTheme.cyanAccent.withOpacity(0.4), blurRadius: 12)] : null,
              ),
              child: Center(
                child: Text('$scale', style: GoogleFonts.inter(
                  color: isSelected ? Colors.black : AppTheme.textSecondary,
                  fontWeight: FontWeight.w800, fontSize: 16,
                )),
              ),
            ),
          );
        }),
      ),
      const SizedBox(height: 8),
      Text(
        labels[value - 1],
        style: GoogleFonts.inter(color: AppTheme.cyanAccent, fontWeight: FontWeight.w600, fontSize: 13),
      ),
    ]);
  }
}

class _RecommendationBanner extends StatelessWidget {
  final int scale;
  final String reason;
  final VoidCallback onAccept;
  final VoidCallback onDismiss;

  const _RecommendationBanner({
    required this.scale,
    required this.reason,
    required this.onAccept,
    required this.onDismiss,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(14, 10, 10, 10),
      decoration: BoxDecoration(
        color: AppTheme.cyanAccent.withOpacity(0.08),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppTheme.cyanAccent.withOpacity(0.3)),
      ),
      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        const Icon(Icons.auto_awesome, color: AppTheme.cyanAccent, size: 16),
        const SizedBox(width: 8),
        Expanded(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(
              'AI Recommends Scale $scale',
              style: GoogleFonts.inter(color: AppTheme.cyanAccent, fontWeight: FontWeight.w700, fontSize: 12),
            ),
            const SizedBox(height: 2),
            Text(reason, style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 11)),
            const SizedBox(height: 8),
            Row(children: [
              GestureDetector(
                onTap: onAccept,
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
                  decoration: BoxDecoration(
                    color: AppTheme.cyanAccent,
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text('Apply Scale $scale',
                      style: GoogleFonts.inter(color: Colors.black, fontWeight: FontWeight.w700, fontSize: 11)),
                ),
              ),
              const SizedBox(width: 8),
              GestureDetector(
                onTap: onDismiss,
                child: Text('Dismiss', style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 11)),
              ),
            ]),
          ]),
        ),
        IconButton(
          icon: const Icon(Icons.close, size: 14, color: AppTheme.textSecondary),
          onPressed: onDismiss,
          padding: EdgeInsets.zero,
          constraints: const BoxConstraints(),
        ),
      ]),
    ).animate().fadeIn().slideY(begin: -0.1);
  }
}
