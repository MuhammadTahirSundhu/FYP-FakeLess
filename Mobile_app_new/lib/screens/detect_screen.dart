import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:just_audio/just_audio.dart';

import '../core/theme.dart';
import '../services/attacker_api_service.dart';
import '../widgets/level_meter.dart';
import '../widgets/waveform_visualizer.dart';
import '../widgets/ab_comparison_player.dart';

class DetectScreen extends StatefulWidget {
  final String? initialAudioPath;
  const DetectScreen({super.key, this.initialAudioPath});

  @override
  State<DetectScreen> createState() => _DetectScreenState();
}

class _DetectScreenState extends State<DetectScreen> {
  final AudioRecorder _recorder = AudioRecorder();
  final AttackerApiService _attackerService = AttackerApiService();
  final TextEditingController _textController = TextEditingController();
  final AudioPlayer _audioPlayerOriginal = AudioPlayer();
  final AudioPlayer _audioPlayerCloned = AudioPlayer();

  String? _audioPath;
  bool _isRecording = false;
  double _amplitude = -60.0;
  bool _isAttacking = false;
  String? _clonedAudioPath;

  bool _isPlayingOriginal = false;
  bool _isPlayingCloned = false;

  @override
  void initState() {
    super.initState();
    // Pre-load audio if navigated from Protect screen
    if (widget.initialAudioPath != null) {
      _audioPath = widget.initialAudioPath;
    }
  }

  @override
  void dispose() {
    if (_isRecording) _recorder.stop();
    _recorder.dispose();
    _textController.dispose();
    _audioPlayerOriginal.dispose();
    _audioPlayerCloned.dispose();
    super.dispose();
  }

  Future<void> _toggleRecording() async {
    if (_isRecording) {
      final path = await _recorder.stop();
      if (!mounted) return;
      setState(() {
        _isRecording = false;
        _amplitude = -60.0;
        _audioPath = path;
        _clonedAudioPath = null;
      });
    } else {
      if (await _recorder.hasPermission()) {
        final dir = await getApplicationDocumentsDirectory();
        final path = p.join(dir.path, 'detect_${DateTime.now().millisecondsSinceEpoch}.wav');
        await _recorder.start(
          const RecordConfig(encoder: AudioEncoder.wav, bitRate: 16000, sampleRate: 16000),
          path: path,
        );
        _recorder.onAmplitudeChanged(const Duration(milliseconds: 80)).listen((amp) {
          if (!mounted) return;
          setState(() => _amplitude = amp.current);
        });
        if (!mounted) return;
        setState(() {
          _isRecording = true;
          _audioPath = null;
          _clonedAudioPath = null;
        });
      }
    }
  }

  Future<void> _pickFile() async {
    final result = await FilePicker.platform.pickFiles(
        type: FileType.custom, allowedExtensions: ['wav', 'mp3', 'm4a']); // Assuming format converter logic is global or we restrict to wav
    if (result != null && result.files.single.path != null) {
      if (!mounted) return;
      setState(() {
        _audioPath = result.files.single.path;
        _clonedAudioPath = null;
      });
    }
  }

  Future<void> _triggerAttack() async {
    if (_audioPath == null) return;
    if (_textController.text.trim().isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter the target text for cloning'), backgroundColor: AppTheme.orangeAccent),
      );
      return;
    }

    setState(() => _isAttacking = true);
    
    try {
      final clonedPath = await _attackerService.simulateAttack(
        audioPath: _audioPath!, 
        targetText: _textController.text.trim()
      );
      if (!mounted) return;
      setState(() {
        _clonedAudioPath = clonedPath;
        _isAttacking = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _isAttacking = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Attacker Error: $e'), backgroundColor: AppTheme.errorRed),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.background,
      body: CustomScrollView(
        slivers: [
          SliverAppBar(
            pinned: true,
            backgroundColor: AppTheme.background,
            automaticallyImplyLeading: widget.initialAudioPath != null,
            leading: widget.initialAudioPath != null
                ? IconButton(
                    icon: const Icon(Icons.arrow_back_ios_new, color: AppTheme.orangeAccent, size: 20),
                    onPressed: () => Navigator.pop(context),
                  )
                : null,
            title: Text('Attacker Simulator',
                style: GoogleFonts.inter(
                    color: AppTheme.orangeAccent,
                    fontWeight: FontWeight.w700,
                    fontSize: 18,
                    letterSpacing: 1)),
          ),
          SliverPadding(
            padding: const EdgeInsets.fromLTRB(20, 12, 20, 100),
            sliver: SliverList(
              delegate: SliverChildListDelegate([
                // ── Hero Banner ──
                Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [AppTheme.errorRed.withOpacity(0.12), AppTheme.background],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: AppTheme.errorRed.withOpacity(0.2)),
                  ),
                  child: Row(children: [
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: AppTheme.errorRed.withOpacity(0.15),
                      ),
                      child: const Icon(Icons.security, color: AppTheme.errorRed, size: 28),
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                        Text('Simulate Voice Cloning Attack',
                            style: GoogleFonts.inter(
                                color: AppTheme.textMain,
                                fontWeight: FontWeight.w700,
                                fontSize: 13)),
                        const SizedBox(height: 4),
                        Text(
                            'Provide audio and target text to a remote cloner model to verify FAKeless protection.',
                            style: GoogleFonts.inter(
                                color: AppTheme.textSecondary, fontSize: 11)),
                      ]),
                    ),
                  ]),
                ).animate().fadeIn().slideY(begin: -0.05),

                const SizedBox(height: 28),

                // ── Pre-loaded audio banner (from Protect screen) ──
                if (widget.initialAudioPath != null)
                  Container(
                    margin: const EdgeInsets.only(bottom: 20),
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                    decoration: BoxDecoration(
                      color: AppTheme.successGreen.withOpacity(0.08),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: AppTheme.successGreen.withOpacity(0.3)),
                    ),
                    child: Row(children: [
                      const Icon(Icons.verified_user, color: AppTheme.successGreen, size: 18),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                          Text('Protected audio pre-loaded',
                              style: GoogleFonts.inter(color: AppTheme.successGreen, fontWeight: FontWeight.w700, fontSize: 13)),
                          Text('Your protected recording is loaded. Run the attacker model below.',
                              style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 11)),
                        ]),
                      ),
                    ]),
                  ).animate().fadeIn().slideY(begin: -0.1),

                // ── Record Button ──
                _SectionLabel(icon: Icons.mic, label: 'Source Audio', color: AppTheme.orangeAccent),
                const SizedBox(height: 20),

                Center(
                  child: GestureDetector(
                    onTap: _isAttacking ? null : _toggleRecording,
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 300),
                      width: 100, height: 100,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        gradient: LinearGradient(
                          colors: _isRecording
                              ? [AppTheme.errorRed, AppTheme.orangeAccent]
                              : [AppTheme.orangeAccent, const Color(0xFFff6b35)],
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                        ),
                        boxShadow: [BoxShadow(
                          color: (_isRecording ? AppTheme.errorRed : AppTheme.orangeAccent).withOpacity(0.5),
                          blurRadius: _isRecording ? 32 : 16,
                          spreadRadius: _isRecording ? 6 : 2,
                        )],
                      ),
                      child: Icon(
                          _isRecording ? Icons.stop_rounded : Icons.mic_rounded,
                          size: 44, color: Colors.black),
                    ),
                  ),
                ).animate(onPlay: _isRecording ? (c) => c.repeat(reverse: true) : null)
                  .scaleXY(begin: 0.95, end: 1.05, duration: 800.ms),

                const SizedBox(height: 10),
                LevelMeter(amplitude: _amplitude, isActive: _isRecording),
                const SizedBox(height: 8),

                Center(
                  child: Text(
                    _isRecording ? '🔴 Recording — tap to stop' :
                    (_audioPath != null ? '✅ Audio ready' : 'Tap to record source'),
                    style: GoogleFonts.inter(
                      color: _isRecording ? AppTheme.errorRed :
                             (_audioPath != null ? AppTheme.successGreen : AppTheme.textSecondary),
                      fontWeight: FontWeight.w500, fontSize: 13,
                    ),
                  ),
                ),

                const SizedBox(height: 14),
                if (!_isRecording)
                  Center(
                    child: OutlinedButton.icon(
                      onPressed: _isAttacking ? null : _pickFile,
                      icon: const Icon(Icons.upload_file, color: AppTheme.orangeAccent),
                      label: Text('Upload Local (.WAV)',
                          style: GoogleFonts.inter(color: AppTheme.orangeAccent, fontWeight: FontWeight.w600)),
                      style: OutlinedButton.styleFrom(
                        side: const BorderSide(color: AppTheme.orangeAccent),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                      ),
                    ),
                  ),

                // ── Waveform ──
                if (_audioPath != null && !_isRecording) ...[
                  const SizedBox(height: 24),
                  _SectionLabel(icon: Icons.graphic_eq, label: 'Audio Selection', color: AppTheme.orangeAccent),
                  const SizedBox(height: 10),
                  WaveformVisualizer(
                    audioPath: _audioPath!,
                    color: AppTheme.orangeAccent,
                    label: p.basename(_audioPath!),
                  ),
                ],

                // ── Deepfake Payload Construction ──
                if (_audioPath != null && !_isRecording) ...[
                  const SizedBox(height: 28),
                  _SectionLabel(icon: Icons.text_snippet, label: 'Target Spoken Text', color: AppTheme.errorRed),
                  const SizedBox(height: 10),
                  TextField(
                    controller: _textController,
                    style: GoogleFonts.inter(color: AppTheme.textMain, fontSize: 14),
                    maxLines: 3,
                    decoration: InputDecoration(
                      hintText: 'e.g., Transfer one million dollars to offshore account...',
                      hintStyle: GoogleFonts.inter(color: AppTheme.textSecondary.withOpacity(0.5)),
                      filled: true,
                      fillColor: AppTheme.surfaceHigh.withOpacity(0.5),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                        borderSide: BorderSide(color: AppTheme.surfaceHigh),
                      ),
                      focusedBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                        borderSide: BorderSide(color: AppTheme.errorRed.withOpacity(0.5)),
                      ),
                    ),
                  ).animate().fadeIn().slideY(begin: 0.1),

                  const SizedBox(height: 24),
                  if (_isAttacking)
                    const Center(
                      child: Column(children: [
                        CircularProgressIndicator(color: AppTheme.errorRed),
                        SizedBox(height: 12),
                        Text('Cloning voice over API...',
                            style: TextStyle(color: AppTheme.textSecondary, fontSize: 13)),
                      ]),
                    )
                  else
                    Container(
                      decoration: BoxDecoration(
                        gradient: LinearGradient(colors: [AppTheme.errorRed, Colors.redAccent]),
                        borderRadius: BorderRadius.circular(14),
                        boxShadow: [BoxShadow(
                            color: AppTheme.errorRed.withOpacity(0.35),
                            blurRadius: 16,
                            offset: const Offset(0, 4))],
                      ),
                      child: Material(
                        color: Colors.transparent,
                        child: InkWell(
                          onTap: _triggerAttack,
                          borderRadius: BorderRadius.circular(14),
                          child: Padding(
                            padding: const EdgeInsets.symmetric(vertical: 16),
                            child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                              const Icon(Icons.smart_toy, color: Colors.black),
                              const SizedBox(width: 10),
                              Text('EXECUTE CLONING ATTACK',
                                  style: GoogleFonts.inter(
                                      color: Colors.black,
                                      fontWeight: FontWeight.w800,
                                      fontSize: 15,
                                      letterSpacing: 1.5)),
                            ]),
                          ),
                        ),
                      ),
                    ).animate().slideY(begin: 0.2).fadeIn(),
                ],

                // ── Resulting Cloned Audio ──
                if (_clonedAudioPath != null && _audioPath != null) ...[
                  const SizedBox(height: 32),
                  const Divider(color: AppTheme.surfaceHigh),
                  const SizedBox(height: 20),
                  _SectionLabel(icon: Icons.compare_arrows, label: 'Attack Results (Listen Closely)', color: AppTheme.orangeAccent),
                  const SizedBox(height: 16),
                  
                  AbComparisonPlayer(
                    isPlayingOriginal: _isPlayingOriginal,
                    isPlayingProtected: _isPlayingCloned,
                    onPlayOriginal: () async {
                      if (_isPlayingCloned) {
                        await _audioPlayerCloned.pause();
                        if (mounted) setState(() => _isPlayingCloned = false);
                      }
                      if (_isPlayingOriginal) {
                        await _audioPlayerOriginal.pause();
                        if (mounted) setState(() => _isPlayingOriginal = false);
                      } else {
                        await _audioPlayerOriginal.setFilePath(_audioPath!);
                        _audioPlayerOriginal.play();
                        if (mounted) setState(() => _isPlayingOriginal = true);
                        _audioPlayerOriginal.playerStateStream.listen((state) {
                          if (state.processingState == ProcessingState.completed) {
                            if (mounted) setState(() => _isPlayingOriginal = false);
                          }
                        });
                      }
                    },
                    onPlayProtected: () async {
                      if (_isPlayingOriginal) {
                        await _audioPlayerOriginal.pause();
                        if (mounted) setState(() => _isPlayingOriginal = false);
                      }
                      if (_isPlayingCloned) {
                        await _audioPlayerCloned.pause();
                        if (mounted) setState(() => _isPlayingCloned = false);
                      } else {
                        await _audioPlayerCloned.setFilePath(_clonedAudioPath!);
                        _audioPlayerCloned.play();
                        if (mounted) setState(() => _isPlayingCloned = true);
                        _audioPlayerCloned.playerStateStream.listen((state) {
                          if (state.processingState == ProcessingState.completed) {
                            if (mounted) setState(() => _isPlayingCloned = false);
                          }
                        });
                      }
                    },
                  ).animate().scaleXY(begin: 0.9).fadeIn(),
                ],
              ]),
            ),
          ),
        ],
      ),
    );
  }
}

class _SectionLabel extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  const _SectionLabel({required this.icon, required this.label, required this.color});

  @override
  Widget build(BuildContext context) {
    return Row(children: [
      Icon(icon, color: color, size: 17),
      const SizedBox(width: 8),
      Text(label, style: GoogleFonts.inter(color: AppTheme.textMain, fontWeight: FontWeight.w700, fontSize: 14)),
      const SizedBox(width: 8),
      Expanded(child: Divider(color: color.withOpacity(0.2))),
    ]);
  }
}
