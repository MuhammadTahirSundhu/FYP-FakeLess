import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import '../core/theme.dart';

/// A/B Audio Comparison Player.
/// Lets the user toggle between Original and Protected audio with an animated switch.
/// Calls [onPlayOriginal] or [onPlayProtected] depending on current selection.
class AbComparisonPlayer extends StatefulWidget {
  final bool isPlayingOriginal;
  final bool isPlayingProtected;
  final VoidCallback onPlayOriginal;
  final VoidCallback onPlayProtected;

  const AbComparisonPlayer({
    super.key,
    required this.isPlayingOriginal,
    required this.isPlayingProtected,
    required this.onPlayOriginal,
    required this.onPlayProtected,
  });

  @override
  State<AbComparisonPlayer> createState() => _AbComparisonPlayerState();
}

class _AbComparisonPlayerState extends State<AbComparisonPlayer> {
  /// false = Original, true = Protected
  bool _showProtected = false;

  bool get _isPlaying => _showProtected ? widget.isPlayingProtected : widget.isPlayingOriginal;

  void _toggleTrack() => setState(() => _showProtected = !_showProtected);

  void _togglePlay() {
    if (_showProtected) {
      widget.onPlayProtected();
    } else {
      widget.onPlayOriginal();
    }
  }

  @override
  Widget build(BuildContext context) {
    final activeColor = _showProtected ? AppTheme.orangeAccent : AppTheme.cyanAccent;
    final label = _showProtected ? 'Protected' : 'Original';
    final sublabel = _showProtected ? 'Anti-clone watermark applied' : 'Raw recording — no protection';

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: AppTheme.cardGradient,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: activeColor.withOpacity(0.3)),
      ),
      child: Column(children: [
        // Track toggle row
        Row(children: [
          _TrackChip(
            label: 'A  Original',
            isActive: !_showProtected,
            color: AppTheme.cyanAccent,
            onTap: () => setState(() => _showProtected = false),
          ),
          const SizedBox(width: 10),
          _TrackChip(
            label: 'B  Protected',
            isActive: _showProtected,
            color: AppTheme.orangeAccent,
            onTap: () => setState(() => _showProtected = true),
          ),
        ]),

        const SizedBox(height: 16),

        // Player row
        AnimatedSwitcher(
          duration: const Duration(milliseconds: 300),
          child: Row(
            key: ValueKey(_showProtected),
            children: [
              // Colour dot
              Container(
                width: 10, height: 10,
                decoration: BoxDecoration(shape: BoxShape.circle, color: activeColor),
              ),
              const SizedBox(width: 10),
              Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('Now listening: $label',
                    style: GoogleFonts.inter(color: activeColor, fontWeight: FontWeight.w700, fontSize: 13)),
                Text(sublabel,
                    style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 11)),
              ])),
              // Big play/pause button
              GestureDetector(
                onTap: _togglePlay,
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  width: 52, height: 52,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: activeColor.withOpacity(0.15),
                    border: Border.all(color: activeColor, width: 1.5),
                    boxShadow: _isPlaying
                        ? [BoxShadow(color: activeColor.withOpacity(0.35), blurRadius: 12)]
                        : null,
                  ),
                  child: Icon(
                    _isPlaying ? Icons.pause_rounded : Icons.play_arrow_rounded,
                    color: activeColor, size: 28,
                  ),
                ),
              ),
            ],
          ).animate().fadeIn(duration: 200.ms),
        ),

        const SizedBox(height: 12),

        // Tap to switch hint
        GestureDetector(
          onTap: _toggleTrack,
          child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
            const Icon(Icons.swap_horiz, color: AppTheme.textSecondary, size: 14),
            const SizedBox(width: 4),
            Text('Tap to switch to ${_showProtected ? "Original (A)" : "Protected (B)"}',
                style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 11)),
          ]),
        ),
      ]),
    ).animate().fadeIn(duration: 400.ms);
  }
}

class _TrackChip extends StatelessWidget {
  final String label;
  final bool isActive;
  final Color color;
  final VoidCallback onTap;
  const _TrackChip({required this.label, required this.isActive, required this.color, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          padding: const EdgeInsets.symmetric(vertical: 8),
          decoration: BoxDecoration(
            color: isActive ? color.withOpacity(0.15) : Colors.transparent,
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: isActive ? color : AppTheme.textSecondary.withOpacity(0.2)),
          ),
          child: Center(
            child: Text(label,
                style: GoogleFonts.inter(
                    color: isActive ? color : AppTheme.textSecondary,
                    fontWeight: isActive ? FontWeight.w700 : FontWeight.w400,
                    fontSize: 12)),
          ),
        ),
      ),
    );
  }
}
