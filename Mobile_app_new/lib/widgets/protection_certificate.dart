import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:share_plus/share_plus.dart';
import '../core/theme.dart';

/// A premium "Protection Certificate" card displayed after successful protection.
/// Shows metadata about the protection session and can be shared.
class ProtectionCertificate extends StatelessWidget {
  final String audioFileName;
  final int scale;
  final double? pesq;
  final double? stoi;
  final DateTime date;
  final bool isLocalMode;

  const ProtectionCertificate({
    super.key,
    required this.audioFileName,
    required this.scale,
    this.pesq,
    this.stoi,
    required this.date,
    this.isLocalMode = false,
  });

  String get _scaleLabel {
    const labels = ['', 'Minimal', 'Light', 'Standard', 'Strong', 'Maximum'];
    return scale >= 1 && scale <= 5 ? labels[scale] : 'Standard';
  }

  String get _certId {
    // Deterministic short ID from timestamp
    final ts = date.millisecondsSinceEpoch;
    return 'FKL-${ts.toRadixString(16).toUpperCase().substring(0, 8)}';
  }

  Color get _scaleColor {
    if (scale <= 2) return AppTheme.successGreen;
    if (scale == 3) return AppTheme.cyanAccent;
    if (scale == 4) return AppTheme.orangeAccent;
    return AppTheme.errorRed;
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF1A1F35), Color(0xFF0F1624), Color(0xFF1A1F35)],
        ),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: AppTheme.cyanAccent.withOpacity(0.35),
          width: 1.5,
        ),
        boxShadow: [
          BoxShadow(color: AppTheme.cyanAccent.withOpacity(0.12), blurRadius: 24, spreadRadius: 2),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ── Header bar ──
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [AppTheme.cyanAccent.withOpacity(0.15), Colors.transparent],
              ),
              borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
              border: Border(
                bottom: BorderSide(color: AppTheme.cyanAccent.withOpacity(0.2)),
              ),
            ),
            child: Row(children: [
              const Icon(Icons.verified_user, color: AppTheme.cyanAccent, size: 20),
              const SizedBox(width: 8),
              Text(
                'Protection Certificate',
                style: GoogleFonts.inter(
                  color: AppTheme.cyanAccent,
                  fontWeight: FontWeight.w800,
                  fontSize: 14,
                  letterSpacing: 1,
                ),
              ),
              const Spacer(),
              if (isLocalMode)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: AppTheme.orangeAccent.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: AppTheme.orangeAccent.withOpacity(0.4)),
                  ),
                  child: Text('Local Mode',
                      style: GoogleFonts.inter(color: AppTheme.orangeAccent, fontSize: 10, fontWeight: FontWeight.w700)),
                ),
            ]),
          ),

          // ── Body ──
          Padding(
            padding: const EdgeInsets.all(20),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              // File name
              Text(
                audioFileName,
                style: GoogleFonts.inter(color: AppTheme.textMain, fontWeight: FontWeight.w700, fontSize: 15),
                overflow: TextOverflow.ellipsis,
              ),
              const SizedBox(height: 4),
              Text(
                'Protected on ${_formatDate(date)}',
                style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 12),
              ),

              const SizedBox(height: 20),

              // Scale badge
              Row(children: [
                _Badge(
                  icon: Icons.security,
                  label: 'Scale $_scaleLabel',
                  color: _scaleColor,
                  value: '$scale/5',
                ),
                const SizedBox(width: 10),
                _Badge(
                  icon: Icons.graphic_eq,
                  label: 'PESQ',
                  color: AppTheme.cyanAccent,
                  value: pesq?.toStringAsFixed(2) ?? 'N/A',
                ),
                const SizedBox(width: 10),
                _Badge(
                  icon: Icons.record_voice_over,
                  label: 'STOI',
                  color: AppTheme.purpleAccent,
                  value: stoi?.toStringAsFixed(2) ?? 'N/A',
                ),
              ]),

              const SizedBox(height: 20),

              // Quality bar
              if (stoi != null)
                Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Row(children: [
                    Text('Voice Quality Preserved',
                        style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 11)),
                    const Spacer(),
                    Text('${(stoi! * 100).toInt()}%',
                        style: GoogleFonts.inter(color: AppTheme.cyanAccent, fontWeight: FontWeight.w700, fontSize: 11)),
                  ]),
                  const SizedBox(height: 6),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(6),
                    child: LinearProgressIndicator(
                      value: stoi!.clamp(0.0, 1.0),
                      backgroundColor: AppTheme.surfaceHigh,
                      valueColor: AlwaysStoppedAnimation(
                        stoi! > 0.8 ? AppTheme.successGreen : stoi! > 0.6 ? AppTheme.cyanAccent : AppTheme.orangeAccent,
                      ),
                      minHeight: 8,
                    ),
                  ),
                ]),

              const SizedBox(height: 20),

              // Cert ID + timestamp row
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                decoration: BoxDecoration(
                  color: AppTheme.surfaceHigh.withOpacity(0.5),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Row(children: [
                  const Icon(Icons.fingerprint, color: AppTheme.textSecondary, size: 16),
                  const SizedBox(width: 8),
                  Text(_certId,
                      style: GoogleFonts.inter(
                          color: AppTheme.textSecondary,
                          fontSize: 12,
                          letterSpacing: 1.5)),
                  const Spacer(),
                  Text(_formatTime(date),
                      style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 11)),
                ]),
              ),

              const SizedBox(height: 16),

              // Share cert button
              SizedBox(
                width: double.infinity,
                child: OutlinedButton.icon(
                  onPressed: () => _shareCertificate(),
                  icon: const Icon(Icons.share_outlined, color: AppTheme.cyanAccent, size: 16),
                  label: Text('Share Certificate',
                      style: GoogleFonts.inter(color: AppTheme.cyanAccent, fontWeight: FontWeight.w600, fontSize: 13)),
                  style: OutlinedButton.styleFrom(
                    side: const BorderSide(color: AppTheme.cyanAccent),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                    padding: const EdgeInsets.symmetric(vertical: 12),
                  ),
                ),
              ),
            ]),
          ),
        ],
      ),
    ).animate().fadeIn(duration: 500.ms).slideY(begin: 0.08);
  }

  void _shareCertificate() {
    final text = '''
🔒 FAKeless Protection Certificate

📄 File: $audioFileName
📅 Date: ${_formatDate(date)} at ${_formatTime(date)}
🛡️ Scale: $scale — $_scaleLabel
📊 PESQ Score: ${pesq?.toStringAsFixed(2) ?? 'N/A'}
🎤 STOI Score: ${stoi?.toStringAsFixed(2) ?? 'N/A'}
✅ Quality Preserved: ${stoi != null ? (stoi! * 100).toInt() : 'N/A'}%
🔖 Certificate ID: $_certId
${isLocalMode ? '\n⚡ Protected via Local Mode (offline)\n' : ''}
Protected with FAKeless — Anti-Voice-Clone Technology
''';
    Share.share(text, subject: 'FAKeless Protection Certificate');
  }

  static String _formatDate(DateTime d) =>
      '${d.day.toString().padLeft(2, '0')} ${_monthName(d.month)} ${d.year}';

  static String _formatTime(DateTime d) =>
      '${d.hour.toString().padLeft(2, '0')}:${d.minute.toString().padLeft(2, '0')}';

  static String _monthName(int m) {
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return months[m - 1];
  }
}

class _Badge extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  final Color color;
  const _Badge({required this.icon, required this.label, required this.value, required this.color});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 8),
        decoration: BoxDecoration(
          color: color.withOpacity(0.08),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: color.withOpacity(0.25)),
        ),
        child: Column(children: [
          Icon(icon, color: color, size: 16),
          const SizedBox(height: 4),
          Text(value, style: GoogleFonts.inter(color: color, fontWeight: FontWeight.w800, fontSize: 13)),
          const SizedBox(height: 2),
          Text(label, style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 9), textAlign: TextAlign.center),
        ]),
      ),
    );
  }
}
