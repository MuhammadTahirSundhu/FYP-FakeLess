import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:share_plus/share_plus.dart';
import '../core/theme.dart';
import '../services/history_service.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  final HistoryService _historyService = HistoryService();
  List<ProtectionSession> _sessions = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final sessions = await _historyService.loadAll();
    if (!mounted) return;
    setState(() {
      _sessions = sessions;
      _loading = false;
    });
  }

  Future<void> _delete(ProtectionSession session) async {
    await _historyService.delete(session.id);
    await _load();
  }

  Future<void> _share(ProtectionSession session) async {
    final file = File(session.audioPath);
    if (await file.exists()) {
      Share.shareXFiles(
        [XFile(session.audioPath)],
        text: 'My FAKeless protected audio — Scale ${session.scale} (${session.scaleLabel})',
      );
    } else {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Audio file not found on device.'), backgroundColor: AppTheme.errorRed),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.background,
      appBar: AppBar(
        title: Text('Protection History', style: GoogleFonts.inter(color: AppTheme.cyanAccent, fontWeight: FontWeight.w700, fontSize: 18, letterSpacing: 1)),
        actions: [
          if (_sessions.isNotEmpty)
            IconButton(
              tooltip: 'Clear all history',
              icon: const Icon(Icons.delete_sweep_outlined, color: AppTheme.textSecondary),
              onPressed: () async {
                final confirmed = await showDialog<bool>(
                  context: context,
                  builder: (ctx) => AlertDialog(
                    backgroundColor: AppTheme.surfaceHigh,
                    title: Text('Clear History', style: GoogleFonts.inter(color: AppTheme.textMain)),
                    content: Text('Delete all saved sessions?', style: GoogleFonts.inter(color: AppTheme.textSecondary)),
                    actions: [
                      TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
                      TextButton(
                        onPressed: () => Navigator.pop(ctx, true),
                        child: const Text('Clear All', style: TextStyle(color: AppTheme.errorRed)),
                      ),
                    ],
                  ),
                );
                if (confirmed == true) {
                  await _historyService.clearAll();
                  await _load();
                }
              },
            ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator(color: AppTheme.cyanAccent))
          : _sessions.isEmpty
              ? _EmptyState()
              : RefreshIndicator(
                  color: AppTheme.cyanAccent,
                  backgroundColor: AppTheme.surface,
                  onRefresh: _load,
                  child: ListView.builder(
                    padding: const EdgeInsets.fromLTRB(16, 12, 16, 100),
                    itemCount: _sessions.length,
                    itemBuilder: (ctx, i) {
                      return _SessionCard(
                        session: _sessions[i],
                        onDelete: () => _delete(_sessions[i]),
                        onShare: () => _share(_sessions[i]),
                      ).animate().fadeIn(delay: (i * 60).ms).slideY(begin: 0.15, delay: (i * 60).ms);
                    },
                  ),
                ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
        Container(
          padding: const EdgeInsets.all(24),
          decoration: BoxDecoration(
            color: AppTheme.cyanAccent.withOpacity(0.08),
            shape: BoxShape.circle,
          ),
          child: const Icon(Icons.lock_clock, color: AppTheme.cyanAccent, size: 52),
        ).animate(onPlay: (c) => c.repeat(reverse: true)).scaleXY(begin: 0.95, end: 1.05, duration: 2.seconds),
        const SizedBox(height: 24),
        Text('No sessions yet', style: GoogleFonts.inter(color: AppTheme.textMain, fontWeight: FontWeight.w600, fontSize: 18)),
        const SizedBox(height: 10),
        Text(
          'Start protecting your voice!\nYour sessions will appear here.',
          textAlign: TextAlign.center,
          style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 14, height: 1.6),
        ),
      ]).animate().fadeIn(delay: 200.ms),
    );
  }
}

class _SessionCard extends StatelessWidget {
  final ProtectionSession session;
  final VoidCallback onDelete;
  final VoidCallback onShare;

  const _SessionCard({required this.session, required this.onDelete, required this.onShare});

  Color get _scaleColor {
    if (session.scale <= 2) return AppTheme.successGreen;
    if (session.scale == 3) return AppTheme.cyanAccent;
    if (session.scale == 4) return AppTheme.orangeAccent;
    return AppTheme.errorRed;
  }

  @override
  Widget build(BuildContext context) {
    final dateStr = '${session.date.day}/${session.date.month}/${session.date.year}  ${session.date.hour.toString().padLeft(2,'0')}:${session.date.minute.toString().padLeft(2,'0')}';

    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      decoration: BoxDecoration(
        gradient: AppTheme.cardGradient,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: _scaleColor.withOpacity(0.25)),
      ),
      child: Column(children: [
        // Header row
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 14, 8, 0),
          child: Row(children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
              decoration: BoxDecoration(
                color: _scaleColor.withOpacity(0.15),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: _scaleColor.withOpacity(0.4)),
              ),
              child: Text(
                'Scale ${session.scale} · ${session.scaleLabel}',
                style: GoogleFonts.inter(color: _scaleColor, fontWeight: FontWeight.w700, fontSize: 12),
              ),
            ),
            if (session.isLocalMode) ...[
              const SizedBox(width: 8),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
                decoration: BoxDecoration(color: AppTheme.orangeAccent.withOpacity(0.15), borderRadius: BorderRadius.circular(8)),
                child: Text('Local Mode', style: GoogleFonts.inter(color: AppTheme.orangeAccent, fontWeight: FontWeight.w600, fontSize: 10)),
              ),
            ],
            const Spacer(),
            IconButton(
              icon: const Icon(Icons.share, color: AppTheme.cyanAccent, size: 20),
              onPressed: onShare,
              tooltip: 'Share',
            ),
            IconButton(
              icon: const Icon(Icons.delete_outline, color: AppTheme.textSecondary, size: 20),
              onPressed: onDelete,
              tooltip: 'Delete',
            ),
          ]),
        ),

        // Metrics row
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 10, 16, 14),
          child: Row(children: [
            _MetricPill(label: 'PESQ', value: session.pesq?.toStringAsFixed(2) ?? 'N/A', color: AppTheme.cyanAccent),
            const SizedBox(width: 10),
            _MetricPill(label: 'STOI', value: session.stoi?.toStringAsFixed(2) ?? 'N/A', color: AppTheme.orangeAccent),
            const Spacer(),
            Row(children: [
              const Icon(Icons.access_time, color: AppTheme.textSecondary, size: 13),
              const SizedBox(width: 4),
              Text(dateStr, style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 11)),
            ]),
          ]),
        ),
      ]),
    );
  }
}

class _MetricPill extends StatelessWidget {
  final String label;
  final String value;
  final Color color;
  const _MetricPill({required this.label, required this.value, required this.color});

  @override
  Widget build(BuildContext context) {
    return Row(children: [
      Text('$label ', style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 12)),
      Text(value, style: GoogleFonts.inter(color: color, fontWeight: FontWeight.w700, fontSize: 14)),
    ]);
  }
}
