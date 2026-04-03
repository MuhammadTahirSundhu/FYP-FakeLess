import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import '../core/theme.dart';
import '../services/history_service.dart';
import 'stats_screen.dart';

class HomeScreen extends StatelessWidget {
  final VoidCallback onStartProtecting;

  const HomeScreen({super.key, required this.onStartProtecting});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.background,
      body: CustomScrollView(
        slivers: [
          _HeroSliver(onStartProtecting: onStartProtecting),
          SliverPadding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            sliver: SliverList(
              delegate: SliverChildListDelegate([
                const SizedBox(height: 32),
                _StatCardsRow(),
                const SizedBox(height: 40),
                _SectionHeader(title: 'How It Works', icon: Icons.auto_awesome),
                const SizedBox(height: 20),
                const _HowItWorksTimeline(),
                const SizedBox(height: 40),
                _SectionHeader(title: 'Understanding Your Metrics', icon: Icons.insights),
                const SizedBox(height: 20),
                const _MetricCards(),
                const SizedBox(height: 40),
                _SectionHeader(title: 'Protection Levels', icon: Icons.security),
                const SizedBox(height: 20),
                const _ProtectionLevels(),
                const SizedBox(height: 100),
              ]),
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────── HERO ────────────────────────────
class _HeroSliver extends StatelessWidget {
  final VoidCallback onStartProtecting;
  const _HeroSliver({required this.onStartProtecting});

  @override
  Widget build(BuildContext context) {
    return SliverToBoxAdapter(
      child: Container(
        height: 380,
        decoration: const BoxDecoration(gradient: AppTheme.heroBg),
        child: Stack(
          children: [
            // Background decorative rings
            Positioned(
              top: -60, right: -60,
              child: _GlowRing(size: 220, color: AppTheme.cyanAccent.withOpacity(0.08)),
            ),
            Positioned(
              bottom: -40, left: -40,
              child: _GlowRing(size: 180, color: AppTheme.purpleAccent.withOpacity(0.07)),
            ),

            // Content
            SafeArea(
              child: Padding(
                padding: const EdgeInsets.all(28),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Top App Actions
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        // Logo
                        Row(children: [
                          Container(
                            padding: const EdgeInsets.all(10),
                            decoration: BoxDecoration(
                              color: AppTheme.cyanAccent.withOpacity(0.15),
                              borderRadius: BorderRadius.circular(14),
                              border: Border.all(color: AppTheme.cyanAccent.withOpacity(0.3)),
                            ),
                            child: const Icon(Icons.shield_outlined, color: AppTheme.cyanAccent, size: 28),
                          ),
                          const SizedBox(width: 12),
                          Text(
                            'FAKeless',
                            style: GoogleFonts.inter(
                              color: AppTheme.cyanAccent,
                              fontWeight: FontWeight.w800,
                              fontSize: 26,
                              letterSpacing: 1.5,
                            ),
                          ),
                        ]).animate().fadeIn(delay: 100.ms),
                        
                        // Usage Stats Icon
                        IconButton(
                          icon: const Icon(Icons.leaderboard_rounded, color: AppTheme.purpleAccent, size: 28),
                          onPressed: () {
                            Navigator.push(context, MaterialPageRoute(builder: (_) => const StatsScreen()));
                          },
                        ).animate().fadeIn(delay: 100.ms),
                      ],
                    ),

                    const Spacer(),

                    // Tagline
                    Text(
                      'Your Voice.\nProtected.',
                      style: GoogleFonts.inter(
                        color: AppTheme.textMain,
                        fontWeight: FontWeight.w800,
                        fontSize: 36,
                        height: 1.15,
                      ),
                    ).animate().fadeIn(delay: 200.ms).slideY(begin: 0.3),
                    const SizedBox(height: 12),
                    Text(
                      'AI-powered audio watermarking that stops\ndeepfake voice cloning in its tracks.',
                      style: GoogleFonts.inter(
                        color: AppTheme.textSecondary,
                        fontSize: 14,
                        height: 1.6,
                      ),
                    ).animate().fadeIn(delay: 350.ms),
                    const SizedBox(height: 28),

                    // CTA button
                    GestureDetector(
                      onTap: onStartProtecting,
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 14),
                        decoration: BoxDecoration(
                          gradient: AppTheme.cyanGradient,
                          borderRadius: BorderRadius.circular(30),
                          boxShadow: [BoxShadow(color: AppTheme.cyanAccent.withOpacity(0.4), blurRadius: 20, offset: const Offset(0, 6))],
                        ),
                        child: Row(mainAxisSize: MainAxisSize.min, children: [
                          const Icon(Icons.security, color: Colors.black, size: 18),
                          const SizedBox(width: 8),
                          Text('Start Protecting', style: GoogleFonts.inter(color: Colors.black, fontWeight: FontWeight.w700, fontSize: 15)),
                        ]),
                      ),
                    ).animate().fadeIn(delay: 450.ms).slideY(begin: 0.4),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _GlowRing extends StatelessWidget {
  final double size;
  final Color color;
  const _GlowRing({required this.size, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size, height: size,
      decoration: BoxDecoration(shape: BoxShape.circle, border: Border.all(color: color, width: 1.5)),
    );
  }
}

// ─────────────────────────────────────────── STAT CARDS ──────────────────────
class _StatCardsRow extends StatelessWidget {
  final HistoryService _historyService = HistoryService();

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<List<ProtectionSession>>(
      future: _historyService.loadAll(),
      builder: (context, snapshot) {
        final count = snapshot.hasData ? snapshot.data!.length : 0;
        
        final stats = [
          {'icon': Icons.layers, 'label': 'Protection\nLevels', 'value': '5 Scales'},
          {'icon': Icons.history, 'label': 'Total Audios\nProtected', 'value': '$count Files'},
          {'icon': Icons.cloud_done, 'label': 'Protection\nMode', 'value': 'Cloud + Local'},
        ];
        
        return Row(
          children: stats.asMap().entries.map((e) {
            final delay = (e.key * 100).ms;
            return Expanded(
              child: Padding(
                padding: EdgeInsets.only(right: e.key < stats.length - 1 ? 12 : 0),
                child: _StatCard(
                  icon: e.value['icon'] as IconData,
                  label: e.value['label'] as String,
                  value: e.value['value'] as String,
                ).animate().fadeIn(delay: delay).slideY(begin: 0.3, delay: delay),
              ),
            );
          }).toList(),
        );
      },
    );
  }
}

class _StatCard extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  const _StatCard({required this.icon, required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        gradient: AppTheme.cardGradient,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.cyanAccent.withOpacity(0.15)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: AppTheme.cyanAccent, size: 22),
          const SizedBox(height: 10),
          Text(value, style: GoogleFonts.inter(color: AppTheme.textMain, fontWeight: FontWeight.w700, fontSize: 13)),
          const SizedBox(height: 4),
          Text(label, style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 10, height: 1.4)),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────── SECTION HEADER ──────────────────
class _SectionHeader extends StatelessWidget {
  final String title;
  final IconData icon;
  const _SectionHeader({required this.title, required this.icon});

  @override
  Widget build(BuildContext context) {
    return Row(children: [
      Icon(icon, color: AppTheme.cyanAccent, size: 20),
      const SizedBox(width: 10),
      Text(title, style: GoogleFonts.inter(color: AppTheme.textMain, fontWeight: FontWeight.w700, fontSize: 17)),
    ]);
  }
}

// ─────────────────────────────────────────── HOW IT WORKS ────────────────────
class _HowItWorksTimeline extends StatelessWidget {
  const _HowItWorksTimeline();

  @override
  Widget build(BuildContext context) {
    final steps = [
      {'step': '01', 'title': 'Record or Upload', 'desc': 'Use the mic to record your voice, or upload any existing .WAV audio file from your device.', 'icon': Icons.mic_rounded},
      {'step': '02', 'title': 'Choose Protection', 'desc': 'Pick a protection scale (1–5) manually, or simply ask the FAKeless AI Guard to do it for you conversationally.', 'icon': Icons.tune},
      {'step': '03', 'title': 'Receive & Share', 'desc': 'Get your AI-protected audio instantly. The watermark is invisible to the ear but lethal to voice cloning AI.', 'icon': Icons.verified_user},
    ];

    return Theme(
      data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
      child: ExpansionTile(
        title: Text('View Process Timeline', style: GoogleFonts.inter(color: AppTheme.textMain, fontWeight: FontWeight.w600, fontSize: 15)),
        collapsedBackgroundColor: AppTheme.surfaceHigh,
        backgroundColor: AppTheme.surfaceHigh,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        collapsedShape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        iconColor: AppTheme.cyanAccent,
        childrenPadding: const EdgeInsets.only(left: 16, right: 16, bottom: 20, top: 10),
        children: steps.asMap().entries.map((e) {
          final isLast = e.key == steps.length - 1;
          return Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Column(children: [
                Container(
                  width: 32, height: 32,
                  decoration: BoxDecoration(shape: BoxShape.circle, color: AppTheme.cyanAccent.withOpacity(0.15)),
                  child: Center(child: Text(e.value['step'] as String, style: GoogleFonts.inter(color: AppTheme.cyanAccent, fontSize: 10, fontWeight: FontWeight.bold))),
                ),
                if (!isLast) Container(width: 1.5, height: 30, color: AppTheme.cyanAccent.withOpacity(0.2)),
              ]),
              const SizedBox(width: 16),
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.only(bottom: 16),
                  child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Row(children: [
                      Icon(e.value['icon'] as IconData, color: AppTheme.cyanAccent, size: 14),
                      const SizedBox(width: 6),
                      Text(e.value['title'] as String, style: GoogleFonts.inter(color: AppTheme.textMain, fontWeight: FontWeight.w600, fontSize: 13)),
                    ]),
                    const SizedBox(height: 4),
                    Text(e.value['desc'] as String, style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 12, height: 1.4)),
                  ]),
                ),
              ),
            ],
          );
        }).toList(),
      ),
    ); 
  }
}

// ─────────────────────────────────────────── METRIC CARDS ────────────────────
class _MetricCards extends StatelessWidget {
  const _MetricCards();

  @override
  Widget build(BuildContext context) {
    return Theme(
      data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
      child: ExpansionTile(
        title: Text('View Core Metrics', style: GoogleFonts.inter(color: AppTheme.textMain, fontWeight: FontWeight.w600, fontSize: 15)),
        collapsedBackgroundColor: AppTheme.surfaceHigh,
        backgroundColor: AppTheme.surfaceHigh,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        collapsedShape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        iconColor: AppTheme.cyanAccent,
        childrenPadding: const EdgeInsets.all(16),
        children: [
          _DenseMetricRow(
            color: AppTheme.cyanAccent,
            title: 'PESQ',
            subtitle: '(Perceptual Quality)',
            desc: 'Aim for 3.5+. The "clarity" rating.',
            icon: Icons.graphic_eq,
          ),
          const SizedBox(height: 12),
          _DenseMetricRow(
            color: AppTheme.orangeAccent,
            title: 'STOI',
            subtitle: '(Intelligibility)',
            desc: 'Aim for >0.75. Measures word understanding.',
            icon: Icons.record_voice_over,
          ),
        ],
      ),
    );
  }
}

class _DenseMetricRow extends StatelessWidget {
  final Color color;
  final String title, subtitle, desc;
  final IconData icon;

  const _DenseMetricRow({required this.color, required this.title, required this.subtitle, required this.desc, required this.icon});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(color: AppTheme.surface, borderRadius: BorderRadius.circular(12), border: Border.all(color: color.withOpacity(0.15))),
      child: Row(children: [
        Icon(icon, color: color, size: 20),
        const SizedBox(width: 12),
        Expanded(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(children: [
              Text(title, style: GoogleFonts.inter(color: color, fontWeight: FontWeight.w700, fontSize: 13)),
              const SizedBox(width: 4),
              Text(subtitle, style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 10)),
            ]),
            const SizedBox(height: 2),
            Text(desc, style: GoogleFonts.inter(color: AppTheme.textMain, fontSize: 11)),
          ]),
        ),
      ]),
    );
  }
}

// ─────────────────────────────────────────── PROTECTION LEVELS ───────────────
class _ProtectionLevels extends StatelessWidget {
  const _ProtectionLevels();

  @override
  Widget build(BuildContext context) {
    final levels = [
      {'scale': 1, 'label': 'Minimal', 'desc': 'Tiny watermark. Like whispering. Great quality, low protection.'},
      {'scale': 2, 'label': 'Light', 'desc': 'Good for casual sharing. Audio stays very clear.'},
      {'scale': 3, 'label': 'Standard', 'desc': 'The sweet spot. Balanced quality and strong protection.'},
      {'scale': 4, 'label': 'Strong', 'desc': 'Slight audio texture change. Powerful defense.'},
      {'scale': 5, 'label': 'Maximum', 'desc': 'Bank-grade protection. Stops the most advanced AI cloners.'},
    ];

    return Theme(
      data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
      child: ExpansionTile(
        title: Text('View Protection Details', style: GoogleFonts.inter(color: AppTheme.textMain, fontWeight: FontWeight.w600, fontSize: 15)),
        collapsedBackgroundColor: AppTheme.surfaceHigh,
        backgroundColor: AppTheme.surfaceHigh,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        collapsedShape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        iconColor: AppTheme.cyanAccent,
        childrenPadding: const EdgeInsets.only(left: 16, right: 16, bottom: 20, top: 10),
        children: levels.asMap().entries.map((e) {
          final level = e.value;
          final scale = level['scale'] as int;
          final progress = scale / 5.0;
          return Padding(
            padding: EdgeInsets.only(bottom: e.key < levels.length - 1 ? 16 : 0),
            child: Row(children: [
              SizedBox(
                width: 28,
                child: Text('$scale', style: GoogleFonts.inter(
                    color: AppTheme.cyanAccent, fontWeight: FontWeight.w800, fontSize: 16)),
              ),
              Expanded(
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Row(children: [
                    Text(level['label'] as String,
                        style: GoogleFonts.inter(color: AppTheme.textMain, fontWeight: FontWeight.w600, fontSize: 13)),
                    const Spacer(),
                    Text('${(progress * 100).toInt()}%',
                        style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 11)),
                  ]),
                  const SizedBox(height: 4),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value: progress,
                      backgroundColor: AppTheme.surface,
                      valueColor: AlwaysStoppedAnimation(
                        Color.lerp(AppTheme.successGreen, AppTheme.errorRed, progress * 0.8)!,
                      ),
                      minHeight: 4,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(level['desc'] as String,
                      style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 11, height: 1.4)),
                ]),
              ),
            ]),
          );
        }).toList(),
      ),
    );
  }
}
