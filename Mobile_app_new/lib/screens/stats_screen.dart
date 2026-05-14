import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';

import '../core/theme.dart';
import '../services/history_service.dart';

class StatsScreen extends StatefulWidget {
  const StatsScreen({super.key});

  @override
  State<StatsScreen> createState() => _StatsScreenState();
}

class _StatsScreenState extends State<StatsScreen> {
  final HistoryService _historyService = HistoryService();
  bool _isLoading = true;
  
  int _totalProtected = 0;
  int _totalSecondsSecured = 0;
  Map<int, int> _scaleBreakdown = {};

  @override
  void initState() {
    super.initState();
    _loadStats();
  }

  Future<void> _loadStats() async {
    final sessions = await _historyService.loadAll();
    
    // Calculate stats
    int total = sessions.length;
    // We don't save exact audio length to local db currently, so we use an average estimate per file (45 seconds) for gamification
    int seconds = total * 45; 
    
    Map<int, int> breakdown = {};
    for (var s in sessions) {
      breakdown[s.scale] = (breakdown[s.scale] ?? 0) + 1;
    }

    if (!mounted) return;
    setState(() {
      _totalProtected = total;
      _totalSecondsSecured = seconds;
      _scaleBreakdown = breakdown;
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Scaffold(
        backgroundColor: AppTheme.background,
        body: Center(child: CircularProgressIndicator(color: AppTheme.cyanAccent)),
      );
    }

    // Format time secured string
    String timeSecuredStr;
    if (_totalSecondsSecured < 60) {
      timeSecuredStr = '$_totalSecondsSecured sec';
    } else if (_totalSecondsSecured < 3600) {
      timeSecuredStr = '${(_totalSecondsSecured / 60).toStringAsFixed(1)} min';
    } else {
      timeSecuredStr = '${(_totalSecondsSecured / 3600).toStringAsFixed(1)} hrs';
    }

    return Scaffold(
      backgroundColor: AppTheme.background,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new, color: AppTheme.purpleAccent, size: 20),
          onPressed: () => Navigator.pop(context),
        ),
        title: Text('Usage Statistics',
            style: GoogleFonts.inter(
                color: AppTheme.purpleAccent,
                fontWeight: FontWeight.w700,
                fontSize: 18,
                letterSpacing: 1)),
      ),
      body: CustomScrollView(
        slivers: [
          SliverPadding(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
            sliver: SliverList(
              delegate: SliverChildListDelegate([
                
                // Top Global Stats
                Row(
                  children: [
                    Expanded(
                      child: _StatBox(
                        title: 'Total Protected',
                        val: _totalProtected.toString(),
                        icon: Icons.shield,
                        color: AppTheme.cyanAccent,
                      ).animate().fadeIn().slideY(begin: 0.1),
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: _StatBox(
                        title: 'Time Secured',
                        val: timeSecuredStr,
                        icon: Icons.access_time_filled,
                        color: AppTheme.orangeAccent,
                      ).animate().fadeIn(delay: 100.ms).slideY(begin: 0.1),
                    ),
                  ],
                ),

                const SizedBox(height: 32),
                
                // Title
                Row(children: [
                  const Icon(Icons.bar_chart, color: AppTheme.purpleAccent, size: 20),
                  const SizedBox(width: 10),
                  Text('Protection Scale Breakdown', style: GoogleFonts.inter(color: AppTheme.textMain, fontWeight: FontWeight.w700, fontSize: 16)),
                ]).animate().fadeIn(delay: 200.ms),
                const SizedBox(height: 16),

                // Breakdown list
                if (_scaleBreakdown.isEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 20),
                    child: Center(
                      child: Text('No data recorded yet. Protect some audio to see stats!', 
                        style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 13),
                        textAlign: TextAlign.center,
                      ),
                    ),
                  )
                else
                  ..._scaleBreakdown.entries.toList().map((e) {
                    final scale = e.key;
                    final count = e.value;
                    final double pct = _totalProtected == 0 ? 0 : count / _totalProtected;

                    return Container(
                      margin: const EdgeInsets.only(bottom: 12),
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: AppTheme.surfaceHigh,
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: AppTheme.textSecondary.withOpacity(0.1)),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Text('Scale $scale', style: GoogleFonts.inter(color: AppTheme.textMain, fontWeight: FontWeight.w600, fontSize: 14)),
                              Text('$count records', style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 12)),
                            ],
                          ),
                          const SizedBox(height: 8),
                          ClipRRect(
                            borderRadius: BorderRadius.circular(6),
                            child: Stack(
                              children: [
                                Container(height: 8, color: AppTheme.background),
                                FractionallySizedBox(
                                  widthFactor: pct,
                                  child: Container(
                                    height: 8,
                                    decoration: BoxDecoration(
                                      color: _getColorForScale(scale),
                                      borderRadius: BorderRadius.circular(6),
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ).animate().slideX(begin: -1.0, duration: 800.ms, curve: Curves.easeOut),
                        ],
                      ),
                    ).animate().fadeIn(delay: 300.ms);
                  }),
              ]),
            ),
          )
        ],
      )
    );
  }

  Color _getColorForScale(int scale) {
    if (scale <= 10) return AppTheme.cyanAccent;
    if (scale <= 50) return AppTheme.purpleAccent;
    return AppTheme.errorRed;
  }
}

class _StatBox extends StatelessWidget {
  final String title;
  final String val;
  final IconData icon;
  final Color color;

  const _StatBox({
    required this.title,
    required this.val,
    required this.icon,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: color.withOpacity(0.08),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: color, size: 28),
          const SizedBox(height: 12),
          Text(val, style: GoogleFonts.inter(color: AppTheme.textMain, fontWeight: FontWeight.w800, fontSize: 24)),
          const SizedBox(height: 4),
          Text(title, style: GoogleFonts.inter(color: AppTheme.textSecondary, fontSize: 12, fontWeight: FontWeight.w500)),
        ]
      ),
    );
  }
}
