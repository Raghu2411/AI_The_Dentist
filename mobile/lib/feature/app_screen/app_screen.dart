import 'package:ai_dental_studio/core/constants/strings.dart';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../../core/constants/assets.dart';
import '../../navigation/routes.dart';

class AppScreen extends StatefulWidget {
  const AppScreen({super.key});

  @override
  State<AppScreen> createState() => _AppScreenState();
}

class _AppScreenState extends State<AppScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 1),
    )..repeat(reverse: true);

    _animation = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(parent: _controller, curve: Curves.easeInOut));
    // Navigate after 4 seconds
    Future.delayed(const Duration(seconds: 4), () {
      if (mounted) {
        // context.replace(Routes.mainScreenPath);
        context.replace(Routes.homeScreenPath);
      }
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;

    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Text(
              Strings.welcome,
              style: textTheme.headlineSmall,
              textAlign: TextAlign.center,
            ),
            FadeTransition(
              opacity: _animation,
              child: Image.asset(Assets.teethIcon, fit: BoxFit.contain),
            ),
          ],
        ),
      ),
    );
  }
}
