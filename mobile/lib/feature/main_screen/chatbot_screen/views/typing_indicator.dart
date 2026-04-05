import 'dart:math';

import 'package:ai_dental_studio/core/constants/dimens.dart';
import 'package:flutter/material.dart';

import '../../../../core/constants/app_colors.dart';

class TypingIndicator extends StatefulWidget {
  const TypingIndicator({super.key});

  @override
  State<TypingIndicator> createState() => _TypingIndicatorState();
}

class _TypingIndicatorState extends State<TypingIndicator>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(Dimens.medium),
      decoration: BoxDecoration(
        color: AppColors.black,
        borderRadius: BorderRadius.only(
          topLeft: Radius.circular(Dimens.borderMedium),
          topRight: const Radius.circular(Dimens.borderMedium),
          bottomLeft: const Radius.circular(Dimens.borderMedium),
          bottomRight: Radius.circular(Dimens.zero),
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: List.generate(3, (index) {
          final delay = index * 0.2;
          return AnimatedBuilder(
            animation: _controller,
            builder: (context, child) {
              final progress = (_controller.value + delay) % 1.0;
              // dots fade + scale sinusoidally
              final opacity = 0.3 + (sin(progress * pi) * 0.7);
              final scale = 0.8 + (sin(progress * pi) * 0.3);

              return Opacity(
                opacity: opacity,
                child: Transform.scale(
                  scale: scale,
                  child: Container(
                    margin: const EdgeInsets.symmetric(
                      horizontal: Dimens.progressStrokeWidth,
                    ),
                    width: Dimens.marginSmall,
                    height: Dimens.xLarge,
                    decoration: const BoxDecoration(
                      color: AppColors.gray200,
                      shape: BoxShape.circle,
                    ),
                  ),
                ),
              );
            },
          );
        }),
      ),
    );
  }
}
