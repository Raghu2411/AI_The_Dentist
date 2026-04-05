import 'package:flutter/material.dart';

import '../core/constants/app_colors.dart';
import '../core/constants/font_sizes.dart';

class EmptyView extends StatelessWidget {
  final String message;

  const EmptyView({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    final TextTheme textTheme = Theme.of(context).textTheme;

    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Center(
          child: Text(
            message,
            textAlign: TextAlign.center,
            style: textTheme.headlineSmall?.copyWith(
              color: AppColors.black,
              fontSize: FontSizes.xxMedium,
            ),
          ),
        ),
      ],
    );
  }
}
