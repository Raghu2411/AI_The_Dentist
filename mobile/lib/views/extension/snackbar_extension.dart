import 'package:ai_dental_studio/core/constants/app_colors.dart';
import 'package:flutter/material.dart';

extension SnackbarExtension on BuildContext {
  void showSnackBar(
    BuildContext context,
    String message, {
    Color backgroundColor = AppColors.green,
    Color textColor = AppColors.black,
  }) {
    final TextTheme textTheme = Theme.of(context).textTheme;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        backgroundColor: backgroundColor,
        content: Text(
          message,
          style: textTheme.bodySmall?.copyWith(color: textColor),
        ),
      ),
    );
    ;
  }
}
