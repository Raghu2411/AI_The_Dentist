import 'package:ai_dental_studio/core/constants/app_colors.dart';
import 'package:flutter/material.dart';

import '../dialogs/app_dialog.dart';

extension AppDialogExtension on BuildContext {
  Future openAppDialog(
    BuildContext context,
    String title,
    String description,
    String positiveButtonLabel,
    VoidCallback onPositiveButtonTap,
  ) {
    return showDialog(
      barrierDismissible: false,
      context: context,
      builder: (BuildContext context) {
        final primaryColor = AppColors.green;
        return AppDialog(
          title: title,
          descriptions: description,
          positiveButtonLabel: positiveButtonLabel,
          positiveButtonColor: AppColors.white,
          positiveButtonBorderColor: AppColors.gray200,
          positiveButtonTextColor: primaryColor,
          negativeButtonTextColor: primaryColor,
          positiveButtonCallback: onPositiveButtonTap,
        );
      },
    );
  }
}
