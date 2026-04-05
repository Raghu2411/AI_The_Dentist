import 'package:ai_dental_studio/core/constants/strings.dart';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../../core/constants/app_colors.dart';
import '../../core/constants/dimens.dart';

class AppDialog extends StatelessWidget {
  final String title, descriptions, positiveButtonLabel, negativeButtonLabel;
  final VoidCallback? positiveButtonCallback;
  final Color? positiveButtonColor;
  final Color? positiveButtonTextColor;
  final Color? negativeButtonTextColor;
  final Color positiveButtonBorderColor;
  final double positiveButtonBorderWidth;
  final Color negativeButtonBorderColor;
  final double negativeButtonBorderWidth;
  final Color negativeButtonColor;

  const AppDialog({
    super.key,
    required this.title,
    required this.descriptions,
    this.positiveButtonLabel = '',
    this.negativeButtonLabel = '',
    this.positiveButtonCallback,
    this.negativeButtonColor = AppColors.error,
    this.positiveButtonColor,
    this.positiveButtonTextColor,
    this.negativeButtonTextColor,
    this.positiveButtonBorderColor = AppColors.borderColor,
    this.negativeButtonBorderColor = AppColors.gray200,
    this.negativeButtonBorderWidth = 1,
    this.positiveButtonBorderWidth = 1,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context).textTheme;
    return Dialog(
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.all(Radius.circular(Dimens.medium)),
      ),
      elevation: Dimens.zero,
      backgroundColor: Colors.transparent,
      insetPadding: const EdgeInsets.all(Dimens.medium),
      child: Container(
        padding: const EdgeInsets.all(Dimens.marginXLarge),
        decoration: const BoxDecoration(
          shape: BoxShape.rectangle,
          color: Colors.white,
          borderRadius: BorderRadius.all(Radius.circular(Dimens.medium)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            Text(
              title,
              style: theme.headlineSmall?.copyWith(fontWeight: FontWeight.w600),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: Dimens.marginMedium),
            Text(
              descriptions,
              style: theme.bodyLarge?.copyWith(
                fontWeight: FontWeight.w500,
                color: AppColors.labelColor,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: Dimens.marginLarge),

            Row(
              children: [
                Expanded(
                  child: TextButton(
                    style: ButtonStyle(
                      shape: WidgetStateProperty.all<RoundedRectangleBorder>(
                        RoundedRectangleBorder(
                          borderRadius: const BorderRadius.all(
                            Radius.circular(Dimens.medium),
                          ),
                          side: BorderSide(
                            color: AppColors.black,
                            width: positiveButtonBorderWidth,
                          ),
                        ),
                      ),
                      backgroundColor: WidgetStateProperty.all<Color>(
                        positiveButtonColor ?? AppColors.white,
                      ),
                    ),
                    onPressed: positiveButtonCallback,
                    child: Padding(
                      padding: const EdgeInsets.all(Dimens.medium),
                      child: Text(
                        positiveButtonLabel,
                        style: theme.labelLarge?.copyWith(
                          color: AppColors.black,
                        ),
                      ),
                    ),
                  ),
                ),
                SizedBox(width: Dimens.large),
                Expanded(
                  child: TextButton(
                    style: ButtonStyle(
                      shape: WidgetStateProperty.all<RoundedRectangleBorder>(
                        RoundedRectangleBorder(
                          borderRadius: const BorderRadius.all(
                            Radius.circular(Dimens.medium),
                          ),
                          side: BorderSide(
                            color: AppColors.error,
                            width: negativeButtonBorderWidth,
                          ),
                        ),
                      ),
                      backgroundColor: WidgetStateProperty.all<Color>(
                        negativeButtonColor,
                      ),
                    ),
                    onPressed: () {
                      context.pop();
                    },
                    child: Padding(
                      padding: const EdgeInsets.all(Dimens.medium),
                      child: Text(
                        Strings.cancel,
                        style: theme.labelLarge?.copyWith(
                          color: AppColors.white,
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
