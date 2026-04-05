import 'dart:io';

import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../../../../core/constants/app_colors.dart';
import '../../../../core/constants/dimens.dart';
import '../../../../core/constants/font_sizes.dart';
import '../../../../core/constants/strings.dart';

class SelectImageBottomSheetView extends StatelessWidget {
  final VoidCallback? onSelectImagePressed;
  final VoidCallback? onCancelPressed;

  const SelectImageBottomSheetView({
    super.key,
    this.onSelectImagePressed,
    this.onCancelPressed,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context).textTheme;
    return SafeArea(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        mainAxisSize: MainAxisSize.min,
        children: [
          const SizedBox(height: Dimens.marginMedium),
          Text(
            Strings.selectOptions,
            textAlign: TextAlign.center,
            style: theme.bodyLarge?.copyWith(color: AppColors.hintColor),
          ),
          const SizedBox(height: Dimens.marginMedium),
          InkWell(
            onTap: () {
              context.pop();
              onSelectImagePressed?.call();
            },
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(
                vertical: Dimens.marginMedium,
              ),
              child: Text(
                Strings.selectImage,
                textAlign: TextAlign.center,
                style: theme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w400,
                  fontSize: FontSizes.xxxMedium,
                ),
              ),
            ),
          ),
          const SizedBox(height: Dimens.medium),
          Divider(
            height: 0.2,
            thickness: 0.2,
            color: Theme.of(context).dividerColor,
          ),
          InkWell(
            onTap: () {
              context.pop();
              onCancelPressed?.call();
            },
            child: Padding(
              padding: EdgeInsets.only(
                top: Dimens.marginMedium,
                bottom: Platform.isIOS ? Dimens.zero : Dimens.xLarge,
              ),
              child: Center(
                child: Text(
                  Strings.cancel,
                  style: theme.titleMedium?.copyWith(
                    fontSize: FontSizes.xxxMedium,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
