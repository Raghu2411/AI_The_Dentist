import 'dart:io';

import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../core/constants/app_colors.dart';
import '../core/constants/dimens.dart';
import '../core/constants/font_sizes.dart';
import '../core/constants/strings.dart';

class ShareFileBottomSheet extends StatelessWidget {
  final VoidCallback? onShareAsLinkPressed;
  final VoidCallback? onShareAsAttachmentPressed;

  const ShareFileBottomSheet({
    super.key,
    this.onShareAsLinkPressed,
    this.onShareAsAttachmentPressed,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context).textTheme;
    return SafeArea(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Padding(
            padding: const EdgeInsets.only(top: Dimens.large),
            child: InkWell(
              onTap: () {
                context.pop();
                onShareAsLinkPressed?.call();
              },
              child: Center(
                child: Text(
                  Strings.shareAsALink,
                  style: theme.titleMedium?.copyWith(
                    fontSize: FontSizes.xxxMedium,
                    fontWeight: FontWeight.w400,
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(height: Dimens.marginXLarge),
          InkWell(
            onTap: () {
              context.pop();
              onShareAsAttachmentPressed?.call();
            },
            child: Center(
              child: Text(
                Strings.shareAsAnAttachment,
                style: theme.titleMedium?.copyWith(
                  fontSize: FontSizes.xxxMedium,
                  fontWeight: FontWeight.w400,
                ),
              ),
            ),
          ),
          const SizedBox(height: Dimens.large),
          Divider(color: AppColors.black.withValues(alpha: 0.3)),
          InkWell(
            onTap: () {
              context.pop();
            },
            child: Padding(
              padding: EdgeInsets.only(
                top: Dimens.large,
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
