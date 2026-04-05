import 'package:ai_dental_studio/core/extension/bottomsheet_extension.dart';
import 'package:ai_dental_studio/views/share_file/cubit/share_file_cubit.dart';
import 'package:flutter/material.dart';

import '../share_file_bottom_sheet.dart';

extension ShareFileBottomSheetExtension on BuildContext {
  void openShareFileBottomSheet(
    BuildContext context,
    String fileUrl,
    ShareFileCubit cubit,
  ) async {
    await context.openBottomSheet(
      ShareFileBottomSheet(
        onShareAsLinkPressed: () => cubit.shareAsLink(fileUrl),
        onShareAsAttachmentPressed: () => cubit.downloadFile(fileUrl),
      ),
    );
  }
}
