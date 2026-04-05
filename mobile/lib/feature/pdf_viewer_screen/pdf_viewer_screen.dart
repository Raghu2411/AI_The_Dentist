import 'dart:io';

import 'package:ai_dental_studio/core/constants/app_colors.dart';
import 'package:ai_dental_studio/views/app_pdf_viewer/app_pdf_viewer.dart';
import 'package:ai_dental_studio/views/app_progress_indicator.dart';
import 'package:ai_dental_studio/views/extension/share_file_bottom_sheet_extension.dart';
import 'package:ai_dental_studio/views/share_file/cubit/share_file_cubit.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:go_router/go_router.dart';

import '../../core/constants/strings.dart';
import '../../navigation/routes.dart';

class PdfViewerScreen extends StatelessWidget {
  final String pdfUrl;
  final String jobId;

  const PdfViewerScreen({super.key, required this.pdfUrl, required this.jobId});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        systemOverlayStyle: const SystemUiOverlayStyle(
          statusBarColor: Colors.black,
          statusBarIconBrightness: Brightness.light,
          statusBarBrightness: Brightness.dark,
        ),
        automaticallyImplyLeading: false,
        leading: Padding(
          padding: const EdgeInsets.only(left: 20),
          child: Row(
            children: [
              InkWell(
                onTap: () {
                  context.pop();
                },
                child: Icon(Icons.arrow_back, color: AppColors.white),
              ),
            ],
          ),
        ),
        actions: [
          IconButton(
            icon: Icon(
              Platform.isAndroid ? Icons.share : Icons.ios_share,
              color: Colors.white,
            ),
            tooltip: Strings.shareFile,
            onPressed: () {
              final cubit = context.read<ShareFileCubit>();
              context.openShareFileBottomSheet(context, pdfUrl, cubit);
            },
          ),
          IconButton(
            icon: Icon(Icons.chat, color: Colors.white),
            tooltip: Strings.chat,
            onPressed: () {
              if (context.canPop()) context.pop();
              if (context.canPop()) context.pop();
              context.go(Routes.chatbotScreenPath, extra: jobId);
            },
          ),
          const SizedBox(width: 8),
        ],
      ),
      backgroundColor: Colors.black,
      body: Center(
        child: BlocConsumer<ShareFileCubit, ShareFileState>(
          listener: (context, state) {
            state.whenOrNull(
              downLoaded: (filePath) {
                final cubit = context.read<ShareFileCubit>();
                context.pop();
                return cubit.shareFileAsAttachment(filePath);
              },
              error: (message) {
                context.pop();
              },
              fileShared: () {},
            );
          },
          builder: (context, state) {
            final isDownloading =
                state.whenOrNull(
                  loading: () => true,
                  downLoading: () => true,
                ) ??
                false;
            return Stack(
              children: [
                IgnorePointer(
                  ignoring: isDownloading,

                  child: AppPdfViewer(url: pdfUrl),
                ),

                if (isDownloading) AppProgressIndicator(),
              ],
            );
          },
        ),
      ),
    );
  }
}
