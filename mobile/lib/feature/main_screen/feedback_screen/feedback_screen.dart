import 'package:ai_dental_studio/feature/main_screen/feedback_screen/feedback_cubit/feedback_cubit.dart';
import 'package:ai_dental_studio/feature/main_screen/feedback_screen/views/feedback_view.dart';
import 'package:ai_dental_studio/views/empty_view.dart';
import 'package:ai_dental_studio/views/extension/snackbar_extension.dart';
import 'package:domain/model/analyses.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../../../core/constants/app_colors.dart';
import '../../../core/constants/font_sizes.dart';
import '../../../core/constants/strings.dart';

class FeedbackScreen extends StatelessWidget {
  final Analyses analyses;

  const FeedbackScreen({super.key, required this.analyses});

  @override
  Widget build(BuildContext context) {
    final TextTheme textTheme = Theme.of(context).textTheme;

    return Scaffold(
      backgroundColor: Colors.grey[100],
      appBar: AppBar(
        title: Text(
          Strings.feedback,
          style: textTheme.headlineSmall?.copyWith(
            color: AppColors.black,
            fontSize: FontSizes.xxMedium,
          ),
        ),
        centerTitle: true,
        backgroundColor: AppColors.white,
      ),
      body: SafeArea(
        child: GestureDetector(
          behavior: HitTestBehavior.translucent,
          onTap: () {
            FocusScope.of(context).unfocus();
          },
          child: BlocConsumer<FeedbackCubit, FeedbackState>(
            listener: (context, state) {
              state.whenOrNull(
                submitted: (message) {
                  context.showSnackBar(context, message);
                },
                error: (message) {
                  context.showSnackBar(
                    context,
                    message,
                    backgroundColor: AppColors.error,
                    textColor: AppColors.white,
                  );
                },
              );
            },
            builder: (context, state) {
              return analyses.jobId == ''
                  ? EmptyView(message: Strings.emptyFeedBAck)
                  : SingleChildScrollView(
                      child: Column(
                        children: [FeedbackView(analyses: analyses)],
                      ),
                    );
            },
          ),
        ),
      ),
    );
  }
}
