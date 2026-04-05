import 'package:ai_dental_studio/core/constants/font_sizes.dart';
import 'package:ai_dental_studio/feature/main_screen/feedback_screen/feedback_cubit/feedback_cubit.dart';
import 'package:domain/model/analyses.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../../../../core/constants/app_colors.dart';
import '../../../../core/constants/dimens.dart';
import '../../../../core/constants/strings.dart';

class FeedbackView extends StatefulWidget {
  final Analyses analyses;

  const FeedbackView({super.key, required this.analyses});

  @override
  State<FeedbackView> createState() => _FeedbackViewState();
}

class _FeedbackViewState extends State<FeedbackView> {
  final TextEditingController nameController = TextEditingController();
  final TextEditingController feedbackController = TextEditingController();
  final _formKey = GlobalKey<FormState>();

  @override
  Widget build(BuildContext context) {
    final TextTheme textTheme = Theme.of(context).textTheme;
    final cubit = context.read<FeedbackCubit>();

    return BlocBuilder<FeedbackCubit, FeedbackState>(
      builder: (context, state) {
        return state.maybeWhen(
          orElse: () {
            return Padding(
              padding: const EdgeInsets.all(Dimens.xLarge),
              child: Form(
                key: _formKey,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      Strings.name,
                      style: textTheme.bodyLarge?.copyWith(
                        color: AppColors.black,
                      ),
                    ),
                    const SizedBox(height: Dimens.marginSmall),
                    TextFormField(
                      controller: nameController,
                      decoration: InputDecoration(
                        hintText: Strings.enterYorName,
                        hintStyle: textTheme.bodyLarge?.copyWith(
                          color: AppColors.hintColor,
                          fontSize: FontSizes.xSmall,
                        ),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(
                            Dimens.borderMedium,
                          ),
                        ),
                      ),
                      validator: (value) => value == null || value.isEmpty
                          ? Strings.pleaseEnterName
                          : null,
                      textInputAction: TextInputAction.done,
                    ),
                    const SizedBox(height: Dimens.xLarge),
                    Text(
                      Strings.yourFeedback,
                      style: textTheme.bodyLarge?.copyWith(
                        color: AppColors.black,
                      ),
                    ),
                    const SizedBox(height: Dimens.marginSmall),
                    TextFormField(
                      controller: feedbackController,
                      maxLines: 4,
                      decoration: InputDecoration(
                        hintText: Strings.writeYourFeedback,
                        hintStyle: textTheme.bodyLarge?.copyWith(
                          color: AppColors.hintColor,
                          fontSize: FontSizes.xSmall,
                        ),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(
                            Dimens.borderMedium,
                          ),
                        ),
                      ),
                      validator: (value) => value == null || value.isEmpty
                          ? Strings.pleaseEnterFeedback
                          : null,
                      textInputAction: TextInputAction.done,
                    ),

                    const SizedBox(height: Dimens.margin40),
                    Padding(
                      padding: EdgeInsets.only(
                        left: Dimens.margin40,
                        right: Dimens.margin40,
                      ),
                      child: Center(
                        child: ElevatedButton.icon(
                          onPressed: _submitFeedback,
                          style: ElevatedButton.styleFrom(
                            padding: EdgeInsets.symmetric(
                              horizontal: Dimens.marginXxLarge,
                              vertical: Dimens.xxMedium,
                            ),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(
                                Dimens.borderMedium,
                              ),
                            ),
                          ),
                          label: Text(
                            Strings.submitFeedback,
                            style: textTheme.labelLarge?.copyWith(
                              color: AppColors.white,
                            ),
                            textAlign: TextAlign.center,
                          ),
                          icon: SizedBox(
                            height: Dimens.xLarge,
                            width: Dimens.xLarge,
                            child: cubit.isSubmitting
                                ? CircularProgressIndicator(color: Colors.green)
                                : SizedBox(),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            );
          },
        );
      },
    );
  }

  Future _submitFeedback() async {
    if (_formKey.currentState!.validate()) {
      final name = nameController.text.trim();
      final feedbackText = feedbackController.text.trim();
      final cubit = context.read<FeedbackCubit>();

      await cubit.submitFeedback(widget.analyses, name, feedbackText);
      // Here you can send the feedback to your server or API
      debugPrint('Name: $name');
      debugPrint('Feedback: $feedbackText');
      // Clear fields after submission
      nameController.clear();
      feedbackController.clear();
    }
  }
}
