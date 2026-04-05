import 'package:ai_dental_studio/core/constants/font_sizes.dart';
import 'package:ai_dental_studio/core/extension/bottomsheet_extension.dart';
import 'package:ai_dental_studio/views/extension/app_dialog_extension.dart';
import 'package:ai_dental_studio/views/extension/snackbar_extension.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:go_router/go_router.dart';

import '../../../core/constants/app_colors.dart';
import '../../../core/constants/dimens.dart';
import '../../../core/constants/strings.dart';
import '../../navigation/routes.dart';
import '../../views/app_pdf_viewer/app_pdf_viewer.dart';
import '../main_screen/home_screen/views/custom_grid_view.dart';
import '../main_screen/home_screen/views/model_dropdown.dart';
import '../main_screen/home_screen/views/select_image_bottom_sheet_view.dart';
import 'cubit/select_image_cubit.dart';
import 'cubit/select_model_cubit.dart';

class SelectRadioGraphScreen extends StatelessWidget {
  const SelectRadioGraphScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    List<String> selectedModel = [Strings.defaultModel];

    return Scaffold(
      appBar: AppBar(
        automaticallyImplyLeading: false,
        leading: Padding(
          padding: const EdgeInsets.only(left: 20),
          child: Row(
            children: [
              InkWell(
                onTap: () {
                  context.pop();
                },
                child: Icon(Icons.arrow_back),
              ),
            ],
          ),
        ),
      ),

      backgroundColor: Colors.white,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(Dimens.marginMedium),
          child: BlocConsumer<SelectImageCubit, SelectImageState>(
            listener: (context, state) {
              state.whenOrNull(
                imageAnalyzedSuccessfully: (analyses, apiUrls) {
                  context.openAppDialog(
                    context,
                    Strings.chat,
                    Strings.chatDescription,
                    Strings.yes,
                    () {
                      context.pop();
                      context.go(
                        Routes.chatbotScreenPath,
                        extra: analyses.jobId,
                      );
                    },
                  );
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
              return Column(
                children: [
                  Center(
                    child: state.maybeWhen(
                      selectedImages: (images) => images.isNotEmpty
                          ? Text(
                              Strings.selectedOPGImage,
                              style: textTheme.titleLarge,
                            )
                          : SizedBox.shrink(),
                      orElse: () => SizedBox.shrink(),
                    ),
                  ),
                  SizedBox(height: Dimens.marginMedium),
                  Expanded(
                    child: state.maybeWhen(
                      initial: () {
                        return Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text(
                              Strings.selectOPGImage,
                              style: textTheme.headlineSmall?.copyWith(
                                color: AppColors.black,
                              ),
                              textAlign: TextAlign.center,
                            ),
                          ],
                        );
                      },
                      selectedImages: (images) {
                        return SingleChildScrollView(
                          child: Column(
                            children: [
                              ModelDropdown(
                                onChanged: (selectedValue) {
                                  if (selectedValue != null) {
                                    context
                                        .read<SelectModelCubit>()
                                        .setSelectedModel(selectedValue);
                                    selectedModel = selectedValue;
                                  }
                                },
                              ),
                              CustomGridView(selectedImages: images, jobId: ''),
                            ],
                          ),
                        );
                      },
                      imageAnalyzedSuccessfully: (analyses, apiUrls) {
                        return SingleChildScrollView(
                          child: Column(
                            children: [
                              CustomGridView(
                                selectedImages: [],
                                predictedImage: apiUrls.predicted_image,
                                originalImage: apiUrls.original_image,
                                predictions: analyses.predictions,
                                jobId: analyses.jobId,
                              ),
                              InkWell(
                                onTap: () {
                                  context.push(
                                    Routes.pdfViewerScreenPath,
                                    extra: {
                                      'pdfUrl': apiUrls.pdf_report,
                                      'jobId': analyses.jobId,
                                    },
                                  );
                                },
                                child: Column(
                                  children: [
                                    Container(
                                      width: 60,
                                      height: 80,
                                      decoration: BoxDecoration(
                                        borderRadius: BorderRadius.circular(8),
                                        border: Border.all(
                                          color: Colors.grey.shade300,
                                        ),
                                      ),
                                      clipBehavior: Clip.antiAlias,
                                      child: AbsorbPointer(
                                        child: AppPdfViewer(
                                          url: apiUrls.pdf_report,
                                        ),
                                      ),
                                    ),
                                    SizedBox(height: 5),
                                    Text(
                                      Strings.analysesReportTapToView,
                                      style: textTheme.bodySmall?.copyWith(
                                        color: AppColors.black,
                                        fontSize: FontSizes.small,
                                      ),
                                      textAlign: TextAlign.center,
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        );
                      },
                      orElse: () => SizedBox.shrink(),
                    ),
                  ),
                  Padding(
                    padding: EdgeInsets.only(
                      top: Dimens.marginLarge,
                      left: Dimens.marginXLarge,
                      right: Dimens.marginXLarge,
                    ),
                    child: state.maybeWhen(
                      selectedImages: (images) {
                        return _elevatedButton(context, images, selectedModel);
                      },
                      submitting: () {
                        return ElevatedButton.icon(
                          onPressed: () {
                            openAttachDocumentsBottomSheet(context);
                          },
                          style: ElevatedButton.styleFrom(
                            minimumSize: const Size.fromHeight(50),
                          ),
                          label: Text(
                            Strings.processing,
                            style: textTheme.titleLarge?.copyWith(
                              color: AppColors.white,
                            ),
                          ),
                          icon: SizedBox(
                            height: Dimens.xLarge,
                            width: Dimens.xLarge,
                            child: CircularProgressIndicator(
                              color: Colors.green,
                            ),
                          ),
                        );
                      },

                      orElse: () => _elevatedButton(context, [], selectedModel),
                    ),
                  ),
                ],
              );
            },
          ),
        ),
      ),
    );
  }

  void openAttachDocumentsBottomSheet(BuildContext context) {
    context.openBottomSheet(
      SelectImageBottomSheetView(
        onSelectImagePressed: () {
          context.read<SelectImageCubit>().selectImage();
        },
      ),
    );
  }

  Widget _elevatedButton(
    BuildContext context,
    List<String> images,
    List<String> selectedModel,
  ) {
    final textTheme = Theme.of(context).textTheme;

    return Row(
      children: [
        Expanded(
          child: ElevatedButton(
            onPressed: () {
              openAttachDocumentsBottomSheet(context);
            },
            style: ElevatedButton.styleFrom(
              minimumSize: const Size.fromHeight(50),
            ),
            child: Text(
              images.isNotEmpty ? Strings.reselect : Strings.select,
              style: textTheme.titleLarge?.copyWith(color: AppColors.white),
            ),
          ),
        ),

        if (images.isNotEmpty) ...[
          SizedBox(width: Dimens.large),

          Expanded(
            child: ElevatedButton(
              onPressed: () {
                final selectedModels = context
                    .read<SelectModelCubit>()
                    .selectedModelValue;
                if (selectedModels.isEmpty) {
                  context.showSnackBar(
                    context,
                    Strings.pleaseSelectAtLeastOne,
                    backgroundColor: AppColors.error,
                    textColor: AppColors.white,
                  );
                } else {
                  context.read<SelectImageCubit>().submitImageForAnalyses(
                    selectedModels,
                  );
                }
              },
              style: ElevatedButton.styleFrom(
                minimumSize: const Size.fromHeight(50),
              ),
              child: Text(
                Strings.upload,
                style: textTheme.titleLarge?.copyWith(color: AppColors.white),
              ),
            ),
          ),
        ],
      ],
    );
  }
}
