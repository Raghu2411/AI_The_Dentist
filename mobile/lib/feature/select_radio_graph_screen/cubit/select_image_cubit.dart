import 'package:domain/domain.dart';
import 'package:domain/model/api_urls.dart';
import 'package:domain/usecase/check_opg_image_analyses_usecase.dart';
import 'package:freezed_annotation/freezed_annotation.dart';
import 'package:image_picker/image_picker.dart';
import 'package:injectable/injectable.dart';
import 'package:logger/logger.dart';

import '../../../../core/cubit/app_cubit.dart';

part 'select_image_cubit.freezed.dart';

@injectable
class SelectImageCubit extends AppCubit<SelectImageState> {
  final CheckOPGImageAnalysesUseCase _useCase;
  final List<String> selectedImages = [];
  final Logger _logger;

  SelectImageCubit(this._useCase, this._logger) : super(const _Initial());

  void selectImage() async {
    try {
      emit(const _AttachingImages());
      final ImagePicker imagePicker = ImagePicker();

      final XFile? pickedFile = await imagePicker.pickImage(
        source: ImageSource.gallery,
      );

      if (pickedFile != null) {
        selectedImages.clear();

        _logger.d('Selected image path: ${pickedFile.path}');
        selectedImages.add(pickedFile.path);

        emit(_SelectedImages(selectedImages));
      } else {
        if (selectedImages.isNotEmpty) {
          emit(_SelectedImages(selectedImages));
        } else {
          emit(const _Initial());
        }
      }
    } catch (e) {
      if (selectedImages.isNotEmpty) {
        emit(_SelectedImages(selectedImages));
      } else {
        emit(const _Initial());
      }
    }
  }

  void submitImageForAnalyses(List<String> selectedModels) async {
    emit(_Submitting());
    final response = await _useCase(selectedImages.first, selectedModels);

    response.when(
      success: (analyses) {
        emit(_ImageAnalyzedSuccessfully(analyses, analyses.apiUrls));
      },
      failed: (error) {
        _logger.d('Babar error: $error');
        emit(_Error(error.message));
      },
    );
  }
}

@freezed
class SelectImageState with _$SelectImageState {
  const factory SelectImageState.initial() = _Initial;

  const factory SelectImageState.fileTypeNotSupported(String message) =
      _FileTypeNotSupported;

  const factory SelectImageState.attachingImages() = _AttachingImages;

  const factory SelectImageState.selectedImages(
    List<String> selectedImagesPath,
  ) = _SelectedImages;

  const factory SelectImageState.submitting() = _Submitting;

  const factory SelectImageState.imageAnalyzedSuccessfully(
    Analyses analyses,
    ApiUrls apiUrls,
  ) = _ImageAnalyzedSuccessfully;

  const factory SelectImageState.error(String message) = _Error;

  const factory SelectImageState.previewImage(String documentPath) =
      _PreviewDocument;
}
