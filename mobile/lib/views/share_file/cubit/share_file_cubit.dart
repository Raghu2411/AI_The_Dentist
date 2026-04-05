import 'dart:io';

import 'package:ai_dental_studio/core/constants/strings.dart';
import 'package:domain/common/result.dart';
import 'package:domain/usecase/download_file_from_url_usecase.dart';
import 'package:freezed_annotation/freezed_annotation.dart';
import 'package:injectable/injectable.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

import '../../../core/cubit/app_cubit.dart';

part 'share_file_cubit.freezed.dart';

@injectable
class ShareFileCubit extends AppCubit<ShareFileState> {
  final DownloadFileFromUrlUseCase downloadFileFromUrlUseCase;

  ShareFileCubit(this.downloadFileFromUrlUseCase) : super(const _Initial());

  Future<void> shareAsLink(String fileUrl) async {
    final result = await SharePlus.instance.share(
      ShareParams(
        text: '${Strings.findAnalysesReport}\n$fileUrl',

        /// when sharing to email
        subject: Strings.dentalAIAnalysesReport,
      ),
    );
    checkShareResultStatus(result);
  }

  void checkShareResultStatus(ShareResult shareResultStatus) {
    if (shareResultStatus.status == ShareResultStatus.success) {
      emit(const _FileShared());
    }
  }

  Future<void> downloadFile(String fileUrl) async {
    if (fileUrl.isNotEmpty) {
      final Directory appDocDir = await getApplicationDocumentsDirectory();

      emit(const _DownLoading());
      final Result result = await downloadFileFromUrlUseCase(
        fileUrl,
        appDocDir,
      );
      result.when(
        success: (filePath) {
          emit(_DownLoaded(filePath));
        },
        failed: (error) {
          emit(_Error(error.message));
        },
      );
    } else {}
  }

  Future<void> shareFileAsAttachment(String fileUrl) async {
    final file = <XFile>[];
    file.add(XFile(fileUrl));
    final result = await SharePlus.instance.share(
      ShareParams(
        files: file,
        text: Strings.findAnalysesReport,
        subject: Strings.dentalAIAnalysesReport,
      ),
    );
    checkShareResultStatus(result);
  }
}

@freezed
sealed class ShareFileState with _$ShareFileState {
  const factory ShareFileState.initial() = _Initial;

  const factory ShareFileState.loading() = _Loading;

  const factory ShareFileState.error(String message) = _Error;

  const factory ShareFileState.downLoading() = _DownLoading;

  const factory ShareFileState.downLoaded(String filePath) = _DownLoaded;

  const factory ShareFileState.downLoadingError(String message) =
      _DownLoadingError;

  const factory ShareFileState.fileShared() = _FileShared;
}
