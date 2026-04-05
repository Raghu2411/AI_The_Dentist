import 'dart:io';

import 'package:domain/domain.dart';
import 'package:domain/usecase/usecase.dart';
import 'package:injectable/injectable.dart';

@injectable
class DownloadFileFromUrlUseCase extends UseCase {
  final EssexDentalApiRepository _repository;

  DownloadFileFromUrlUseCase(this._repository);

  Future<Result> call(String fileUrl, Directory appDocDir) async =>
      await _repository.downloadFileFromUrl(fileUrl, appDocDir);
}
