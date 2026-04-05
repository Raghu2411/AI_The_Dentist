import 'package:domain/domain.dart';
import 'package:domain/usecase/usecase.dart';
import 'package:injectable/injectable.dart';

@injectable
class CheckOPGImageAnalysesUseCase extends UseCase {
  final EssexDentalApiRepository _repository;

  CheckOPGImageAnalysesUseCase(this._repository);

  Future<Result<Analyses>> call(
    String OPGImagePath,
    List<String> selectedModels,
  ) async {
    return _repository.checkPrediction(OPGImagePath, selectedModels);
  }
}
