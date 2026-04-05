import 'package:domain/domain.dart';
import 'package:domain/model/feedback.dart';
import 'package:domain/usecase/usecase.dart';
import 'package:injectable/injectable.dart';

@injectable
class SubmitFeedbackUseCase extends UseCase {
  final EssexDentalApiRepository _repository;

  SubmitFeedbackUseCase(this._repository);

  Future<Result<Feedback>> call(
    Analyses analyses,
    String name,
    String feedbackText,
  ) async {
    return _repository.submitFeedback(analyses, name, feedbackText);
  }
}
