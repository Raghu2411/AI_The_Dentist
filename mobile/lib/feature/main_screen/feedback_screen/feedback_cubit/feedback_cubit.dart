import 'package:ai_dental_studio/core/cubit/app_cubit.dart';
import 'package:domain/domain.dart';
import 'package:domain/usecase/submit_feedback_usecase.dart';
import 'package:freezed_annotation/freezed_annotation.dart';
import 'package:injectable/injectable.dart';

part 'feedback_cubit.freezed.dart';

@injectable
class FeedbackCubit extends AppCubit<FeedbackState> {
  final SubmitFeedbackUseCase _submitFeedbackUseCase;

  FeedbackCubit(this._submitFeedbackUseCase) : super(const _Initial());

  Future<void> submitFeedback(
    Analyses analyses,
    String name,
    String feedbackText,
  ) async {
    emit(const _Submitting());
    final result = await _submitFeedbackUseCase(analyses, name, feedbackText);

    result.when(
      success: (submitted) {
        emit(_Submitted(submitted.message));
      },
      failed: (error) {
        emit(_Error(error.message));
      },
    );
  }

  bool get isSubmitting => state is _Submitting;
}

@freezed
sealed class FeedbackState with _$FeedbackState {
  const factory FeedbackState.initial() = _Initial;

  const factory FeedbackState.submitting() = _Submitting;

  const factory FeedbackState.submitted(String message) = _Submitted;

  const factory FeedbackState.error(String errorMessage) = _Error;
}
