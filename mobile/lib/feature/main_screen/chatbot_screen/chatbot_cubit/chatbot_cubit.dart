import 'package:domain/usecase/chatbot_communication_usecase.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:freezed_annotation/freezed_annotation.dart';
import 'package:injectable/injectable.dart';

part 'chatbot_cubit.freezed.dart';

@injectable
class ChatbotCubit extends Cubit<ChatbotState> {
  final ChatbotCommunicationUseCase _chatbotCommunicationUseCase;

  ChatbotCubit(this._chatbotCommunicationUseCase)
    : super(const _ChatbotState());

  void sendQuery(String query, String jobId) async {
    emit(state.copyWith(isLoading: true, isError: false));
    final cleanQuery = query.trim();
    final result = await _chatbotCommunicationUseCase(
      cleanQuery,
      jobId,
      state.history,
    );

    result.when(
      success: (message) {
        final updatedHistory = [
          ...state.history,
          {'role': 'user', 'content': cleanQuery},
          {'role': 'assistant', 'content': message.response},
        ];
        emit(
          state.copyWith(
            history: updatedHistory,
            isLoading: false,
            isError: false,
          ),
        );
      },
      failed: (error) {
        emit(
          state.copyWith(isLoading: false, isError: true, error: error.message),
        );
      },
    );
  }
}

@freezed
sealed class ChatbotState with _$ChatbotState {
  const factory ChatbotState({
    @Default([]) List<Map<String, String>> history,
    @Default(false) bool isLoading,
    @Default(false) bool isError,
    String? error,
  }) = _ChatbotState;
}
