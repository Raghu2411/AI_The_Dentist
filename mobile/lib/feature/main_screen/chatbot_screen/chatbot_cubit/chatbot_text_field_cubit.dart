import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:freezed_annotation/freezed_annotation.dart';
import 'package:injectable/injectable.dart';

part 'chatbot_text_field_cubit.freezed.dart';

@injectable
class ChatbotTextFieldCubit extends Cubit<ChatbotTextFieldState> {
  ChatbotTextFieldCubit() : super(const _ChatbotTextFieldState());

  void checkIfFieldEmpty(String query) async {
    bool isEmptyText = query.isNotEmpty;
    emit(state.copyWith(isEmpty: isEmptyText));
  }
}

@freezed
sealed class ChatbotTextFieldState with _$ChatbotTextFieldState {
  const factory ChatbotTextFieldState({@Default(true) bool isEmpty}) =
      _ChatbotTextFieldState;
}
